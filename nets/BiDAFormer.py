# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from .backbone import mit_b0


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class EdgeGatingUnit(nn.Module):
    """边缘门控单元，增强边界特征"""

    def __init__(self, in_channels):
        super(EdgeGatingUnit, self).__init__()
        self.conv_edge = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        edge_map = self.sigmoid(self.conv_edge(x))
        return x * edge_map  # 特征与边缘门控相乘


class CSCA(nn.Module):
    """修正后的跨尺度交叉注意力模块（维度匹配版本）"""

    def __init__(self, in_channels):
        super(CSCA, self).__init__()
        self.q_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # Query投影
        self.kv_conv = nn.Conv2d(in_channels, in_channels * 2, kernel_size=1)  # Key/Value联合投影
        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)  # 输出卷积

    def forward(self, x_high, x_mid):
        """
        x_high: 高层特征 (B, C, H, W)，如_c4（小分辨率，高语义）
        x_mid: 中层特征 (B, C, H, W)，如_c3（中分辨率，中语义）
        """
        B, C, H, W = x_high.shape  # H和W为c1的尺寸（如128x128）

        # Query投影：(B, C, H, W) -> (B, H*W, C)
        q = self.q_conv(x_high).view(B, C, -1).transpose(1, 2)  # (B, H*W, C)

        # Key/Value投影：(B, 2C, H, W) -> (B, 2C, H*W) -> 拆分为K/V (B, C, H*W)
        kv = self.kv_conv(x_mid).view(B, 2 * C, -1)
        k, v = kv.chunk(2, dim=1)  # k: (B, C, H*W), v: (B, C, H*W)

        # 计算注意力权重：(B, H*W, C) × (B, C, H*W) = (B, H*W, H*W)
        attn = self.softmax(torch.bmm(q, k) / (C ** 0.5))  # 缩放点积注意力

        # 特征融合：(B, H*W, H*W) × (B, C, H*W) = (B, H*W, C) -> (B, C, H, W)
        out = torch.bmm(attn, v.transpose(1, 2)).transpose(1, 2).view(B, C, H, W)
        return self.out_conv(out)


class LightClassAwareAttention(nn.Module):
    """彻底简化的类别感知注意力（通道注意力，无空间矩阵计算）"""
    def __init__(self, in_channels, class_idx=1):
        super().__init__()
        self.class_idx = class_idx
        # 通道注意力（轻量版）
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化，避免空间尺寸问题
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 4),  # 降维
            nn.ReLU(),
            nn.Linear(in_channels // 4, in_channels),
            nn.Sigmoid()
        )
        self.out_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x, seg_logits):
        b, c, h, w = x.size()

        # 1. 类别掩码（基于seg_logits）
        target_mask = (seg_logits.argmax(dim=1) == self.class_idx).float().unsqueeze(1)  # (b,1,h,w)
        x_masked = x * target_mask  # 仅关注目标类别区域

        # 2. 通道注意力（无空间矩阵计算，彻底避免尺寸问题）
        y = self.avg_pool(x_masked).view(b, c)  # (b,c) → 全局信息
        y = self.fc(y).view(b, c, 1, 1)  # (b,c,1,1) → 通道权重
        x_att = x * y  # 通道加权

        # 3. 输出与残差连接
        return self.out_conv(x_att) + x  # 残差连接保持特征稳定


class DualClassAwareAttention(nn.Module):
    """双类别注意力（保持软融合门控）"""
    def __init__(self, in_channels, num_classes=4, oil_idx=1, water_idx=3):
        super().__init__()
        self.caa_oil = LightClassAwareAttention(in_channels, class_idx=oil_idx)
        self.caa_water = LightClassAwareAttention(in_channels, class_idx=water_idx)
        self.oil_idx = oil_idx
        self.water_idx = water_idx
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, seg_logits):
        B, C, H, W = x.shape
        total_pixels = H * W

        # 计算类别占比（用于软融合权重）
        pred_mask = seg_logits.argmax(dim=1)
        oil_pixels = (pred_mask == self.oil_idx).float().sum(dim=(1, 2))
        water_pixels = (pred_mask == self.water_idx).float().sum(dim=(1, 2))
        oil_ratio = oil_pixels / total_pixels
        water_ratio = water_pixels / total_pixels

        # 软融合权重
        ratio_diff = (oil_ratio - water_ratio) / self.temperature
        oil_weight = torch.sigmoid(ratio_diff).view(B, 1, 1, 1)
        water_weight = 1 - oil_weight

        # 双分支计算（基于通道注意力，无尺寸冲突）
        feat_oil = self.caa_oil(x, seg_logits)
        feat_water = self.caa_water(x, seg_logits)
        return oil_weight * feat_oil + water_weight * feat_water


class BiDAFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, num_classes=4, in_channels=[32, 64, 160, 256],
                 embedding_dim=768, dropout_ratio=0.1,oil_idx=1, water_idx=3):
        super(BiDAFormerHead, self).__init__()
        c1_in, c2_in, c3_in, c4_in = in_channels

        # 基础特征投影（保持不变）
        self.linear_c4 = MLP(input_dim=c4_in, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in, embed_dim=embedding_dim)

        # 边缘增强与跨尺度融合（保持不变）
        self.egu_c4 = EdgeGatingUnit(embedding_dim)
        self.egu_c3 = EdgeGatingUnit(embedding_dim)
        self.csca = CSCA(in_channels=embedding_dim)

        # 双类别感知注意力（带软融合门控）
        self.dual_caa = DualClassAwareAttention(embedding_dim, oil_idx=oil_idx, water_idx=water_idx)
        # 初步分割生成
        self.linear_fuse_light = ConvModule(c1=embedding_dim * 4, c2=embedding_dim, k=1)
        self.seg_logits_conv = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)

        # 特征融合与输出（保持不变）
        self.linear_fuse = ConvModule(c1=embedding_dim * 4, c2=embedding_dim, k=1)
        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        n, _, h_c4, w_c4 = c4.shape
        # 校验c1尺寸（确保上采样目标正确）
        h_c1, w_c1 = c1.shape[2], c1.shape[3]

        # 1. 特征投影与上采样
        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, h_c4, w_c4)
        _c4 = F.interpolate(_c4, size=(h_c1, w_c1), mode='bilinear', align_corners=False)
        _c4 = self.egu_c4(_c4)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=(h_c1, w_c1), mode='bilinear', align_corners=False)
        _c3 = self.egu_c3(_c3)

        _c4_c3 = self.csca(_c4, _c3)  # 形状: (n, 256, h_c1, w_c1)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=(h_c1, w_c1), mode='bilinear', align_corners=False)
        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, h_c1, w_c1)

        # 2. 初步分割生成
        初步融合 = torch.cat([_c4_c3, _c3, _c2, _c1], dim=1)
        seg_logits = self.seg_logits_conv(self.linear_fuse_light(初步融合))

        # 3. 双类别注意力（带形状校验）
        _c4_c3_enhanced = self.dual_caa(_c4_c3, seg_logits)

        # 4. 最终输出
        _c = self.linear_fuse(torch.cat([_c4_c3_enhanced, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        return self.linear_pred(x)


class BiDAFormer(nn.Module):
    def __init__(self, num_classes=4, phi='b0', pretrained=False,oil_idx=1, water_idx=3):
        super(BiDAFormer, self).__init__()
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[phi]
        self.backbone = {
            'b0': mit_b0,
        }[phi](pretrained)
        self.embedding_dim = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[phi]
        self.decode_head = BiDAFormerHead(
            num_classes, self.in_channels, self.embedding_dim,
            oil_idx=oil_idx, water_idx=water_idx  # 传入类别索引
        )

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)

        x = self.backbone.forward(inputs)
        x = self.decode_head.forward(x)

        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x