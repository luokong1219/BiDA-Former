import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from bidaformer import BiDAFormer_Segmentation
from utils.utils_metrics import compute_mIoU  # 需返回混淆矩阵`hist`

if __name__ == "__main__":
    # -------------------------- 核心参数配置 --------------------------
    miou_mode = 0  # 0: 预测+计算指标 | 1: 仅预测 | 2: 仅计算指标
    num_classes = 4  # 总类别（0=背景, 1=oil, 2=others, 3=water）
    valid_class_ids = [1, 2, 3]  # 有效类别（仅oil、others、water）
    valid_class_names = ["oil", "others", "water"]  # 对应名称
    model_path = "train_result/best_epoch_weights.pth"  # 模型权重
    Data_path = "OilSpillDatasets"  # 数据集根目录
    output_dir = "miou_out"  # 结果目录
    metrics_file = os.path.join(output_dir, "overall_metrics.txt")  # 指标文件
    pred_dir = os.path.join(output_dir, "detection-results")  # 预测结果目录
    # -----------------------------------------------------------------

    # 创建目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    # 读取测试集ID
    test_txt = os.path.join(Data_path, "VOC2007/ImageSets/Segmentation/test.txt")
    with open(test_txt, "r", encoding="utf-8") as f:
        image_ids = [line.strip() for line in f.readlines()]
    gt_dir = os.path.join(Data_path, "VOC2007/SegmentationClass/")  # 真实标签目录

    # -------------------------- 步骤1：生成预测结果 --------------------------
    if miou_mode == 0 or miou_mode == 1:
        print("加载模型...")
        bidaformer = BiDAFormer_Segmentation(
            model_path=model_path,
            num_classes=num_classes,
            phi="b0",
            input_shape=[512, 512],
            cuda=True
        )
        print("模型加载完成，开始预测...")

        for image_id in tqdm(image_ids, desc="生成预测结果"):
            img_path = os.path.join(Data_path, f"VOC2007/JPEGImages/{image_id}.jpg")
            gt_path = os.path.join(gt_dir, f"{image_id}.png")
            if not os.path.exists(img_path) or not os.path.exists(gt_path):
                continue

            # 预测并保存单通道结果
            img = Image.open(img_path)
            pred_png = bidaformer.get_miou_png(img)
            pred_png.save(os.path.join(pred_dir, f"{image_id}.png"))

            # 释放内存
            del img, pred_png
            torch.cuda.empty_cache()
        print("预测结果保存完成！")

    # -------------------------- 步骤2：计算整体性能指标（严格按公式） --------------------------
    if miou_mode == 0 or miou_mode == 2:
        print("计算性能指标（严格按公式，仅含oil、others、water）...")
        # 完整类别名（供compute_mIoU内部使用）
        name_classes = ["background", "oil", "others", "water"]

        # 获取混淆矩阵（hist: 行=真实标签，列=预测标签）
        hist, _, _, _ = compute_mIoU(
            gt_dir, pred_dir, image_ids, num_classes, name_classes
        )

        total_pixels = hist.sum()  # 测试集总像素数（含背景）
        total_TP, total_FP, total_FN, total_TN = 0, 0, 0, 0

        for c in valid_class_ids:
            # 1. 真正例（TP）：真实=c，预测=c
            TP = hist[c, c] if (c < hist.shape[0] and c < hist.shape[1]) else 0

            # 2. 假正例（FP）：真实≠c，预测=c → 列c总和 - TP
            FP = np.sum(hist[:, c]) - TP if c < hist.shape[1] else 0

            # 3. 假负例（FN）：真实=c，预测≠c → 行c总和 - TP
            FN = np.sum(hist[c, :]) - TP if c < hist.shape[0] else 0

            # 4. 真负例（TN）：总像素 - (TP + FP + FN)
            TN = total_pixels - (TP + FP + FN) if total_pixels != 0 else 0

            # 累加指标
            total_TP += TP
            total_FP += FP
            total_FN += FN
            total_TN += TN

        # -------------------------- 按公式计算指标 --------------------------
        # 1. Accuracy = (TP + TN) / (TP + TN + FP + FN)
        denominator_acc = total_TP + total_TN + total_FP + total_FN
        accuracy = (total_TP + total_TN) / denominator_acc if denominator_acc != 0 else 0.0

        # 2. Precision = TP / (TP + FP)
        denominator_precision = total_TP + total_FP
        precision = total_TP / denominator_precision if denominator_precision != 0 else 0.0

        # 3. Recall = TP / (TP + FN)
        denominator_recall = total_TP + total_FN
        recall = total_TP / denominator_recall if denominator_recall != 0 else 0.0

        # 4. F1 = 2*Precision*Recall / (Precision + Recall)
        if precision + recall < 1e-8:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        # 5. mIoU（保持您要求的公式：ΣTP / (ΣTP + ΣFP + ΣFN)）
        denominator_miou = total_TP + total_FP + total_FN
        mIoU = total_TP / denominator_miou if denominator_miou != 0 else 0.0

        # -------------------------- 保存结果到文件 --------------------------
        with open(metrics_file, "w", encoding="utf-8") as f:
            f.write("### 语义分割整体性能指标（严格按公式，仅含 oil、others、water）\n")
            f.write("=" * 60 + "\n")
            f.write(f"mIoU: {mIoU:.6f} （公式：ΣTP / (ΣTP + ΣFP + ΣFN)）\n")
            f.write(f"Accuracy: {accuracy:.6f} （公式：(TP+TN)/(TP+TN+FP+FN)）\n")
            f.write(f"Precision: {precision:.6f} （公式：TP/(TP+FP)）\n")
            f.write(f"Recall: {recall:.6f} （公式：TP/(TP+FN)）\n")
            f.write(f"F1-Score: {f1:.6f} （公式：2*P*R/(P+R)）\n")
            f.write("=" * 60 + "\n")
            f.write(f"测试样本数：{len(image_ids)}\n")
            f.write(f"总像素数（含背景）：{total_pixels}\n")
            f.write(f"有效类累计TP：{total_TP} | FP：{total_FP} | FN：{total_FN} | TN：{total_TN}\n")

        print(f"指标已保存至：{metrics_file}")
        print("性能指标计算完成（严格符合公式定义）！")