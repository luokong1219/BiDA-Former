import os
from PIL import Image
import numpy as np
'''
预测产生的detection-results文件夹中的预测标签图像，是单通道像素0,1,2,3视觉上显示为全黑的图像
此脚本作用把单通道图像转化为彩色掩膜图像
以便与原数据集中的图像进行视觉上的对比
'''
# 定义颜色映射表
color_map = {
    0: (0, 0, 0),  # background
    1: (255, 0, 124),  # oil
    2: (255, 204, 51),  # others
    3: (51, 221, 255)  # water
}


def convert_to_color(input_dir, output_dir):
    """将单通道预测结果转换为彩色掩膜"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有预测图像文件
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    print(f"找到 {len(image_files)} 个预测图像")

    for image_file in image_files:
        # 读取单通道预测图像
        image_path = os.path.join(input_dir, image_file)
        pred_image = Image.open(image_path).convert('L')  # 确保为单通道灰度图
        pred_array = np.array(pred_image)

        # 创建彩色掩膜
        h, w = pred_array.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)

        # 根据颜色映射表填充颜色
        for class_id, color in color_map.items():
            mask = (pred_array == class_id)
            color_mask[mask] = color

        # 保存彩色掩膜
        color_image = Image.fromarray(color_mask)
        output_path = os.path.join(output_dir, image_file)
        color_image.save(output_path)

        print(f"已转换: {image_file} -> {output_path}")

    print(f"转换完成，彩色掩膜保存在: {output_dir}")


if __name__ == "__main__":
    # 当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 输入目录（detection-results文件夹）
    input_dir = os.path.join(script_dir, "detection-results")

    # 输出目录（彩色结果文件夹）
    output_dir = os.path.join(script_dir, "color-detection-results")

    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误: 找不到输入目录 '{input_dir}'")
    else:
        convert_to_color(input_dir, output_dir)