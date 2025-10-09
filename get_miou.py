import os

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

from bidaformer import BiDAFormer_Segmentation
from utils.utils_metrics import compute_mIoU, show_results

'''
此脚本是用来进行测试集性能指标评估，主要是使用训练好的权重对测试集进行一系列评估
结果会保存在miou_out文件夹中，会生成各个类别的iou，miou，recall，f1分数等
还会生成整个测试集对应的预测单通道图像，保存在detection-results文件夹中
进行指标评估需要注意以下几点：
1、该文件生成的图为灰度图，因为值比较小，按照PNG形式的图看是没有显示效果的，所以看到近似全黑的图是正常的。
2、该文件计算的是验证集的miou，当前该库将测试集当作验证集使用，不单独划分测试集
'''


if __name__ == "__main__":

    miou_mode       = 0
    num_classes     = 4
    name_classes    = ["background", "oil", "others", "water"]
    model_path = "train_result/best_epoch_weights.pth"  # 使用你训练好的最佳权重

    Data_path  = 'OilSpillDatasets'

    test_txt_path = os.path.join(Data_path, "VOC2007/ImageSets/Segmentation/test.txt")

    # 创建输出目录
    miou_out_path = "miou_out"
    pred_dir = os.path.join(miou_out_path, 'detection-results')
    os.makedirs(pred_dir, exist_ok=True)

    # 读取测试集图像ID
    with open(test_txt_path, 'r', encoding='utf-8') as f:  # 显式指定文件读取编码
        image_ids = [line.strip() for line in f.readlines()]
    gt_dir = os.path.join(Data_path, "VOC2007/SegmentationClass/")

    pred_metrics = {}  # 保存逐像素预测指标
    if miou_mode == 0 or miou_mode == 1:
        print("Load model.")
        # 初始化模型并加载自定义权重
        bidaformer = BiDAFormer_Segmentation(
            model_path=model_path,
            num_classes=4,  # 必须与训练时一致
            phi='b0',  # 与训练时使用的主干网络一致（如 b0）
            input_shape=[512, 512],
            cuda=True
        )
        print("Load model done.")

        print("Generating predictions...")
        all_f1 = []
        all_acc = []
        for image_id in tqdm(image_ids):
            img_path = os.path.join(Data_path, f"VOC2007/JPEGImages/{image_id}.jpg")
            gt_path = os.path.join(gt_dir, f"{image_id}.png")
            if not os.path.exists(img_path) or not os.path.exists(gt_path):
                continue

            # 预测并保存
            img = Image.open(img_path)
            pred_png = bidaformer.get_miou_png(img)
            pred_png.save(os.path.join(pred_dir, f"{image_id}.png"))

            # 计算单图指标
            gt_arr = np.array(Image.open(gt_path)).flatten()
            pred_arr = np.array(pred_png).flatten()
            valid_idx = gt_arr < num_classes  # 过滤忽略类
            if valid_idx.sum() == 0:
                continue
            valid_gt = gt_arr[valid_idx]
            valid_pred = pred_arr[valid_idx]

            f1 = f1_score(valid_gt, valid_pred, average='weighted')
            acc = accuracy_score(valid_gt, valid_pred)
            all_f1.append(f1)
            all_acc.append(acc)

            # 释放内存
            del valid_gt, valid_pred, gt_arr, pred_arr, img, pred_png
            torch.cuda.empty_cache()

        # 保存逐像素指标
        if all_f1 and all_acc:
            pred_metrics["mean_f1"] = np.mean(all_f1)
            pred_metrics["mean_acc"] = np.mean(all_acc)
        print("Prediction done.")


    miou_metrics = {}  # 保存混淆矩阵推导指标
    if miou_mode == 0 or miou_mode == 2:
        print("Computing mIoU...")
        hist, IoUs, Recalls, Precisions = compute_mIoU(
            gt_dir, pred_dir, image_ids, num_classes, name_classes
        )

        # 全局指标（含背景）
        global_PA = np.diag(hist).sum() / hist.sum() if hist.sum() != 0 else 0.0

        # 有效类别（排除背景）
        valid_classes = [1, 2, 3]  # oil, others, water
        class_names = ["oil", "others", "water"]
        class_metrics = {}

        for idx, cn in zip(valid_classes, class_names):
            iou = IoUs[idx] if idx < len(IoUs) else 0.0
            recall = Recalls[idx] if idx < len(Recalls) else 0.0
            prec = Precisions[idx] if idx < len(Precisions) else 0.0

            # 计算F1（避免除零）
            if prec + recall < 1e-8:
                f1 = 0.0
            else:
                f1 = 2 * prec * recall / (prec + recall)

            class_metrics[cn] = {
                "IoU": iou,
                "Recall": recall,
                "Precision": prec,
                "F1": f1
            }

        # 类别平均指标
        mIoU = np.mean([class_metrics[cn]["IoU"] for cn in class_names])
        mean_F1 = np.mean([class_metrics[cn]["F1"] for cn in class_names])
        miou_metrics.update({
            "global_PA": global_PA,
            "mIoU": mIoU,
            "mean_F1": mean_F1,
            "class": class_metrics,
            "IoUs": IoUs
        })


        # 绘制IoU柱状图
        plt.figure(figsize=(12, 6))
        iou_percents = [miou_metrics["IoUs"][i] * 100 for i in valid_classes]
        plt.bar(class_names, iou_percents, color=['#FF9999', '#99FF99', '#9999FF'])
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('IoU (%)')
        plt.title('Oil、Others、Water IoU Distribution')
        for x, y in zip(class_names, iou_percents):
            plt.text(x, y + 1, f'{y:.2f}%', ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(miou_out_path, 'class_iou.png'))
        plt.close()
        print("mIoU computation done.")

    # ---------------------------
    # 3. 统一写入metrics.txt（解决乱码核心：utf-8-sig）
    # ---------------------------
    metric_lines = []
    if miou_mode == 0 or miou_mode == 1:
        if "mean_f1" in pred_metrics:
            metric_lines.extend([
                "【Pixel-wise Prediction Metrics】",
                f"Weighted F1-Score (excluding ignore classes): {pred_metrics['mean_f1']:.4f}",
                f"Accuracy (excluding ignore classes): {pred_metrics['mean_acc']:.4f}",
                ""
            ])
        else:
            metric_lines.append("Warning: No valid prediction data, unable to compute pixel-wise metrics!\n")

    if miou_mode == 0 or miou_mode == 2:
        if miou_metrics:
            metric_lines.extend([
                "【Global and Class-level Metrics】",
                f"Global PA (including background, all pixels): {miou_metrics['global_PA']:.4f}",
                f"mIoU (average of oil, others, water): {miou_metrics['mIoU']:.4f}",
                f"Mean F1 (average of oil, others, water): {miou_metrics['mean_F1']:.4f}",
                ""
            ])
            for cn in class_names:
                m = miou_metrics["class"][cn]
                metric_lines.extend([
                    f"=== {cn.upper()} Metrics ===",
                    f"IoU: {m['IoU']:.4f} → Percentage: {m['IoU'] * 100:.2f}%",
                    f"Recall: {m['Recall']:.4f} → Percentage: {m['Recall'] * 100:.2f}%",
                    f"Precision: {m['Precision']:.4f} → Percentage: {m['Precision'] * 100:.2f}%",
                    f"F1: {m['F1']:.4f} → Percentage: {m['F1'] * 100:.2f}%",
                    ""
                ])
            metric_lines.append("【IoU per Class (including background)】")
            for i, cn in enumerate(name_classes):  # 使用原始英文类别名
                iou_pct = miou_metrics["IoUs"][i] * 100 if i < len(miou_metrics["IoUs"]) else 0.0
                metric_lines.append(f"{cn}: {iou_pct:.2f}%")
            overall_mIoU = np.nanmean(miou_metrics["IoUs"]) * 100 if len(miou_metrics["IoUs"]) > 0 else 0.0
            metric_lines.append(f"mIoU (all classes, including background): {overall_mIoU:.2f}%")
        else:
            metric_lines.append("Warning: No valid confusion matrix, unable to compute mIoU metrics!\n")

    if metric_lines:
        metrics_path = os.path.join(miou_out_path, "metrics.txt")
        with open(metrics_path, 'w', encoding='utf-8') as f:  # 可以使用普通utf-8，不含中文
            f.write('\n'.join(metric_lines))
        print(f"Metrics saved to: {metrics_path} (Encoding: UTF-8)")
    else:
        print("No metrics to save.")

    # 在写入文件后再绘制图表，确保IoUs数组有效
    if miou_mode == 0 or miou_mode == 2:
        if miou_metrics and "IoUs" in miou_metrics and len(miou_metrics["IoUs"]) > 0:
            # 绘制IoU柱状图（仅绘制有效类别）
            plt.figure(figsize=(12, 6))
            iou_percents = [miou_metrics["IoUs"][i] * 100 for i in valid_classes]
            plt.bar(class_names, iou_percents, color=['#FF9999', '#99FF99', '#9999FF'])
            plt.xticks(rotation=45, ha='right')
            plt.ylabel('IoU (%)')
            plt.title('Oil、Others、Water IoU Distribution')
            for x, y in zip(class_names, iou_percents):
                plt.text(x, y + 1, f'{y:.2f}%', ha='center')
            plt.tight_layout()
            plt.savefig(os.path.join(miou_out_path, 'class_iou.png'))
            plt.close()
            print("IoU柱状图已保存")
        else:
            print("警告：无有效IoU数据，无法生成柱状图")