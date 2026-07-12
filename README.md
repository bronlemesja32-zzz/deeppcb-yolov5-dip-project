# 基于 YOLOv5s 的 PCB 表面缺陷检测与图像处理辅助分析

本项目是《数字图像处理》课程大作业，围绕 DeepPCB 数据集完成 PCB 表面缺陷检测、传统图像处理候选区域生成及结果分析。

## 方法

- YOLOv5s：检测 open、short、mousebite、spur、copper、pin_hole 六类缺陷。
- 传统方法：利用配准模板进行图像差分，并结合 Otsu 阈值、形态学处理和连通域分析生成类别无关候选区域。

## 主要结果

- YOLOv5s official test：P=0.951，R=0.944，mAP50=0.968，mAP50:95=0.594。
- 传统方法：IoU=0.33 时 F1=0.658。

## 目录说明

```text
github_release/
├── report/                 # 最终 PDF、LaTeX 源文件、引用图片与表格
├── scripts/                # 数据检查、转换、自检、可视化和传统基线脚本
├── configs/                # DeepPCB 的 YOLOv5 数据配置
├── logs/                   # 关键运行与一致性检查摘要
├── outputs/ip_baseline/    # 传统基线汇总指标与 PAD 标定表
├── README.md
├── RELEASE_CHECKLIST.md
└── .gitignore
```

## 数据说明

完整 DeepPCB 图片数据不随本仓库提供，请从 [DeepPCB 官方仓库](https://github.com/tangsanli5201/DeepPCB) 获取。数据目录已由 `.gitignore` 排除。

## 权重说明

训练权重 `best.pt`、`last.pt` 不随本仓库提供，所有 `*.pt` 和 `*.pth` 文件均已由 `.gitignore` 排除。

## 复现说明

仓库中的脚本用于数据检查与转换、转换结果自检、标注可视化、传统图像处理基线和 PAD 标定。YOLOv5s 训练与测试需另行准备 [YOLOv5 官方仓库](https://github.com/ultralytics/yolov5) 和完整 DeepPCB 数据，并按本机数据位置更新 `configs/deeppcb.yaml` 中的 `path`。

报告可在 `report/` 目录中使用 XeLaTeX 和 BibTeX 编译。  

## 课程说明

本仓库仅用于整理和展示课程作业材料，不声称方法创新、SOTA 或工业部署能力。
