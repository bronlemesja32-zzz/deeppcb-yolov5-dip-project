# Step 5 官方 test split 最终评估报告

> 状态: **评估完成, 核验通过, 已停止。** 未安装/升级/卸载任何包, 未重新训练,
> 未修改数据集(评估首次扫描 test split 时 yolov5 自动生成 labels/test.cache 索引,
> 与 train/val.cache 同性质, 不改动任何图像与标签; 评估后重跑 03_verify_dataset.py 全部通过),
> 未覆盖正式训练目录(runs/train/deeppcb_yolov5s_50ep 原样, val.py 输出在 runs/val/ 下)。
> 所有数字来自 2026-07-12 实际运行输出。

## 1. 环境(与 Step 3/4 同一解释器, 运行横幅原文)

```
YOLOv5  2885cb4 Python-3.12.13 torch-2.7.1+cu128 CUDA:0 (NVIDIA GeForce RTX 5080 Laptop GPU, 16303MiB)
```

- python.exe: `C:\Users\qintx\Desktop\dip-2022-spring-master\dip-2022-spring-master\LAB\project\DocRescue\.venv\Scripts\python.exe`
- 权重: `runs/train/deeppcb_yolov5s_50ep/weights/best.pt`(Step 4 产物, 对应 epoch 38)
- 数据: `pcb_project/datasets/deeppcb/deeppcb.yaml`

## 2. 实际执行命令

工作目录 `C:\Users\qintx\Desktop\clauded\yolov5`:

```
python val.py --weights runs/train/deeppcb_yolov5s_50ep/weights/best.pt --data ../pcb_project/datasets/deeppcb/deeppcb.yaml --img 640 --batch 16 --task test --name deeppcb_yolov5s_50ep_test --save-txt --save-conf
```

关键生效参数(日志回显): `task=test, conf_thres=0.001, iou_thres=0.6, max_det=300,
half=False, save_txt=True, save_conf=True`; 退出码 0。

## 3. 核验清单(用户 10 项, 逐项实测)

| # | 核验项 | 结果 |
|---|---|---|
| 1 | 确实读取 test split | 是。日志回显 `task=test`, 且扫描行为 `test: New cache created: ...datasets\deeppcb\labels\test.cache`(扫描的是 labels/test) |
| 2 | test 图像数 = 500 | 是(指标行 Images=500; 数据目录实数 500 jpg/500 txt) |
| 3 | test 缺陷框数 = 3140 | 是(指标行 Instances=3140, 与 Step 2 转换报告的 test 框数完全一致) |
| 4 | runs/val/deeppcb_yolov5s_50ep_test/ 存在 | True |
| 5 | confusion_matrix / PR / P / R / F1_curve.png | 5 个全部存在 |
| 6 | 预测 txt 生成 | 是: labels/ 下 **500 个 txt**(每图一个, 格式 class xc yc w h conf, 见下方样例) |
| 7 | test 总体指标 | 见第 4 节 |
| 8 | 六类分项指标 | 见第 4 节 |
| 9 | 预测可视化 | val_batch0/1/2_pred.jpg(共 48 张 test 图的预测拼图, 含对应 _labels.jpg 真值拼图); 抽视 val_batch0_pred.jpg 确认框贴合缺陷、置信度多为 0.8–0.9、六类均出现 |
| 10 | 完成后停止 | 已停止, 未做后续实验, 未写 LaTeX |

预测 txt 样例(`labels/00041200.txt` 前 3 行, class xc yc w h conf):
```
4 0.38195 0.508901 0.0463638 0.0627203 0.927031
5 0.64743 0.608326 0.0668864 0.0447409 0.913948
3 0.594405 0.497349 0.0528337 0.0455523 0.910872
```

## 4. 官方 test split 指标(500 图, 3140 框, best.pt 融合层)

| class | Images | Instances | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| **all** | 500 | 3140 | **0.951** | **0.944** | **0.968** | **0.594** |
| open | 500 | 659 | 0.973 | 0.973 | 0.976 | 0.547 |
| short | 500 | 478 | 0.936 | 0.863 | 0.912 | 0.482 |
| mousebite | 500 | 586 | 0.969 | 0.951 | 0.981 | 0.572 |
| spur | 500 | 483 | 0.985 | 0.927 | 0.970 | 0.606 |
| copper | 500 | 464 | 0.980 | 0.970 | 0.982 | 0.734 |
| pin_hole | 500 | 470 | 0.865 | 0.979 | 0.986 | 0.623 |

- 推理速度: 0.2ms 预处理 + 2.1ms 推理 + 1.8ms NMS / 图(batch 16, 640×640, RTX 5080)
- 观察(仅陈述数据): 六类 mAP50 均 ≥0.912; 最弱类为 short(R 0.863, mAP50-95 0.482),
  最强类为 copper(mAP50-95 0.734); pin_hole 的 P 偏低(0.865)但 R 最高之一(0.979)。
  test(0.968/0.594)相对 val(0.985/0.651)有正常的泛化差距。

## 5. 产物路径汇总

- 评估目录: `C:\Users\qintx\Desktop\clauded\yolov5\runs\val\deeppcb_yolov5s_50ep_test\`
  - confusion_matrix.png, PR_curve.png, P_curve.png, R_curve.png, F1_curve.png
  - val_batch0/1/2_labels.jpg(真值), val_batch0/1/2_pred.jpg(预测)
  - labels\*.txt × 500(带置信度的预测结果, 可供后续图像处理辅助分析模块直接使用)
- 完整控制台输出: 本次评估任务日志已核对, 关键行原样摘录于本报告第 2/4 节

## 6. 评估后完整性复检

- `03_verify_dataset.py` → **ALL CHECKS PASSED**(12/12)
- 数据集 labels/ 下现有附加索引: train.cache, val.cache, test.cache(均为 yolov5 扫描缓存)
- `runs/train/deeppcb_yolov5s_50ep/weights/best.pt` 原样存在(训练目录未被触碰)
