# Step 4 正式训练报告(YOLOv5s, 50 epoch)

> 状态: **训练完成, 核验通过。未进入 test 评估(按用户要求停止)。**
> 约束遵守: 未安装/升级/卸载任何包; 未改动 DeepPCB YOLO 数据集; 未覆盖 deeppcb_smoke;
> 未修改模型结构; 使用官方 yolov5 代码(master 2885cb4)与官方 yolov5s.pt 预训练权重。
> 所有数字来自实际运行输出。

## 1. 环境(与 Step 3 冒烟完全一致, 训练前实测)

- python.exe(绝对路径):
  `C:\Users\qintx\Desktop\dip-2022-spring-master\dip-2022-spring-master\LAB\project\DocRescue\.venv\Scripts\python.exe`
  (Python 3.12.13)
- torch 2.7.1+cu128 / torchvision 0.22.1+cu128 / ultralytics 8.4.83(--no-deps 安装) /
  numpy 1.26.4 / opencv 4.11.0 / pillow 11.1.0
- `torch.cuda.is_available()` = True; GPU: NVIDIA GeForce RTX 5080 Laptop GPU (16303 MiB)
- yolov5 代码: `C:\Users\qintx\Desktop\clauded\yolov5`(官方 master, commit 2885cb4, 未修改)
- 数据: `pcb_project/datasets/deeppcb/deeppcb.yaml`(train 900 / val 100 / test 500, 种子 20260712)

## 2. 启动前预检(2026-07-12 实测)

| 检查 | 结果 |
|---|---|
| venv python 存在 | True |
| 目标目录 runs/train/deeppcb_yolov5s_50ep 不存在(不会覆盖) | True(不存在) |
| deeppcb_smoke 目录完好 | True |
| yolov5s.pt 就位 | True |
| deeppcb.yaml 就位 | True |

## 3. 实际执行命令

工作目录 `C:\Users\qintx\Desktop\clauded\yolov5`, 以 Start-Process 独立进程启动
(PYTHONUNBUFFERED=1, stdout/stderr 分别重定向到本目录 formal_train_console.log / .err.log):

```
C:\Users\qintx\...\DocRescue\.venv\Scripts\python.exe train.py --img 640 --batch 16 --epochs 50 --data ../pcb_project/datasets/deeppcb/deeppcb.yaml --weights yolov5s.pt --name deeppcb_yolov5s_50ep
```

- 启动时刻: 2026-07-12 04:45:26(另存于 formal_train_start_time.txt), PID 25012

## 4. 训练结果(2026-07-12 实测)

### 4.1 完成情况与耗时

- 训练日志: `50 epochs completed in 0.098 hours`(约 5.9 分钟, 纯训练循环)
- 进程墙钟: 启动 04:45:26 → 退出 04:53:36, 共 **8 分 10 秒**(含启动、每 epoch 验证、
  best.pt 最终验证与绘图; 结束时刻由进程监视器记录)
- results.csv 恰好 **50 行数据(epoch 0–49)** = 50 epoch 完整完成
- 全程无 Traceback/OOM/中断(日志监视器仅捕获正常信号)

### 4.2 产物核验(逐项 Test-Path 实测, 目录 `runs\train\deeppcb_yolov5s_50ep\`)

| 要求项 | 结果 |
|---|---|
| weights/best.pt | 存在(14.5MB, optimizer 已剥离), weights/last.pt 亦存在 |
| results.csv | 存在, 50 行 |
| confusion_matrix.png | 存在 |
| PR_curve.png / P_curve.png / R_curve.png / F1_curve.png | 均存在 |
| results.png | 存在 |
| val_batch*_pred.jpg | 存在: val_batch0/1/2_pred.jpg(含对应 _labels.jpg) |
| 其他 | labels.jpg, labels_correlogram.jpg, train_batch0/1/2.jpg, hyp.yaml, opt.yaml, TensorBoard events |

### 4.3 results.csv 最后一行(epoch 49, in-training 验证, val split)

| precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|
| 0.95873 | 0.93499 | 0.94906 | 0.43635 |

### 4.4 best.pt 最终验证(训练结束后自动执行, val 100 图 674 框, 融合层)

| class | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **all** | **0.978** | **0.973** | **0.985** | **0.651** |
| open | 0.959 | 0.985 | 0.994 | 0.629 |
| short | 0.990 | 0.984 | 0.982 | 0.574 |
| mousebite | 0.968 | 0.966 | 0.986 | 0.626 |
| spur | 0.989 | 0.952 | 0.971 | 0.608 |
| copper | 0.997 | 0.990 | 0.995 | 0.752 |
| pin_hole | 0.963 | 0.960 | 0.983 | 0.714 |

### 4.5 末行(0.436)与最终验证(0.651)差异说明(基于 results.csv 全 50 行实际数据)

yolov5 的 best.pt 按 fitness = 0.1×mAP50 + 0.9×mAP50-95 选取, 逐行计算得
**best epoch = 38**(P 0.97748, R 0.97294, mAP50 0.98511, mAP50-95 0.64837),
与 best.pt 最终验证(0.978/0.973/0.985/0.651)一致(同权重两次评估的正常微差)。
训练后期(ep46–49)mAP50-95 在 0.36–0.50 间波动(val 仅 100 图, 小验证集抖动属正常),
epoch 49 恰为波动低点, 故 4.3 与 4.4 数值差异是 checkpoint 选择机制的正常结果, 非异常。

每 5 epoch 采样(in-training val, 摘自 results.csv):

| epoch | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| 0 | 0.043 | 0.245 | 0.037 | 0.010 |
| 10 | 0.834 | 0.812 | 0.831 | 0.298 |
| 20 | 0.848 | 0.799 | 0.808 | 0.274 |
| 30 | 0.920 | 0.900 | 0.909 | 0.350 |
| **38(best)** | **0.977** | **0.973** | **0.985** | **0.648** |
| 40 | 0.944 | 0.947 | 0.954 | 0.456 |
| 45 | 0.978 | 0.958 | 0.972 | 0.497 |
| 49 | 0.959 | 0.935 | 0.949 | 0.436 |

### 4.6 约束遵守复核(训练后实测)

- 未安装/升级/卸载任何包(本步骤零 pip 操作)
- deeppcb_smoke 目录原样: best.pt 存在, results.csv 仍为 2 行
- 数据集: 训练后重跑 `03_verify_dataset.py` → **ALL CHECKS PASSED**(12/12),
  labels/ 下仍只有 train.cache 与 val.cache 两个附加索引(冒烟阶段已生成, 本次复用)
- 模型结构零修改, 使用官方 yolov5s.yaml 派生结构(nc=6 由 --data 自动覆盖)与官方 yolov5s.pt

## 4.7 日志唯一性审计(2026-07-12 应用户要求补充)

对 formal_train_console.err.log 剥离 ANSI 颜色码后统计:
- `train: weights=` 头部行: **1 次**(第 1 行; 原始文件中 `train: ` 与 `weights=` 之间存在
  ANSI 码 `\x1b[0m`, 直接按连续字符串 grep 会漏检, 需先剥码)
- `data=` 参数: **仅 1 次**, 值为 `data=../pcb_project/datasets/deeppcb/deeppcb.yaml`
  (`../` 前缀, 相对 yolov5 工作目录; 不存在 `./pcb_project` 变体; stdout log 中 0 次)
- `YOLOv5` 环境横幅 1 次, `Starting training for` 1 次 → 单次运行, 无多 run 混入
- 注意: `grep -o "epochs=[0-9]*"` 会同时截出 `epochs=3`(来自超参行 `warmup_epochs=3.0`
  的子串)与 `epochs=50`(头部参数), 前者非第二次运行
- 交叉验证: run 目录 opt.yaml(yolov5 官方参数快照)记录 data/epochs/name/save_dir/resume
  与日志头逐项一致; 且日志文件由 Start-Process 启动时新建, 仅属于 PID 25012 单一进程
- 结论: **日志纯净, 本报告所有引用无需修正**

## 5. 日志与产物路径汇总

- 训练控制台日志: `pcb_project/logs/formal_train_console.log`(stdout) 与
  `formal_train_console.err.log`(stderr, 主日志在此)
- 启动时刻文件: `pcb_project/logs/formal_train_start_time.txt`
- 运行目录: `C:\Users\qintx\Desktop\clauded\yolov5\runs\train\deeppcb_yolov5s_50ep\`
- 最优权重: `runs\train\deeppcb_yolov5s_50ep\weights\best.pt`(对应 epoch 38)

按要求在此停止: **未做 test(500 图)评估, 未写 LaTeX, 未做额外实验。**
