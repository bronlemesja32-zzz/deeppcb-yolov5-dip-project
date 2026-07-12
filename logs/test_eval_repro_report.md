# Step 7A: Official test 评估复跑(证据硬化)报告

> 目的: 不改模型不改数据, 复跑一次 official test 评估, 补齐完整 stdout/stderr/退出码/
> 结构化指标证据。所有数字来自 2026-07-12 实际运行。

## 1. 执行信息

- 实际命令(工作目录 `C:\Users\qintx\Desktop\clauded\yolov5`):
  ```
  val.py --weights runs/train/deeppcb_yolov5s_50ep/weights/best.pt --data ../pcb_project/datasets/deeppcb/deeppcb.yaml --img 640 --batch 16 --task test --name deeppcb_yolov5s_50ep_test_repro --save-txt --save-conf
  ```
- 解释器: `C:\Users\qintx\Desktop\dip-2022-spring-master\dip-2022-spring-master\LAB\project\DocRescue\.venv\Scripts\python.exe`(Python 3.12.13, torch 2.7.1+cu128, CUDA:0 RTX 5080 Laptop 16303MiB — 与原评估同一环境, 横幅一致)
- 开始时间: 2026-07-12 08:09:46; 结束时间: 2026-07-12 08:11:00(时长 73.4 s)
- 退出码: **0**
- 输出目录: `yolov5/runs/val/deeppcb_yolov5s_50ep_test_repro/`(全新目录, **未覆盖**旧目录
  `deeppcb_yolov5s_50ep_test`)
- 完整日志: `pcb_project/logs/test_eval_repro_stdout.log`(0 字节 — yolov5 全部日志经 logging
  输出到 stderr, 属正常), `test_eval_repro_stderr.log`(6762 字节, 含参数回显/环境横幅/指标表)
- 结构化指标: `pcb_project/logs/test_eval_repro_metrics.csv`(由 stderr 表格 awk 解析生成, 非手抄)
- task=test 确认: 参数回显行 `task=test`; 本次 stderr 无 "New cache created"(labels/test.cache
  已存在被只读复用, 数据集未被写入)

## 2. 复跑指标(500 图, 3140 instances)

| class | Images | Instances | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| **all** | 500 | 3140 | 0.951 | 0.944 | 0.968 | 0.594 |
| open | 500 | 659 | 0.973 | 0.973 | 0.976 | 0.547 |
| short | 500 | 478 | 0.936 | 0.863 | 0.912 | 0.482 |
| mousebite | 500 | 586 | 0.969 | 0.951 | 0.981 | 0.572 |
| spur | 500 | 483 | 0.985 | 0.927 | 0.970 | 0.606 |
| copper | 500 | 464 | 0.980 | 0.970 | 0.982 | 0.734 |
| pin_hole | 500 | 470 | 0.865 | 0.979 | 0.986 | 0.623 |

## 3. 与原 test_eval_report.md 的对比

- 总体与六类的 P/R/mAP50/mAP50-95: **在日志显示精度(3 位小数)下逐项完全一致**, 含 Images 与
  Instances 计数(500/3140 及各类分项)。无任何差异, 亦无需要记录的浮点微差。
- 唯一不同的是推理速度行(本次 0.4/2.6/1.7 ms vs 原 0.2/2.1/1.8 ms/图, 预处理/推理/NMS),
  属机器瞬时负载差异, 与指标无关。
- 结论: **指标可精确复现, 无不一致, 继续打包(Step 7C)的前提成立。**

## 4. 约束遵守

未训练; 零 pip 操作; 未修改 DeepPCB 原始数据与 YOLO 数据集(cache 只读复用);
未触碰 `runs/train/deeppcb_yolov5s_50ep`; 未覆盖旧 test 评估目录; 仅新增
repro 输出目录与本报告及三个日志/CSV 文件。
