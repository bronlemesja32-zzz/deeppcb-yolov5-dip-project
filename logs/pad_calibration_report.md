# Step 7B: PAD 标定证据硬化报告

> 目的: 把 ip_baseline_report.md §5 中仅以文本表格存在的 PAD val 扫描结果落盘为
> 结构化 CSV。所有数字来自 2026-07-12 实际运行 `06_export_pad_calibration.py`。

## 1. 执行方式

- 新增只读脚本: `pcb_project/06_export_pad_calibration.py`
- **未修改** `05_ip_baseline.py`: 通过 `importlib.util.spec_from_file_location` 按路径加载它,
  直接复用其 `propose_raw`(差分→中值 5→Otsu→开 3/闭 5→连通域→面积/宽高过滤)、
  `pad_boxes`、`greedy_match`、`prf`、`load_gt` 及参数常量(PAD_CANDIDATES/IOU_THRS/路径),
  保证与正式 baseline **同一份代码、同一组参数**
- 解释器: DocRescue .venv python 3.12.13(与 Step 6 一致)
- 输出: `pcb_project/outputs/ip_baseline/pad_calibration.csv`(8 行 × 17 列)

## 2. 范围与结果

- **只使用 val split**: manifest.csv 中 split=val 的 100 张图(断言校验), 674 个真值框,
  原始(未外扩)候选 684 个。**未触碰 test: 是的, 全程未读取任何 test 行。**
- PAD 扫描范围: {0, 4, 6, 8, 10, 12, 14, 16} px(与 05_ip_baseline.py 的 PAD_CANDIDATES 一致)
- 结果(完整数值见 CSV):

| pad_px | F1@0.33 | F1@0.50 |
|---|---|---|
| 0 | 0.0619 | 0.0059 |
| 4 | 0.5567 | 0.1340 |
| 6 | 0.6907 | 0.4978 |
| 8 | 0.6996 | 0.6657 |
| **10** | **0.7025** | **0.6878** |
| 12 | 0.6996 | 0.6863 |
| 14 | 0.6996 | 0.6819 |
| 16 | 0.6966 | 0.6318 |

## 3. 结论

- 最优 PAD: **10 px**(两个 IoU 阈值下 F1 同时最优)
- 选择依据: ① 双阈值同时最优; ② 位于 8–14 的平台区, 对取值不敏感(稳健);
  ③ 与 DeepPCB 真值框"缺陷像素外带边距"的标注风格量级吻合(小缺陷真值框 18×19 px
  隐含每侧 ~8–9 px 边距)
- 与 ip_baseline_report.md §5 的文本表格对比: **8 行 × 6 个指标逐项一致**(0.0614/0.0623/0.0619
  … 0.6915/0.7018/0.6966), 且脚本打印的 `frozen PAD_PX in 05_ip_baseline.py: 10` 证实
  正式 baseline 使用的就是该标定值
- 是否触碰 test: **否**

## 4. 约束遵守

未训练; 零 pip 操作; 未修改 05_ip_baseline.py 及任何已有产物;
新增文件仅: `06_export_pad_calibration.py`, `pad_calibration.csv`, 本报告。
