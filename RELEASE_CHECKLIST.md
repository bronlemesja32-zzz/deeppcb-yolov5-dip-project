# 发布材料检查清单

## 已包含的文件

### 报告

- `report/main.pdf`
- `report/main.tex`
- `report/references.bib`
- `report/figures/`：报告实际引用的 11 张图片
  - `class_distribution.png`
  - `bbox_wh_distribution.png`
  - `yolo_test_PR_curve.png`
  - `yolo_test_confusion_matrix.png`
  - `yolo_test_val_batch0_pred.jpg`
  - `ip_vis_00041200.jpg`
  - `ip_vis_92000111.jpg`
  - `fail_fp_90100028.jpg`
  - `fail_fn_20085310.jpg`
  - `fail_yolo_ok_ip_fail_20085298.jpg`
  - `fail_ip_found_yolo_miss_90100033.jpg`
- `report/tables/`：报告实际引用的 7 个表格源文件
  - `tab_split.tex`
  - `tab_classmap.tex`
  - `tab_yolo_overall.tex`
  - `tab_yolo_perclass.tex`
  - `tab_ip_metrics.tex`
  - `tab_ip_params.tex`
  - `tab_pad_sweep.tex`

### 脚本与配置

- `scripts/01_inspect_dataset.py`
- `scripts/02_convert_to_yolo.py`
- `scripts/03_verify_dataset.py`
- `scripts/04_visualize_checks.py`
- `scripts/05_ip_baseline.py`
- `scripts/06_export_pad_calibration.py`
- `configs/deeppcb.yaml`

### 关键日志摘要

- `logs/conversion_report.md`
- `logs/verify_report.md`
- `logs/formal_train_report.md`
- `logs/test_eval_report.md`
- `logs/test_eval_repro_report.md`
- `logs/ip_baseline_report.md`
- `logs/pad_calibration_report.md`
- `logs/final_experiment_consistency_audit.md`

### 汇总输出

- `outputs/ip_baseline/ip_baseline_metrics.csv`
- `outputs/ip_baseline/pad_calibration.csv`

### 仓库说明文件

- `README.md`
- `.gitignore`
- `RELEASE_CHECKLIST.md`

## 未包含的文件及原因

- 完整 DeepPCB 图片数据与转换后的 `datasets/`：体积较大，应从官方仓库单独获取并在本地生成。
- `best.pt`、`last.pt` 及其他 `*.pt`/`*.pth` 权重：大文件，不随课程材料仓库提供。
- 完整 `runs/` 目录、大量 YOLO 预测 txt、训练批次图和非报告必需曲线：数量或体积较大，关键结果已由最终报告和摘要日志保留。
- `.venv/`、Python 缓存和 `*.cache`：本机环境或可再生成文件，不适合提交。
- 大型控制台原始日志：仅保留便于阅读的关键摘要报告。
- ZIP、7Z、RAR 等归档文件：避免重复打包和无谓增大仓库体积。
- 未被最终报告引用的图片与表格：不属于最终提交所需材料。

## 发布状态

- [x] 是否包含完整数据集：否
- [x] 是否包含权重：否
- [x] 是否包含代码：是
- [x] 是否包含最终 PDF：是
- [x] 是否包含关键日志摘要：是
- [x] 是否初始化 Git 仓库：否
- [x] 是否推送到 GitHub/Gitee：否
