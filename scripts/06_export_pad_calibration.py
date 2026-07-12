# -*- coding: utf-8 -*-
"""
Step 7B: 把 PAD 标定扫描(val split)落盘为结构化 CSV。

- 不修改 05_ip_baseline.py: 通过 importlib 按路径加载它, 复用其中的
  流水线(propose_raw/pad_boxes)与评价函数(greedy_match/prf)及全部参数,
  保证与 ip_baseline_report.md 中的文本表格同源同参。
- 只处理 val split(100 图), 不触碰 test。
- 输出: pcb_project/outputs/ip_baseline/pad_calibration.csv
"""
import csv
import importlib.util
import os
from collections import Counter

import cv2

PROJ = os.path.dirname(os.path.abspath(__file__))

spec = importlib.util.spec_from_file_location(
    "ip_baseline", os.path.join(PROJ, "05_ip_baseline.py"))
ip = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ip)   # 仅执行模块级定义(main 有 __main__ 保护), 无副作用

OUT_CSV = os.path.join(PROJ, "outputs", "ip_baseline", "pad_calibration.csv")


def main():
    with open(ip.MANIFEST, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "val"]
    assert len(rows) == 100, f"val rows = {len(rows)} != 100"

    raws = []
    total_gt = 0
    total_props = 0
    for r in rows:
        temp = cv2.imread(os.path.join(ip.DEEPPCB, r["source_temp_image"].replace("/", os.sep)),
                          cv2.IMREAD_GRAYSCALE)
        test = cv2.imread(os.path.join(ip.DEEPPCB, r["source_test_image"].replace("/", os.sep)),
                          cv2.IMREAD_GRAYSCALE)
        assert temp is not None and test is not None, r["image_id"]
        gt = ip.load_gt(os.path.join(ip.DEEPPCB, r["source_label"].replace("/", os.sep)))
        raw, _, _, _ = ip.propose_raw(temp, test)
        raws.append((raw, gt, test.shape))
        total_gt += len(gt)
        total_props += len(raw)

    out_rows = []
    for pad in ip.PAD_CANDIDATES:
        stat = {t: Counter() for t in ip.IOU_THRS}
        for raw, gt, (H, W) in raws:
            props = ip.pad_boxes(raw, pad, W, H)
            for t in ip.IOU_THRS:
                tp, fp, fn, _ = ip.greedy_match(props, gt, t)
                stat[t].update({"tp": tp, "fp": fp, "fn": fn})
        rec = {"pad_px": pad, "split": "val", "images": len(rows),
               "gt_boxes": total_gt, "proposal_boxes": total_props}
        for t, tag in ((0.33, "033"), (0.50, "050")):
            c = stat[t]
            p, rc, f1 = ip.prf(c["tp"], c["fp"], c["fn"])
            rec.update({f"TP_{tag}": c["tp"], f"FP_{tag}": c["fp"], f"FN_{tag}": c["fn"],
                        f"P_{tag}": round(p, 4), f"R_{tag}": round(rc, 4),
                        f"F1_{tag}": round(f1, 4)})
        out_rows.append(rec)

    with open(OUT_CSV, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print("[OK] val images:", len(rows), " gt:", total_gt, " raw proposals:", total_props)
    best = max(out_rows, key=lambda r: r["F1_033"])
    print("[OK] best pad by F1_033:", best["pad_px"],
          " F1_033:", best["F1_033"], " F1_050:", best["F1_050"])
    print("[OK] frozen PAD_PX in 05_ip_baseline.py:", ip.PAD_PX)
    print("[OK] csv:", OUT_CSV)


if __name__ == "__main__":
    main()
