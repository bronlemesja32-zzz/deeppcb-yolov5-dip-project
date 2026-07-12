# -*- coding: utf-8 -*-
"""
Step 6: 传统图像处理辅助分析模块 (class-agnostic defect proposal baseline)

流程: 灰度化 -> 模板差分(absdiff) -> 中值平滑 -> Otsu 二值化
      -> 形态学开(去噪)/闭(连接) -> 连通域分析 -> 候选框过滤
      -> 与真值框做贪心 IoU 匹配(IoU=0.33 / 0.50), 输出 box-level 指标

定位: 无类别缺陷候选区域生成(proposal), 不做 6 类分类, 不与 YOLO 按 mAP 对比。
YOLO test 预测(conf>=YOLO_VIZ_CONF)仅用于可视化对比与失败案例筛选
(同样的 class-agnostic 匹配, 只看定位, 非官方指标)。

只处理 manifest.csv 中 split=test 的 500 张官方测试图。
只读取 DeepPCB 原始数据与已有产物, 只往 pcb_project/outputs/ip_baseline/ 写新文件。
"""
import csv
import os
import random
from collections import Counter, defaultdict

import cv2
import numpy as np

# ---------------- 参数(报告参数表与此处一致) ----------------
MEDIAN_KSIZE = 5          # 差分图中值滤波核
MORPH_OPEN_KSIZE = 3      # 开运算核(椭圆), 去细小噪声/配准边缘毛刺
MORPH_CLOSE_KSIZE = 5     # 闭运算核(椭圆), 连接断裂缺陷区域
MIN_AREA = 50             # 连通域最小像素面积
MIN_WH = 8                # 候选框最小宽/高(px)
MAX_WH = 250              # 候选框最大宽/高(px), 防配准失败产生的长条
PAD_PX = 10               # 候选框每侧外扩(px)。DeepPCB 真值框在缺陷像素外带明显边距,
                          # 连通域紧贴缺陷像素, 需按标注风格外扩才能在 IoU 上可比。
                          # 该值在 val split(100 图)上标定(python 05_ip_baseline.py calibrate),
                          # 冻结后应用于 test, 不在 test 上调参。
IOU_THRS = (0.33, 0.50)   # 0.33=DeepPCB 官方评估默认约束(宽松定位), 0.50=常见检测阈值
YOLO_VIZ_CONF = 0.25      # YOLO 预测仅用于可视化/案例筛选时的置信度过滤
SEED = 20260712
PAD_CANDIDATES = (0, 4, 6, 8, 10, 12, 14, 16)  # calibrate 模式扫描
N_VIS_PER_GROUP = 3       # 每 group 抽取可视化数(不足 24 再随机补)
N_VIS_MIN = 24
N_FAIL_PER_CASE = 4

PROJ = os.path.dirname(os.path.abspath(__file__))
DEEPPCB = os.path.join(os.path.dirname(PROJ), "DeepPCB")
MANIFEST = os.path.join(PROJ, "datasets", "deeppcb", "manifest.csv")
YOLO_PRED = os.path.join(os.path.dirname(PROJ), "yolov5", "runs", "val",
                         "deeppcb_yolov5s_50ep_test", "labels")
OUT = os.path.join(PROJ, "outputs", "ip_baseline")
BOX_DIR = os.path.join(OUT, "traditional_boxes")
VIS_DIR = os.path.join(OUT, "visualizations")
FAIL_DIR = os.path.join(OUT, "failure_cases")

CLASS_NAMES = ["open", "short", "mousebite", "spur", "copper", "pin_hole"]
COL_GT = (0, 200, 0)        # 绿
COL_IP = (0, 165, 255)      # 橙
COL_Y5 = (255, 80, 0)       # 蓝(BGR)


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    iw = min(ax2, bx2) - max(ax1, bx1)
    ih = min(ay2, by2) - max(ay1, by1)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua


def greedy_match(props, gts, thr):
    """props: [(x1,y1,x2,y2,score)] 按 score 降序贪心; 每个 GT 至多匹配一次。
    返回 (tp, fp, fn, matched_gt_index_set)"""
    matched = set()
    tp = 0
    for p in sorted(props, key=lambda t: -t[4]):
        best, bj = 0.0, -1
        for j, g in enumerate(gts):
            if j in matched:
                continue
            v = iou(p, g)
            if v > best:
                best, bj = v, j
        if best >= thr:
            matched.add(bj)
            tp += 1
    return tp, len(props) - tp, len(gts) - tp, matched


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def load_gt(path):
    boxes = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            x1, y1, x2, y2, t = map(int, line.split())
            boxes.append((x1, y1, x2, y2, t))
    return boxes


def load_yolo(image_id, W, H):
    """返回 conf>=YOLO_VIZ_CONF 的 [(x1,y1,x2,y2,conf,cls)]"""
    p = os.path.join(YOLO_PRED, image_id + ".txt")
    out = []
    if not os.path.exists(p):
        return out
    with open(p, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 6:
                continue
            c, xc, yc, w, h, conf = int(parts[0]), *map(float, parts[1:])
            if conf < YOLO_VIZ_CONF:
                continue
            out.append(((xc - w / 2) * W, (yc - h / 2) * H,
                        (xc + w / 2) * W, (yc + h / 2) * H, conf, c))
    return out


def propose_raw(temp_gray, test_gray):
    """核心传统流水线(未外扩)。返回 (raw_props[(x1,y1,x2,y2,score)], diff, bw_otsu, morph)
    过滤(面积/宽高)作用在原始连通域尺寸上, PAD 外扩由 pad_boxes 单独施加。"""
    diff = cv2.absdiff(test_gray, temp_gray)
    sm = cv2.medianBlur(diff, MEDIAN_KSIZE)
    _, bw = cv2.threshold(sm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (MORPH_OPEN_KSIZE, MORPH_OPEN_KSIZE))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (MORPH_CLOSE_KSIZE, MORPH_CLOSE_KSIZE))
    opened = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k_open)
    morph = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k_close)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(morph, connectivity=8)
    props = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < MIN_AREA or w < MIN_WH or h < MIN_WH or w > MAX_WH or h > MAX_WH:
            continue
        mask = labels == i
        score = float(diff[mask].mean()) / 255.0  # 连通域内平均差分强度归一化
        props.append((int(x), int(y), int(x + w), int(y + h), round(score, 4)))
    return props, diff, bw, morph


def pad_boxes(props, pad, W, H):
    """按标注风格对候选框每侧外扩 pad px 并裁剪到图像范围"""
    return [(max(0, b[0] - pad), max(0, b[1] - pad),
             min(W, b[2] + pad), min(H, b[3] + pad), b[4]) for b in props]


def propose(temp_gray, test_gray):
    raw, diff, bw, morph = propose_raw(temp_gray, test_gray)
    H, W = test_gray.shape
    return pad_boxes(raw, PAD_PX, W, H), diff, bw, morph


def calibrate():
    """在 val split(100 图)上扫描 PAD_PX 候选值, 打印各值的 box-level 指标。
    选定值冻结进脚本顶部 PAD_PX 后再跑 test, 避免在 test 上调参。"""
    with open(MANIFEST, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "val"]
    print(f"[CALIBRATE] val images: {len(rows)}, pad candidates: {PAD_CANDIDATES}")
    raws = []
    for r in rows:
        temp = cv2.imread(os.path.join(DEEPPCB, r["source_temp_image"].replace("/", os.sep)),
                          cv2.IMREAD_GRAYSCALE)
        test = cv2.imread(os.path.join(DEEPPCB, r["source_test_image"].replace("/", os.sep)),
                          cv2.IMREAD_GRAYSCALE)
        gt = load_gt(os.path.join(DEEPPCB, r["source_label"].replace("/", os.sep)))
        raw, _, _, _ = propose_raw(temp, test)
        raws.append((raw, gt, test.shape))
    print("pad |  IoU=0.33: P / R / F1        |  IoU=0.50: P / R / F1")
    for pad in PAD_CANDIDATES:
        stat = {t: Counter() for t in IOU_THRS}
        for raw, gt, (H, W) in raws:
            props = pad_boxes(raw, pad, W, H)
            for t in IOU_THRS:
                tp, fp, fn, _ = greedy_match(props, gt, t)
                stat[t].update({"tp": tp, "fp": fp, "fn": fn})
        line = f"{pad:3d}"
        for t in IOU_THRS:
            c = stat[t]
            p, rc, f1 = prf(c["tp"], c["fp"], c["fn"])
            line += f" | {p:.4f} / {rc:.4f} / {f1:.4f}"
        print(line)


def panel_header(img, title):
    bar = np.full((26, img.shape[1], 3), 235, np.uint8)
    cv2.putText(bar, title, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1,
                cv2.LINE_AA)
    return np.vstack([bar, img])


def draw_boxes(img, boxes, color, thick=2):
    for b in boxes:
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color, thick)


def render(temp_gray, test_gray, gt, props, yolo, diff, bw, morph):
    """4 栏: template | test+GT | diff/otsu/morph 复合 | test+IP+Y5+GT"""
    p1 = cv2.cvtColor(temp_gray, cv2.COLOR_GRAY2BGR)

    p2 = cv2.cvtColor(test_gray, cv2.COLOR_GRAY2BGR)
    draw_boxes(p2, gt, COL_GT)

    # 复合诊断图: 背景=拉伸后的 diff(暗), 中灰=Otsu 保留但被形态学去掉, 白=最终 mask
    base = cv2.normalize(diff, None, 0, 100, cv2.NORM_MINMAX).astype(np.uint8)
    p3g = base.copy()
    p3g[bw > 0] = 170
    p3g[morph > 0] = 255
    p3 = cv2.cvtColor(p3g, cv2.COLOR_GRAY2BGR)

    p4 = cv2.cvtColor(test_gray, cv2.COLOR_GRAY2BGR)
    draw_boxes(p4, gt, COL_GT, 2)
    draw_boxes(p4, props, COL_IP, 2)
    draw_boxes(p4, [(b[0], b[1], b[2], b[3]) for b in yolo], COL_Y5, 1)
    for i, (txt, col) in enumerate((("GT", COL_GT), ("IP", COL_IP), ("Y5", COL_Y5))):
        cv2.putText(p4, txt, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2,
                    cv2.LINE_AA)

    sep = np.full((p1.shape[0] + 26, 4, 3), 255, np.uint8)
    panels = [panel_header(p1, "template"),
              panel_header(p2, "test + GT(green)"),
              panel_header(p3, "diff(dark) / otsu(grey) / morph(white)"),
              panel_header(p4, "test + IP(orange) + Y5(blue) + GT(green)")]
    row = panels[0]
    for p in panels[1:]:
        row = np.hstack([row, sep, p])
    return row


def main():
    import shutil
    for d in (BOX_DIR, VIS_DIR, FAIL_DIR):   # 重建, 防止参数变更后旧文件残留
        if os.path.isdir(d):
            shutil.rmtree(d)
    for d in (OUT, BOX_DIR, VIS_DIR, FAIL_DIR):
        os.makedirs(d, exist_ok=True)

    with open(MANIFEST, encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == "test"]
    assert len(rows) == 500, f"test rows = {len(rows)} != 500"

    per_img = []
    agg = {t: Counter() for t in IOU_THRS}
    total_props = 0
    total_gt = 0
    empty_prop_imgs = []
    cache = {}  # image_id -> (paths, 用于二次渲染)

    for r in rows:
        image_id = r["image_id"]
        temp_p = os.path.join(DEEPPCB, r["source_temp_image"].replace("/", os.sep))
        test_p = os.path.join(DEEPPCB, r["source_test_image"].replace("/", os.sep))
        gt_p = os.path.join(DEEPPCB, r["source_label"].replace("/", os.sep))
        temp = cv2.imread(temp_p, cv2.IMREAD_GRAYSCALE)
        test = cv2.imread(test_p, cv2.IMREAD_GRAYSCALE)
        assert temp is not None and test is not None, image_id
        H, W = test.shape
        gt = load_gt(gt_p)
        yolo = load_yolo(image_id, W, H)

        props, diff, bw, morph = propose(temp, test)
        total_props += len(props)
        total_gt += len(gt)
        if not props:
            empty_prop_imgs.append(image_id)

        with open(os.path.join(BOX_DIR, image_id + ".txt"), "w", encoding="ascii",
                  newline="\n") as f:
            f.write("".join(f"{b[0]} {b[1]} {b[2]} {b[3]} {b[4]:.4f}\n" for b in props))

        rec = {"image_id": image_id, "group_id": r["group_id"],
               "gt_boxes": len(gt), "proposal_boxes": len(props)}
        ip_matched = {}
        for t in IOU_THRS:
            tp, fp, fn, matched = greedy_match(props, gt, t)
            p, rc, f1 = prf(tp, fp, fn)
            tag = f"{t:.2f}".replace("0.", "0")[-3:]  # 033 / 050
            rec.update({f"TP_{tag}": tp, f"FP_{tag}": fp, f"FN_{tag}": fn,
                        f"precision_{tag}": round(p, 4), f"recall_{tag}": round(rc, 4),
                        f"f1_{tag}": round(f1, 4)})
            agg[t].update({"tp": tp, "fp": fp, "fn": fn})
            ip_matched[t] = matched
        # YOLO class-agnostic 定位匹配(仅案例筛选/讨论用, conf>=0.25)
        ytp, yfp, yfn, y_matched = greedy_match(
            [(b[0], b[1], b[2], b[3], b[4]) for b in yolo], gt, 0.33)
        rec.update({"yolo_tp_033": ytp, "yolo_fn_033": yfn,
                    "ip_only_gt_033": len(ip_matched[0.33] - y_matched)})
        per_img.append(rec)
        cache[image_id] = (temp_p, test_p, gt_p)

    assert total_gt == 3140, f"total gt = {total_gt} != 3140"

    # ---------------- 输出 CSV ----------------
    per_csv = os.path.join(OUT, "ip_baseline_per_image.csv")
    with open(per_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_img[0].keys()))
        w.writeheader()
        w.writerows(per_img)

    met_csv = os.path.join(OUT, "ip_baseline_metrics.csv")
    with open(met_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["iou_threshold", "TP", "FP", "FN", "precision", "recall", "f1",
                    "total_proposals", "total_gt", "avg_proposals_per_image",
                    "images", "images_without_proposal"])
        summary_lines = []
        for t in IOU_THRS:
            c = agg[t]
            p, rc, f1 = prf(c["tp"], c["fp"], c["fn"])
            w.writerow([t, c["tp"], c["fp"], c["fn"], round(p, 4), round(rc, 4),
                        round(f1, 4), total_props, total_gt,
                        round(total_props / len(rows), 3), len(rows),
                        len(empty_prop_imgs)])
            summary_lines.append(
                f"IoU={t}: TP={c['tp']} FP={c['fp']} FN={c['fn']} "
                f"P={p:.4f} R={rc:.4f} F1={f1:.4f}")

    # ---------------- 可视化选样(覆盖 group 与类别) ----------------
    rng = random.Random(SEED)
    by_group = defaultdict(list)
    gt_cls_of = {}
    for r, rec in zip(rows, per_img):
        by_group[r["group_id"]].append(r["image_id"])
        gt_cls_of[r["image_id"]] = {b[4] for b in load_gt(
            os.path.join(DEEPPCB, r["source_label"].replace("/", os.sep)))}
    chosen = []
    for g in sorted(by_group):
        ids = sorted(by_group[g])
        rng.shuffle(ids)
        chosen += ids[:N_VIS_PER_GROUP]
    pool = [i for ids in by_group.values() for i in ids if i not in chosen]
    rng.shuffle(pool)
    while len(chosen) < N_VIS_MIN and pool:
        chosen.append(pool.pop())
    covered = set().union(*(gt_cls_of[i] for i in chosen))
    for cls in range(1, 7):                      # 类别覆盖补选
        if cls not in covered:
            cand = next((i for i in pool if cls in gt_cls_of[i]), None)
            if cand:
                chosen.append(cand)
                covered |= gt_cls_of[cand]

    group_of = {rec["image_id"]: rec["group_id"] for rec in per_img}

    def render_id(image_id, out_path):
        temp_p, test_p, gt_p = cache[image_id]
        temp = cv2.imread(temp_p, cv2.IMREAD_GRAYSCALE)
        test = cv2.imread(test_p, cv2.IMREAD_GRAYSCALE)
        H, W = test.shape
        gt = load_gt(gt_p)
        yolo = load_yolo(image_id, W, H)
        props, diff, bw, morph = propose(temp, test)
        cv2.imwrite(out_path, render(temp, test, gt, props, yolo, diff, bw, morph),
                    [cv2.IMWRITE_JPEG_QUALITY, 92])

    for i in chosen:
        render_id(i, os.path.join(VIS_DIR, f"{group_of[i]}_{i}.jpg"))

    # ---------------- 失败案例 ----------------
    fails = []
    top_fp = sorted(per_img, key=lambda r: -r["FP_033"])[:N_FAIL_PER_CASE]
    top_fn = sorted(per_img, key=lambda r: -r["FN_033"])[:N_FAIL_PER_CASE]
    yolo_ok = [r for r in per_img if r["yolo_fn_033"] == 0]
    yolo_ok_ip_fail = sorted(yolo_ok, key=lambda r: r["f1_033"])[:N_FAIL_PER_CASE]
    ip_only = sorted([r for r in per_img if r["ip_only_gt_033"] > 0],
                     key=lambda r: -r["ip_only_gt_033"])[:N_FAIL_PER_CASE]
    for tag, group in (("fp", top_fp), ("fn", top_fn),
                       ("yolo_ok_ip_fail", yolo_ok_ip_fail),
                       ("ip_found_yolo_miss", ip_only)):
        for r in group:
            render_id(r["image_id"],
                      os.path.join(FAIL_DIR, f"{tag}_{r['image_id']}.jpg"))
        fails.append((tag, [(r["image_id"], r["FP_033"], r["FN_033"], r["f1_033"],
                             r["yolo_fn_033"], r["ip_only_gt_033"]) for r in group]))

    # ---------------- 控制台摘要(ASCII) ----------------
    print("[OK] test images:", len(rows))
    print("[OK] total gt boxes:", total_gt)
    print("[OK] total proposals:", total_props,
          " avg/img:", round(total_props / len(rows), 3))
    print("[OK] images without proposal:", len(empty_prop_imgs),
          empty_prop_imgs[:10])
    for line in summary_lines:
        print("[METRIC]", line)
    print("[OK] visualizations:", len(chosen), "->", VIS_DIR)
    print("[OK] classes covered in vis:", sorted(covered))
    for tag, items in fails:
        print(f"[FAIL-CASE {tag}]",
              ["%s FP=%d FN=%d f1=%.3f yFN=%d ipOnly=%d" % it for it in items])
    print("[OK] outputs:", met_csv, per_csv, BOX_DIR, sep="\n  ")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "calibrate":
        calibrate()
    else:
        main()
