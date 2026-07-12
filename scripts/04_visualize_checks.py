# -*- coding: utf-8 -*-
"""
Step 4: 标注可视化抽查 + 数据集统计图

- 从转换后的 YOLO 标签(而非源标注)反算像素框画到图上, 端到端验证转换正确性
- 30 张: train 12 / val 8 / test 10, 种子 20260712, 贪心保证 6 类尽量覆盖
- 类别分布柱状图 + bbox 宽高散点图 (像素值由 YOLO 归一化标签乘回实际 W,H 得到)
图表文字用英文, 避免 Windows 控制台/字体编码问题; 数字均来自实际文件。
"""
import os
import random
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

SEED = 20260712
PROJ = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PROJ, "datasets", "deeppcb")
VIS = os.path.join(PROJ, "outputs", "label_check")
STATS = os.path.join(PROJ, "outputs", "dataset_stats")

NAMES = ["open", "short", "mousebite", "spur", "copper", "pin_hole"]
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#00bcd4"]
QUOTA = {"train": 12, "val": 8, "test": 10}


def read_label(split, base):
    p = os.path.join(OUT, "labels", split, base + ".txt")
    boxes = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            boxes.append((int(parts[0]), *map(float, parts[1:])))
    return boxes


def pick_images(rng):
    """每 split 先贪心选能覆盖未出现类别的图, 再随机补足配额"""
    chosen = {}
    covered = set()
    for split in ("train", "val", "test"):
        ids = sorted(f[:-4] for f in os.listdir(os.path.join(OUT, "labels", split))
                     if f.endswith(".txt"))
        rng.shuffle(ids)
        cls_of = {b: {bx[0] for bx in read_label(split, b)} for b in ids}
        sel = []
        for b in ids:                      # 贪心: 优先带来新类别的图
            if len(sel) >= QUOTA[split]:
                break
            if cls_of[b] - covered:
                sel.append(b)
                covered |= cls_of[b]
        for b in ids:                      # 随机补足
            if len(sel) >= QUOTA[split]:
                break
            if b not in sel:
                sel.append(b)
        chosen[split] = sel
    return chosen, covered


def draw_one(split, base, font):
    img_p = os.path.join(OUT, "images", split, base + ".jpg")
    with Image.open(img_p) as im:
        im = im.convert("RGB")
        W, H = im.width, im.height
        dr = ImageDraw.Draw(im)
        for cid, xc, yc, w, h in read_label(split, base):
            x1 = (xc - w / 2) * W
            y1 = (yc - h / 2) * H
            x2 = (xc + w / 2) * W
            y2 = (yc + h / 2) * H
            c = COLORS[cid]
            dr.rectangle([x1, y1, x2, y2], outline=c, width=2)
            label = NAMES[cid]
            tw = dr.textlength(label, font=font)
            ty = y1 - 14 if y1 >= 14 else y2
            dr.rectangle([x1, ty, x1 + tw + 4, ty + 13], fill=c)
            dr.text((x1 + 2, ty), label, fill="white", font=font)
        im.save(os.path.join(VIS, f"{split}_{base}.jpg"), quality=95)


def main():
    os.makedirs(VIS, exist_ok=True)
    os.makedirs(STATS, exist_ok=True)
    try:
        font = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    rng = random.Random(SEED)
    chosen, covered = pick_images(rng)
    n_drawn = 0
    for split, ids in chosen.items():
        for b in ids:
            draw_one(split, b, font)
            n_drawn += 1
    print(f"[OK] drew {n_drawn} images to {VIS}")
    print(f"     classes covered in selection: "
          f"{sorted(NAMES[c] for c in covered)}")

    # ---- 全量统计 (读全部转换后标签 + 实际图像尺寸) ----
    cls_per_split = {s: Counter() for s in QUOTA}
    wh_px = defaultdict(list)           # cid -> [(w_px, h_px)]
    for split in QUOTA:
        for f in os.listdir(os.path.join(OUT, "labels", split)):
            if not f.endswith(".txt"):
                continue
            base = f[:-4]
            with Image.open(os.path.join(OUT, "images", split, base + ".jpg")) as im:
                W, H = im.width, im.height
            for cid, xc, yc, w, h in read_label(split, base):
                cls_per_split[split][cid] += 1
                wh_px[cid].append((w * W, h * H))

    # 图 1: 类别分布柱状图 (per split 分组)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = range(6)
    width = 0.27
    for i, (split, color) in enumerate(
            zip(("train", "val", "test"), ("#4363d8", "#f58231", "#3cb44b"))):
        vals = [cls_per_split[split][c] for c in x]
        bars = ax.bar([xi + (i - 1) * width for xi in x], vals, width,
                      label=f"{split} (n={sum(vals)})", color=color)
        ax.bar_label(bars, fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(NAMES)
    ax.set_xlabel("class")
    ax.set_ylabel("instances")
    ax.set_title("DeepPCB YOLO dataset: class distribution by split (seed=20260712)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p1 = os.path.join(STATS, "class_distribution.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # 图 2: bbox 宽高散点 (像素)
    fig, ax = plt.subplots(figsize=(7, 7))
    for cid in range(6):
        pts = wh_px[cid]
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=6, alpha=0.35,
                   color=COLORS[cid], label=f"{NAMES[cid]} (n={len(pts)})")
    ax.set_xlabel("bbox width (px)")
    ax.set_ylabel("bbox height (px)")
    ax.set_title("DeepPCB bbox width/height distribution (all splits)")
    ax.legend(markerscale=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(STATS, "bbox_wh_distribution.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    all_w = [p[0] for pts in wh_px.values() for p in pts]
    all_h = [p[1] for pts in wh_px.values() for p in pts]
    print(f"[OK] stats figures: {p1} ; {p2}")
    print(f"     total boxes plotted: {len(all_w)}, "
          f"w px range [{min(all_w):.0f},{max(all_w):.0f}], "
          f"h px range [{min(all_h):.0f},{max(all_h):.0f}]")


if __name__ == "__main__":
    main()
