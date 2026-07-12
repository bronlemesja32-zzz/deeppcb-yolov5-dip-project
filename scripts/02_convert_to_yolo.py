# -*- coding: utf-8 -*-
"""
Step 2: DeepPCB -> YOLOv5 格式转换

划分规则 (方案 A 修正版, 用户 2026-07-12 确认):
- 官方 test.txt 的 500 张 -> test split, 只用于最终测试
- 官方 trainval.txt 的 1000 张 -> 900 train / 100 val
- 随机种子固定为 20260712
- val 按 group 内比例抽取: 用最大余数法给每个 group 分配 val 配额(合计恰为 100),
  组内用 random.Random(20260712).sample 抽取

坐标转换公式 (逐图实际读取 W,H, 不硬编码 640):
    x_center = ((x1 + x2) / 2) / W
    y_center = ((y1 + y2) / 2) / H
    width    = (x2 - x1) / W
    height   = (y2 - y1) / H
    class_id = type - 1

类别映射: open=0, short=1, mousebite=2, spur=3, copper=4, pin_hole=5
(源标注 type: 1-open, 2-short, 3-mousebite, 4-spur, 5-copper, 6-pin-hole;
 copper 在报告中可解释为 spurious copper/余铜, 代码与 yaml 统一写 copper)

源标注为空格分隔的 5 个整数: x1 y1 x2 y2 type (已在 Step 1 全量验证)
"""
import csv
import os
import random
import shutil
import sys
from collections import Counter, defaultdict

from PIL import Image

SEED = 20260712
VAL_TOTAL = 100

PROJ = os.path.dirname(os.path.abspath(__file__))
PCB = os.path.join(os.path.dirname(PROJ), "DeepPCB", "PCBData")
OUT = os.path.join(PROJ, "datasets", "deeppcb")
LOGS = os.path.join(PROJ, "logs")

NAMES = ["open", "short", "mousebite", "spur", "copper", "pin_hole"]


def load_split(fname):
    """返回 [(img_rel, txt_rel, base_id, group_id), ...], 保持文件行序"""
    entries = []
    with open(os.path.join(PCB, fname), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_rel, txt_rel = line.split()
            base = os.path.basename(img_rel)[:-4]          # 00041000
            group = img_rel.split("/")[0]                  # group00041
            entries.append((img_rel, txt_rel, base, group))
    return entries


def assign_val(trainval, seed, val_total):
    """最大余数法按 group 分配 val 配额, 组内随机抽样。返回 val base_id 集合与配额表。"""
    by_group = defaultdict(list)
    for e in trainval:
        by_group[e[3]].append(e[2])
    n_total = len(trainval)

    groups = sorted(by_group)
    quota_f = {g: len(by_group[g]) * val_total / n_total for g in groups}
    quota = {g: int(quota_f[g]) for g in groups}
    remain = val_total - sum(quota.values())
    # 按小数部分降序补足, 同分按组名排序保证确定性
    for g in sorted(groups, key=lambda g: (-(quota_f[g] - quota[g]), g))[:remain]:
        quota[g] += 1

    rng = random.Random(seed)
    val_ids = set()
    for g in groups:
        ids = sorted(by_group[g])
        val_ids.update(rng.sample(ids, quota[g]))
    table = [(g, len(by_group[g]), quota[g]) for g in groups]
    return val_ids, table


def convert_one(img_rel, txt_rel, base, split):
    """复制图像 + 生成 YOLO 标签。返回 (num_boxes, W, H, 源行列表异常数)"""
    src_img = os.path.join(PCB, img_rel[:-4].replace("/", os.sep) + "_test.jpg")
    src_txt = os.path.join(PCB, txt_rel.replace("/", os.sep))
    dst_img = os.path.join(OUT, "images", split, base + ".jpg")
    dst_txt = os.path.join(OUT, "labels", split, base + ".txt")

    with Image.open(src_img) as im:
        W, H = im.width, im.height

    lines = []
    with open(src_txt, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            x1, y1, x2, y2, t = map(int, raw.split())
            # x_center = ((x1 + x2) / 2) / W
            # y_center = ((y1 + y2) / 2) / H
            # width    = (x2 - x1) / W
            # height   = (y2 - y1) / H
            # class_id = type - 1
            xc = ((x1 + x2) / 2) / W
            yc = ((y1 + y2) / 2) / H
            w = (x2 - x1) / W
            h = (y2 - y1) / H
            cid = t - 1
            assert 0 <= cid <= 5, f"{txt_rel}: type={t}"
            assert 0.0 <= xc - w / 2 and xc + w / 2 <= 1.0, f"{txt_rel}: x 越界"
            assert 0.0 <= yc - h / 2 and yc + h / 2 <= 1.0, f"{txt_rel}: y 越界"
            lines.append(f"{cid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    shutil.copy2(src_img, dst_img)
    with open(dst_txt, "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    return len(lines), W, H


def main():
    trainval = load_split("trainval.txt")
    test = load_split("test.txt")

    # base_id 全局唯一性
    all_ids = [e[2] for e in trainval + test]
    assert len(all_ids) == len(set(all_ids)), "base_id 存在重复, 不能扁平化输出"

    val_ids, quota_table = assign_val(trainval, SEED, VAL_TOTAL)
    assert len(val_ids) == VAL_TOTAL

    # 重建输出目录, 保证无陈旧文件
    for sub in ("images", "labels"):
        p = os.path.join(OUT, sub)
        if os.path.isdir(p):
            shutil.rmtree(p)
        for split in ("train", "val", "test"):
            os.makedirs(os.path.join(p, split), exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)

    manifest_rows = []
    non640 = []
    split_stat = defaultdict(lambda: [0, 0])  # split -> [imgs, boxes]

    def handle(entries, split_of):
        for img_rel, txt_rel, base, group in entries:
            split = split_of(base)
            n, W, H = convert_one(img_rel, txt_rel, base, split)
            if (W, H) != (640, 640):
                non640.append((base, W, H))
            split_stat[split][0] += 1
            split_stat[split][1] += n
            manifest_rows.append({
                "split": split,
                "image_id": base,
                "group_id": group,
                "source_test_image": "PCBData/" + img_rel[:-4] + "_test.jpg",
                "source_temp_image": "PCBData/" + img_rel[:-4] + "_temp.jpg",
                "source_label": "PCBData/" + txt_rel,
                "output_image": f"datasets/deeppcb/images/{split}/{base}.jpg",
                "output_label": f"datasets/deeppcb/labels/{split}/{base}.txt",
                "num_boxes": n,
            })

    handle(trainval, lambda b: "val" if b in val_ids else "train")
    handle(test, lambda b: "test")

    # 源 temp 图存在性 (manifest 引用它, 顺带校验)
    missing_temp = [r["image_id"] for r in manifest_rows
                    if not os.path.exists(os.path.join(os.path.dirname(PCB),
                                          r["source_temp_image"].replace("/", os.sep)))]

    # manifest.csv
    mpath = os.path.join(OUT, "manifest.csv")
    with open(mpath, "w", encoding="utf-8-sig", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        wr.writeheader()
        wr.writerows(manifest_rows)

    # deeppcb.yaml
    ypath = os.path.join(OUT, "deeppcb.yaml")
    out_posix = OUT.replace("\\", "/")
    with open(ypath, "w", encoding="ascii", newline="\n") as f:
        f.write(
            "# DeepPCB -> YOLOv5 dataset config (generated by 02_convert_to_yolo.py)\n"
            f"# split: official test.txt -> test; trainval.txt -> 900 train / 100 val, seed={SEED}\n"
            f"path: {out_posix}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n"
            "nc: 6\n"
            f"names: [{', '.join(NAMES)}]\n"
        )

    # conversion_report.md
    rpath = os.path.join(LOGS, "conversion_report.md")
    with open(rpath, "w", encoding="utf-8", newline="\n") as f:
        f.write("# DeepPCB -> YOLO 转换报告\n\n")
        f.write("生成脚本: `02_convert_to_yolo.py`(本报告全部数字由该脚本实际运行产生)\n\n")
        f.write("## 转换公式(逐图实际读取 W,H, 不硬编码 640)\n\n```\n")
        f.write("x_center = ((x1 + x2) / 2) / W\n")
        f.write("y_center = ((y1 + y2) / 2) / H\n")
        f.write("width    = (x2 - x1) / W\n")
        f.write("height   = (y2 - y1) / H\n")
        f.write("class_id = type - 1\n```\n\n")
        f.write("## 类别映射\n\n")
        f.write("| 源 type | 类名 | YOLO class_id |\n|---|---|---|\n")
        for i, n in enumerate(NAMES):
            f.write(f"| {i+1} | {n} | {i} |\n")
        f.write("\n注: copper 即 spurious copper(余铜), 代码与 yaml 统一写 `copper`;"
                " pin_hole 统一用下划线。\n\n")
        f.write(f"## 划分\n\n- 随机种子: **{SEED}**\n")
        f.write("- 官方 `test.txt` 500 张 → test(仅最终测试)\n")
        f.write("- 官方 `trainval.txt` 1000 张 → train 900 / val 100\n")
        f.write("- val 配额按 group 最大余数法分配, 组内 `random.Random(seed).sample` 抽取\n\n")
        f.write("### 各 group 的 val 配额\n\n| group | trainval 图数 | val 配额 |\n|---|---|---|\n")
        for g, n, q in quota_table:
            f.write(f"| {g} | {n} | {q} |\n")
        f.write(f"| 合计 | {sum(n for _, n, _ in quota_table)} "
                f"| {sum(q for _, _, q in quota_table)} |\n")
        f.write("\n## 输出统计\n\n| split | 图像数 | 缺陷框数 |\n|---|---|---|\n")
        for s in ("train", "val", "test"):
            f.write(f"| {s} | {split_stat[s][0]} | {split_stat[s][1]} |\n")
        f.write(f"\n## 非 640x640 图像\n\n")
        if non640:
            f.write("| image_id | W | H |\n|---|---|---|\n")
            for b, W, H in non640:
                f.write(f"| {b} | {W} | {H} |\n")
        else:
            f.write("无。全部图像实测均为 640x640, 归一化按每张图实际 W,H 计算。\n")
        f.write(f"\n## 源模板图缺失\n\n{missing_temp if missing_temp else '无'}\n")

    print("[OK] conversion done")
    for s in ("train", "val", "test"):
        print(f"  {s}: images={split_stat[s][0]} boxes={split_stat[s][1]}")
    print(f"  non-640 images: {len(non640)}; missing temp: {len(missing_temp)}")
    print(f"  manifest: {mpath}")
    print(f"  yaml:     {ypath}")
    print(f"  report:   {rpath}")


if __name__ == "__main__":
    main()
