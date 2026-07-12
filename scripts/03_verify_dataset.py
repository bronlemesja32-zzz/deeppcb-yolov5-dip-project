# -*- coding: utf-8 -*-
"""
Step 3: 转换结果自检 (独立于转换脚本, 全部从磁盘重新读取)

检查项:
 1. 每个 split 的图像数 / 标签数 / 缺陷框数
 2. 每类实例数 (per split + 合计)
 3. 空标签文件数
 4. 非法框数 (字段数!=5 / class_id 超出 0-5 / 数值不可解析 / w<=0 或 h<=0)
 5. 标签坐标是否全部在 [0,1] (含框边缘 xc±w/2, yc±h/2, 容差 1e-6 对应 6 位小数舍入)
 6. 图像与标签是否一一对应
 7. test split 是否严格等于官方 test.txt 的 500 个 id
 8. train/val 是否严格划分官方 trainval.txt 的 1000 个 id (不相交, 900/100)
 9. 每张图的框数与源标注逐一对比; 类别分布与源标注 (type-1) 逐类对比
"""
import os
import sys
from collections import Counter, defaultdict

PROJ = os.path.dirname(os.path.abspath(__file__))
PCB = os.path.join(os.path.dirname(PROJ), "DeepPCB", "PCBData")
OUT = os.path.join(PROJ, "datasets", "deeppcb")
LOGS = os.path.join(PROJ, "logs")

NAMES = ["open", "short", "mousebite", "spur", "copper", "pin_hole"]
EPS = 1e-6

failures = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"[{tag}] {name}" + (f" -- {detail}" if detail and not ok else ""))
    if not ok:
        failures.append(f"{name}: {detail}")
    return ok


def load_official(fname):
    ids = {}
    with open(os.path.join(PCB, fname), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            img_rel, txt_rel = line.split()
            base = os.path.basename(img_rel)[:-4]
            ids[base] = txt_rel
    return ids


def main():
    official_test = load_official("test.txt")
    official_trainval = load_official("trainval.txt")

    report = ["# 转换自检报告\n", "生成脚本: `03_verify_dataset.py`, 独立重读磁盘文件。\n"]

    split_ids = {}
    per_split_boxes = {}
    per_split_cls = {}
    empty_labels = defaultdict(list)
    illegal = []
    out_of_range = []
    boxcount_by_id = {}

    for split in ("train", "val", "test"):
        img_dir = os.path.join(OUT, "images", split)
        lbl_dir = os.path.join(OUT, "labels", split)
        imgs = {f[:-4] for f in os.listdir(img_dir) if f.endswith(".jpg")}
        lbls = {f[:-4] for f in os.listdir(lbl_dir) if f.endswith(".txt")}
        split_ids[split] = imgs

        check(f"{split}: image/label one-to-one", imgs == lbls,
              f"only-img={sorted(imgs - lbls)[:5]} only-lbl={sorted(lbls - imgs)[:5]}")

        nbox = 0
        cls = Counter()
        for b in sorted(lbls):
            p = os.path.join(lbl_dir, b + ".txt")
            with open(p, encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            if not lines:
                empty_labels[split].append(b)
            boxcount_by_id[b] = len(lines)
            for ln, line in enumerate(lines, 1):
                parts = line.split()
                if len(parts) != 5:
                    illegal.append(f"{split}/{b}:{ln} fields={len(parts)}")
                    continue
                try:
                    cid = int(parts[0]); xc, yc, w, h = map(float, parts[1:])
                except ValueError:
                    illegal.append(f"{split}/{b}:{ln} unparsable")
                    continue
                if not 0 <= cid <= 5:
                    illegal.append(f"{split}/{b}:{ln} cid={cid}")
                    continue
                if w <= 0 or h <= 0:
                    illegal.append(f"{split}/{b}:{ln} w={w} h={h}")
                    continue
                if not (-EPS <= xc - w/2 and xc + w/2 <= 1 + EPS and
                        -EPS <= yc - h/2 and yc + h/2 <= 1 + EPS and
                        0 <= xc <= 1 and 0 <= yc <= 1 and w <= 1 and h <= 1):
                    out_of_range.append(f"{split}/{b}:{ln} {line}")
                    continue
                cls[cid] += 1
                nbox += 1
        per_split_boxes[split] = nbox
        per_split_cls[split] = cls

    # split 与官方文件的严格对应
    check("test == official test.txt ids",
          split_ids["test"] == set(official_test),
          f"diff={len(split_ids['test'] ^ set(official_test))}")
    tv_out = split_ids["train"] | split_ids["val"]
    check("train+val == official trainval.txt ids",
          tv_out == set(official_trainval), f"diff={len(tv_out ^ set(official_trainval))}")
    check("train/val disjoint", not (split_ids["train"] & split_ids["val"]))
    check("sizes 900/100/500",
          (len(split_ids["train"]), len(split_ids["val"]), len(split_ids["test"]))
          == (900, 100, 500),
          f"got {[len(split_ids[s]) for s in ('train', 'val', 'test')]}")

    check("no illegal boxes", not illegal, f"n={len(illegal)} e.g. {illegal[:3]}")
    check("all coords in [0,1]", not out_of_range,
          f"n={len(out_of_range)} e.g. {out_of_range[:3]}")
    n_empty = sum(len(v) for v in empty_labels.values())
    check("empty labels == 0", n_empty == 0,
          f"n={n_empty} e.g. {dict((k, v[:3]) for k, v in empty_labels.items())}")

    # 与源标注逐图对比框数、逐类对比分布
    src_cls = Counter()
    mismatch = []
    src_txt_of = {}
    src_txt_of.update(official_trainval)
    src_txt_of.update(official_test)
    for b, txt_rel in src_txt_of.items():
        n = 0
        with open(os.path.join(PCB, txt_rel.replace("/", os.sep)), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                t = int(line.split()[4])
                src_cls[t - 1] += 1
                n += 1
        if boxcount_by_id.get(b) != n:
            mismatch.append(f"{b}: src={n} out={boxcount_by_id.get(b)}")
    check("per-image box count matches source", not mismatch,
          f"n={len(mismatch)} e.g. {mismatch[:3]}")

    total_cls = Counter()
    for s in per_split_cls:
        total_cls.update(per_split_cls[s])
    check("class distribution matches source (type-1)", total_cls == src_cls,
          f"out={dict(total_cls)} src={dict(src_cls)}")

    # ---- 报告 ----
    report.append("\n## 每 split 统计\n")
    report.append("| split | 图像数 | 标签数 | 缺陷框数 | 空标签数 |")
    report.append("|---|---|---|---|---|")
    for s in ("train", "val", "test"):
        report.append(f"| {s} | {len(split_ids[s])} | {len(split_ids[s])} "
                      f"| {per_split_boxes[s]} | {len(empty_labels[s])} |")
    report.append("\n## 每类实例数\n")
    report.append("| class_id | 类名 | train | val | test | 合计 |")
    report.append("|---|---|---|---|---|---|")
    for i, n in enumerate(NAMES):
        tr = per_split_cls["train"][i]
        va = per_split_cls["val"][i]
        te = per_split_cls["test"][i]
        report.append(f"| {i} | {n} | {tr} | {va} | {te} | {tr + va + te} |")
    report.append(f"\n## 合法性\n")
    report.append(f"- 非法框数: {len(illegal)}")
    report.append(f"- 坐标越界 [0,1] 框数: {len(out_of_range)} (容差 {EPS})")
    report.append(f"- 空标签文件数: {n_empty}")
    report.append(f"- 与源标注逐图框数不一致: {len(mismatch)}")
    report.append("\n## 划分与官方文件对应\n")
    report.append(f"- test 严格等于官方 test.txt: {split_ids['test'] == set(official_test)}")
    report.append(f"- train+val 严格等于官方 trainval.txt: {tv_out == set(official_trainval)}")
    report.append(f"- train/val 交集为空: {not (split_ids['train'] & split_ids['val'])}")
    report.append(f"- 数量 train/val/test = {len(split_ids['train'])}/"
                  f"{len(split_ids['val'])}/{len(split_ids['test'])}")
    report.append(f"\n## 结论\n\n{'全部检查通过。' if not failures else '存在失败项: ' + '; '.join(failures)}")

    os.makedirs(LOGS, exist_ok=True)
    rpath = os.path.join(LOGS, "verify_report.md")
    with open(rpath, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(report) + "\n")
    print(f"report: {rpath}")
    print(f"RESULT: {'ALL CHECKS PASSED' if not failures else f'{len(failures)} FAILURES'}")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
