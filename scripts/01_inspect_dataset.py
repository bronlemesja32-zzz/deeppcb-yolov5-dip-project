# -*- coding: utf-8 -*-
"""
Step 1: DeepPCB 数据集结构与标注格式验证
- 校验 trainval.txt / test.txt 中每条记录对应的 _test.jpg / _temp.jpg / .txt 是否存在
- 解析全部标注文件: 分隔符、字段数、类别取值、坐标合法性
- 用 PIL 读取全部 test 图像的实际尺寸
- 统计各 split 的类别分布与缺陷框宽高
所有输出均来自对磁盘文件的实际读取, 不做任何假设。
"""
import os
import sys
from collections import Counter, defaultdict

from PIL import Image

ROOT = r"C:\Users\qintx\Desktop\clauded\DeepPCB"
PCB = os.path.join(ROOT, "PCBData")

CLASS_NAMES = {1: "open", 2: "short", 3: "mousebite", 4: "spur", 5: "copper", 6: "pin-hole"}


def load_split(name):
    path = os.path.join(PCB, name)
    entries = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                print(f"[异常] {name}:{ln} 字段数 {len(parts)} != 2: {line!r}")
                continue
            entries.append((parts[0], parts[1]))
    return entries


def main():
    trainval = load_split("trainval.txt")
    test = load_split("test.txt")
    print(f"trainval.txt 条目数: {len(trainval)}")
    print(f"test.txt     条目数: {len(test)}")

    missing = []
    sep_counter = Counter()
    field_counter = Counter()
    cls_per_split = {"trainval": Counter(), "test": Counter()}
    box_w, box_h = [], []
    coord_anomalies = []
    img_sizes = Counter()
    defects_per_img = []
    checked_imgs = 0

    for split_name, entries in (("trainval", trainval), ("test", test)):
        for img_rel, txt_rel in entries:
            base = img_rel[:-4]  # 去掉 .jpg
            test_img = os.path.join(PCB, base + "_test.jpg")
            temp_img = os.path.join(PCB, base + "_temp.jpg")
            txt = os.path.join(PCB, txt_rel)
            for p in (test_img, temp_img, txt):
                if not os.path.exists(p):
                    missing.append(p)
                    continue

            # 实际图像尺寸 (只读头, 快)
            if os.path.exists(test_img):
                with Image.open(test_img) as im:
                    img_sizes[(im.width, im.height, im.mode)] += 1
                checked_imgs += 1

            if not os.path.exists(txt):
                continue
            with open(txt, encoding="utf-8") as f:
                n_def = 0
                for ln, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    if "," in line:
                        sep_counter["comma"] += 1
                        parts = line.split(",")
                    else:
                        sep_counter["space"] += 1
                        parts = line.split()
                    field_counter[len(parts)] += 1
                    if len(parts) != 5:
                        coord_anomalies.append(f"{txt_rel}:{ln} 字段数={len(parts)}: {line!r}")
                        continue
                    try:
                        x1, y1, x2, y2, t = map(int, parts)
                    except ValueError:
                        coord_anomalies.append(f"{txt_rel}:{ln} 非整数: {line!r}")
                        continue
                    n_def += 1
                    cls_per_split[split_name][t] += 1
                    box_w.append(x2 - x1)
                    box_h.append(y2 - y1)
                    if not (0 <= x1 < x2 and 0 <= y1 < y2):
                        coord_anomalies.append(f"{txt_rel}:{ln} 坐标顺序异常: {line!r}")
                    if x2 > 640 or y2 > 640:
                        coord_anomalies.append(f"{txt_rel}:{ln} 坐标超出640: {line!r}")
                    if t not in CLASS_NAMES:
                        coord_anomalies.append(f"{txt_rel}:{ln} 未知类别 {t}: {line!r}")
                defects_per_img.append(n_def)

    print(f"\n缺失文件数: {len(missing)}")
    for m in missing[:10]:
        print("  缺失:", m)

    print(f"\n实际读取的 test 图像数: {checked_imgs}")
    print("图像尺寸分布 (宽, 高, 模式) -> 数量:")
    for k, v in img_sizes.most_common():
        print(f"  {k}: {v}")

    print(f"\n标注行分隔符统计: {dict(sep_counter)}")
    print(f"每行字段数统计: {dict(field_counter)}")

    total = Counter()
    for split_name in ("trainval", "test"):
        c = cls_per_split[split_name]
        total.update(c)
        print(f"\n[{split_name}] 缺陷框总数 = {sum(c.values())}")
        for t in sorted(c):
            print(f"  type={t} ({CLASS_NAMES.get(t, '?')}): {c[t]}")
    print(f"\n[全集] 缺陷框总数 = {sum(total.values())}")
    for t in sorted(total):
        print(f"  type={t} ({CLASS_NAMES.get(t, '?')}): {total[t]}")

    if box_w:
        import statistics as st
        print(f"\n缺陷框宽度: min={min(box_w)} max={max(box_w)} mean={st.mean(box_w):.1f}")
        print(f"缺陷框高度: min={min(box_h)} max={max(box_h)} mean={st.mean(box_h):.1f}")
        print(f"每图缺陷数: min={min(defects_per_img)} max={max(defects_per_img)} "
              f"mean={st.mean(defects_per_img):.2f}")

    print(f"\n坐标/字段/类别异常条数: {len(coord_anomalies)}")
    for a in coord_anomalies[:20]:
        print("  异常:", a)

    # group90100 中 temp 比 test 多一张, 找出未配对的 temp
    g = os.path.join(PCB, "group90100", "90100")
    temps = {f[:-9] for f in os.listdir(g) if f.endswith("_temp.jpg")}
    tests = {f[:-9] for f in os.listdir(g) if f.endswith("_test.jpg")}
    print(f"\ngroup90100: 未配对的 temp 基名 = {sorted(temps - tests)}, "
          f"未配对的 test 基名 = {sorted(tests - temps)}")


if __name__ == "__main__":
    main()
