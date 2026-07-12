# DeepPCB -> YOLO 转换报告

生成脚本: `02_convert_to_yolo.py`(本报告全部数字由该脚本实际运行产生)

## 转换公式(逐图实际读取 W,H, 不硬编码 640)

```
x_center = ((x1 + x2) / 2) / W
y_center = ((y1 + y2) / 2) / H
width    = (x2 - x1) / W
height   = (y2 - y1) / H
class_id = type - 1
```

## 类别映射

| 源 type | 类名 | YOLO class_id |
|---|---|---|
| 1 | open | 0 |
| 2 | short | 1 |
| 3 | mousebite | 2 |
| 4 | spur | 3 |
| 5 | copper | 4 |
| 6 | pin_hole | 5 |

注: copper 即 spurious copper(余铜), 代码与 yaml 统一写 `copper`; pin_hole 统一用下划线。

## 划分

- 随机种子: **20260712**
- 官方 `test.txt` 500 张 → test(仅最终测试)
- 官方 `trainval.txt` 1000 张 → train 900 / val 100
- val 配额按 group 最大余数法分配, 组内 `random.Random(seed).sample` 抽取

### 各 group 的 val 配额

| group | trainval 图数 | val 配额 |
|---|---|---|
| group00041 | 200 | 20 |
| group13000 | 198 | 20 |
| group20085 | 291 | 29 |
| group44000 | 50 | 5 |
| group50600 | 50 | 5 |
| group77000 | 100 | 10 |
| group92000 | 111 | 11 |
| 合计 | 1000 | 100 |

## 输出统计

| split | 图像数 | 缺陷框数 |
|---|---|---|
| train | 900 | 6199 |
| val | 100 | 674 |
| test | 500 | 3140 |

## 非 640x640 图像

无。全部图像实测均为 640x640, 归一化按每张图实际 W,H 计算。

## 源模板图缺失

无
