# P0 A 快筛结论：FER+ 标签重标注（2026-11-02）

> 目的：验证"把训练/验证标签换成 FER+ 众包标签"能否提升模型。
> 结论：**按项目既定指标（原始 FER2013 val 标签）为 No-Go，不替换主模型、不做 UI 选项。**
> 但发现一个有价值的现象：**在两套标签一致的样本子集上，FER+ 模型反而更好（+3.1pp Acc / +4.2pp F1）**。

## 1. 数据准备（`代码\18_ferplus_labels.py`）

- 下载 `fer2013new.csv`（1.53 MB，GitHub microsoft/FERPlus，引用 Barsoum et al. 2016）
- 按 Usage 分块 + 行序对齐（**实测 35,887 行、拆分完全一致**，不依赖 Image name）
- 映射：`contempt→disgust`；`unknown/NF` 不计票；全票 unknown/NF 的样本丢弃
- 产物：`数据\fer2013\ferplus_labels.npz`（硬标签 + 软标签 + valid mask）、`数据\fer2013\ferplus_labels_meta.json`
- 丢弃：train 150 / val 10 / test 16；与原始标签一致率：train 64.3% / val 63.6% / test 63.8%
- FER+ 训练集分布：neutral 9433 / happy 7616 / sad 3934 / surprise 3481 / angry 2714 / fear 812 / disgust 569（比原始 FER2013 更均衡，disgust 436→569、fear 大幅增加）

## 2. 训练配置（A 快筛：1 seed，v2 同配方）

- 模型 `efficientnet_b0@160`，seed 42，Lion lr=1e-4、wd=1e-4、label smoothing 0.1、mixup 0.2、强增强、EMA 0.999、warmup 2、grad-clip 1.0、AMP、类别权重、early stop patience 10
- 训练：**早停于第 26 轮（最佳 epoch 16，EMA 权重）**，总耗时约 **35 分钟**（含评估/SWA）
- 新权重：`模型\ferplus_hard_s42.pth`（**不覆盖任何现有权重**；ckpt 内 `labels="ferplus"`）
- 记录（修复 08 的 `import re` 缺失 bug 后补写）：`结果\v2_records\ferplus_hard_s42.json`、`结果\train_ferplus_hard_s42.json`、`结果\training_curves_ferplus_hard_s42.png`、`结果\ferplus_logs\ferplus_hard_s42.log`

## 3. 评估结果（validation，3589 张；test 未使用）

| 模型 | 评估标签口径 | Acc | Macro-F1 |
|---|---|---|---|
| v2 原标签模型（`best_model.pth`，seed42） | 原始标签全集 | **72.61%** | **0.7036** |
| FER+ 硬标签模型（`ferplus_hard_s42.pth`） | 原始标签全集 | 61.27% | 0.5539 |
| FER+ 硬标签模型（TTA） | 原始标签全集 | 61.55% | 0.5526 |
| v2 原标签模型 | FER+ 标签（有效 3579 张） | 65.74% | 0.5765 |
| FER+ 硬标签模型 | FER+ 标签（有效 3579 张） | **84.16%** | **0.7689** |
| v2 原标签模型 | **两套标签一致子集（2276 张）** | 87.21% | 0.8138 |
| FER+ 硬标签模型 | **两套标签一致子集（2276 张）** | **90.29%** | **0.8554** |

（原始标签全集上 v2 seed42 为 72.61/0.7036，与历史记录 72.64/0.7039 的小差异来自评估实现：本表批量推理 vs 06 的逐图口径，属正常浮点/批处理差异。）

## 4. 解释与决策

1. **按项目既定指标（原始 FER2013 val 标签）**：FER+ 模型 61.27% < 72.61%，**未过 Go/No-Go 门槛** → 不替换主模型、不做 UI 选项；继续做 B（软标签 3 seeds）预期不会改变"标签定义不同"这一根本问题，故**不进入 B**（省 3~4h GPU）；
2. **为什么看起来差这么多**：FER+ 与原标签只有 ~64% 一致，FER+ 模型是"按 FER+ 标准"学的；在原标签上评它，等于用另一套答案判卷，因此掉 11pp；
3. **有价值的发现**：在**两套标签一致的 2276 张（63.4%）**上，FER+ 模型 Acc 90.29% vs 原模型 87.21%（+3.1pp），Macro-F1 0.8554 vs 0.8138（+4.2pp）——说明 FER+ 模型在"标签无争议"的样本上更强；剩下差异主要是对**边界/模糊样本**的标注定义不同；
4. **项目取舍**：所有历史成绩（v1 69.71%、v2 75.93%、label_hit 0.6773 等）都以原始 FER2013 标签为准；若换成 FER+ 标签体系，需要重新定义评估口径与所有对比数字，超出本次课程项目范围。因此**保持 v2 主模型不变**，把本次结果作为"负结论 + 标签体系分析"写进报告/PPT。

## 5. 产物清单

- 数据：`数据\fer2013\fer2013new.csv`、`ferplus_labels.npz`、`ferplus_labels_meta.json`
- 模型：`模型\ferplus_hard_s42.pth`
- 结果：`结果\evaluation_ferplus_hard_s42_val.json/.txt`、`结果\evaluation_ferplus_hard_s42_val_tta_tta.json/.txt`、`结果\train_ferplus_hard_s42.json`、`结果\training_curves_ferplus_hard_s42.png`、`结果\v2_records\ferplus_hard_s42.json`
- 代码：`代码\18_ferplus_labels.py`（新增）、`代码\03_dataset.py`（新增可选 `--labels ferplus`，默认行为不变）、`代码\08_transfer_train.py`（新增 `--labels`，并修复 `import re` 缺失 bug）
