# A 快筛：focal loss / 类别平衡（2026-11-02）

## 结论
**单模型 No-Go**：focal loss（γ=1 + 类别权重）没有提升总体指标，macro-F1 反而略降；
**集成有一点多样性价值但很小**：3-seed + focal 的 4 模型集成 val 74.03/0.7211 vs 3-seed 73.92/0.7194（+0.11pp / +0.17pp），不值得为它把演示改成 4 模型。

## 实验设置
- 模型：efficientnet_b0@160，seed42，v2 同配方（Lion 1e-4、LS 0.1、mixup 0.2、强增强、EMA 0.999、warmup 2、clip 1.0、AMP）
- 唯一改动：`--balance focal`（focal loss γ=1.0，仍带类别权重）
- 训练：24 轮早停（最佳 epoch14 EMA），总用时 30.7 分钟
- 权重：`模型\v2_focal_s42.pth`（新文件，不覆盖现有权重）

## 结果（validation，3589 张；test 未使用）

| 模型 | 无 TTA Acc / F1 | 6 视角 TTA Acc / F1 |
|---|---|---|
| v2 seed42 基线 | 72.61% / 0.7036 | **73.00% / 0.7046** |
| focal seed42 | 72.69% / 0.6992 | 73.00% / 0.7021 |

每类 recall 对比（focal vs 基线）：
- fear 54.0% vs 53.2%（+0.8pp，唯一明显受益的弱类）
- neutral 75.1% vs 73.8%（+1.3pp）
- happy 89.4% vs 88.3%（+1.1pp）
- **sad 60.0% vs 61.9%（−1.9pp）**、disgust 58.9% vs 60.7%（−1.8pp）、surprise 84.8% vs 86.8%（−2.0pp）

混淆结构基本没变：sad→neutral 16.8%、fear→sad 17.5% 仍是最主要错误。

## 集成检验（focal 是否能提供多样性）

| 集成配置 | val Acc | Macro-F1 |
|---|---|---|
| 3-seed（v2 原方案） | 73.92% | 0.7194 |
| 3-seed + focal（4 模型） | **74.03%** | **0.7211** |

提升 +0.11pp / +0.17pp，属于噪声量级；且演示要多加载一个模型、多一份显存/推理，不划算。

## 后续建议
- **不把 focal 模型设为默认/替换主模型**；可保留为实验记录（`v2_focal_s42.pth`）。
- 如果继续做"弱类专项"，比 focal 更值得试的是：**fear/sad 定向过采样或分层分类**（但需注意测试集仍只评一次）。
- 产物：`结果\evaluation_v2_focal_s42_val*.json/.txt`、`结果\evaluation_v2plusfocal_val_tta.*`、`结果\train_v2_focal_s42.json`、`结果\training_curves_v2_focal_s42.png`、实验记录 `结果\v2_records\v2_focal_s42.json`
