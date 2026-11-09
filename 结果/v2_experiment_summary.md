# 模型优化 v2 实验结果汇总（只统计验证集 val，未评估 test）

- 生成时间：2026-11-07 19:55:28
- 选模型规则：按 **val_macro_f1 优先、val_acc 次之** 挑最佳 epoch；所有实验均未使用 test 集。
- 权重目录：`模型\{experiment_name}.pth`；训练历史：`结果\train_{exp}.json`；曲线：`结果\training_curves_{exp}.png`；完整训练日志：`结果\v2_logs\{exp}.log`。

## 阶段 2：backbone 对比（阶段 1 最优配方）

| 实验名 | 模型 | 输入 | 轮数(跑/设) | 优化器 | AMP | EMA | aug | mixup | balance | LS | **val_macro_f1** | **val_acc** | best_epoch | 耗时(min) | 权重来源 | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2_effb0_160 | efficientnet_b0 | 160 | 30/30 | lion | 开 | 0.999 | strong | 0.2 | none | 0.1 | 0.7039 | 72.64% | 23 | 59.7 | ema | P2: efficientnet_b0@160，R4 最优配方 |
| v2_effv2b0_s42 | tf_efficientnetv2_b0.in1k | 160 | 30/30 | lion | 开 | 0.999 | strong | 0.2 | none | 0.1 | 0.7034 | 71.77% | 22 | 44.4 | ema | E: EfficientNetV2-B0@160, v2 同配方 seed42 |
| v2_resnet34_128 | resnet34 | 128 | 30/30 | lion | 开 | 0.999 | strong | 0.2 | none | 0.1 | 0.6880 | 71.16% | 24 | 41.0 | ema | P1: resnet34@128，R4 最优配方 |

## P0

| 实验名 | 模型 | 输入 | 轮数(跑/设) | 优化器 | AMP | EMA | aug | mixup | balance | LS | **val_macro_f1** | **val_acc** | best_epoch | 耗时(min) | 权重来源 | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ferplus_hard_s42 | efficientnet_b0 | 160 | 26/30 | lion | 开 | 0.999 | strong | 0.2 | none | 0.1 | 0.7689 | 84.16% | 16 | 34.9 | ema | P0 A快筛：FER+硬标签，v2配方同seed42 |

## 阶段 1：配方消融（resnet18@128, 30 epoch, AMP）

| 实验名 | 模型 | 输入 | 轮数(跑/设) | 优化器 | AMP | EMA | aug | mixup | balance | LS | **val_macro_f1** | **val_acc** | best_epoch | 耗时(min) | 权重来源 | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2_r1_amp | resnet18 | 128 | 30/30 | adamw | 开 | 关 | basic | 关 | none | 0.05 | 0.6457 | 68.35% | 21 | 30.6 | raw | R1 基线: adamw+AMP+basic aug+类别权重 |
| v2_r2_ema | resnet18 | 128 | 26/30 | adamw | 开 | 0.999 | basic | 关 | none | 0.05 | 0.6546 | 67.90% | 16 | 29.9 | ema | R2=R1+EMA0.999+SWA5+warmup2+clip1.0 |
| v2_r3_mix | resnet18 | 128 | 30/30 | adamw | 开 | 0.999 | strong | 0.2 | none | 0.1 | 0.6587 | 69.10% | 30 | 37.2 | raw | R3=R2+mixup0.2+strong aug+LS0.1 |
| v2_r4_lion | resnet18 | 128 | 30/30 | lion | 开 | 0.999 | strong | 0.2 | none | 0.1 | 0.6722 | 69.96% | 23 | 35.0 | ema | R4=R3 配置+Lion lr1e-4 |

## 阶段 4：最优配置多种子（seed 42/43/44）

| 实验名 | 模型 | 输入 | 轮数(跑/设) | 优化器 | AMP | EMA | aug | mixup | balance | LS | **val_macro_f1** | **val_acc** | best_epoch | 耗时(min) | 权重来源 | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2_seed42 | efficientnet_b0 | 160 | 30/30 | lion | 开 | 0.999 | strong | 0.2 | none | 0.1 | 0.7039 | 72.64% | 23 | 59.7 | ema | 由 P2 v2_effb0_160 复制（effb0@160, R4 配方, 30ep, seed=42） |
| v2_seed43 | efficientnet_b0 | 160 | 27/30 | lion | 开 | 0.999 | strong | 0.2 | none | 0.1 | 0.7014 | 72.14% | 18 | 53.3 | ema | E2 超时保护：23:45 未跑完即停，取当时最佳 ckpt（已跑 27 轮；耗时约 53.3 min） |
| v2_seed44 | efficientnet_b0 | 160 | 26/30 | lion | 开 | 0.999 | strong | 0.2 | none | 0.1 | 0.7021 | 71.89% | 16 | 53.8 | ema | seed44: 与 P2 相同配置（effb0@160+R4 配方） |

## 其他实验

| 实验名 | 模型 | 输入 | 轮数(跑/设) | 优化器 | AMP | EMA | aug | mixup | balance | LS | **val_macro_f1** | **val_acc** | best_epoch | 耗时(min) | 权重来源 | 备注 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2_droppath_s42 | efficientnet_b0 | 160 | 40/40 | lion | 开 | 0.999 | strong | 0.2 | none | 0.1 | 0.6969 | 71.91% | 29 | 57.5 | raw | B-light: drop_path=0.1 + 40ep @160, v2 同配方 seed42 |
| v2_focal_s42 | efficientnet_b0 | 160 | 24/30 | lion | 开 | 0.999 | strong | 0.2 | focal | 0.1 | 0.6995 | 72.72% | 14 | 30.7 | ema | A：focal loss (gamma=1)+类别权重，其余同 v2 seed42 |
| v3_expand_s42 | efficientnet_b0 | 160 | 21/30 | lion | 开 | 0.999 | strong | 0.2 | none | 0.1 | 0.7156 | 73.35% | 11 | 43.5 | ema | A方案扩训练集：publicTest并入（留1500早停），v2配方 seed42 |
| v3_expand_s43 | efficientnet_b0 | 160 | 22/30 | lion | 开 | 0.999 | strong | 0.2 | none | 0.1 | 0.6942 | 72.55% | 12 | 45.0 | ema | A方案扩训练集：publicTest并入（留1500早停），v2配方 seed43 |
| v3_expand_s44 | efficientnet_b0 | 160 | 28/30 | lion | 开 | 0.999 | strong | 0.2 | none | 0.1 | 0.6988 | 72.29% | 18 | 50.3 | ema | A方案扩训练集：publicTest并入（留1500早停），v2配方 seed44 |

## 当前最优

- **ferplus_hard_s42**（efficientnet_b0@160，val_macro_f1=0.7689，val_acc=84.16%，best_epoch=16）

> 说明：v1 历史基线 ResNet18@128 20ep（fp32/basic aug，仅 val，参考）：val_acc 67.51% / macro-F1 0.6557。v2 全部实验均只做 val，test 集留给 Lead 最终决定后再评估。
