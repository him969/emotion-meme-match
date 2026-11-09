# A 方案记录（2026-11-07：进行中→已完成，结论 No-Go）

> **最终状态（2026-11-07 20:05）：已完成全部 3 seeds + 集成 + privateTest 一次评估；结果 75.17% / 0.7400，低于 v2 的 75.93% / 0.7443 → No-Go，不替换主模型。**
> 本文件前半部分记录暂停时的断点（历史）；最终结果见文末 §7。
> 训练进程已停止，GPU 已释放；v3 权重已归档到 `模型\_archived_v3_expand\`（不出现在演示权重列表）；测试集共评估 2 次（v2 最终、A 方案最终），A 未胜出。

## 1. A 方案定义（已和用户确认）

把 FER2013 的 **publicTest（原 val，3,589 张）并入训练集**；因为不能再用它做验证，采用：
- 从合并后的 32,298 张里**按类别分层留出 1,501 张**做早停/选轮；
- 实际训练集 **30,797 张**（28,709 + 2,088，+7.3%）；测试集 privateTest 3,589 张**保持不动**；
- 配方沿用 v2：efficientnet_b0@160、Lion、AMP、mixup 0.2、label smoothing 0.1、EMA 0.999、SWA last5、strong aug、30 轮、patience 10；
- 计划 3 seeds（42/43/44）+ 3-seed 集成 + 6 视角 TTA，最后 **privateTest 只评一次**（这将是第 2 次 test 评估：v2 最终、A 方案最终）。

## 2. 已完成（代码 + 数据）

| 项 | 内容 |
|---|---|
| `代码\03_dataset.py` | `build_transfer_loaders(...)` 新增 **可选** 参数 `merge_val=False, val_holdout=0, split_seed=42`；默认 False，**旧行为完全不变**（demo/selftest/其他脚本不受影响）；打印合并/留出信息；stats 里带 merge 元信息 |
| `代码\08_transfer_train.py` | 新增 `--merge-val`、`--val-holdout`（默认 1500）、`--split-seed`（默认 42）；训练日志打印这些参数；`--drop-path` 之前已加（默认 0） |
| 数据核对 | `train=30,797` / `holdout=1,501` / `test=3,589`；holdout 类别分布 `[207, 23, 213, 377, 255, 167, 259]`；放大到 160×160 用时约 14s；类别权重 `[1.034, 9.381, 1.004, 0.569, 0.842, 1.287, 0.828]` |
| 编译 | `py_compile` 03/08 均通过 |

## 3. 训练断点（被中断）

| 任务 | 日志 | 状态 | 已保存的 partial ckpt（**不要用**） |
|---|---|---|---|
| `v3_expand_s42` | `结果\diagnose_logs\v3_expand_s42.log` | 跑到 **epoch 15/30** 被手动停止 | `模型\_paused_v3_partial\v3_expand_s42.pth`：epoch10 EMA，holdout acc=71.89 / F1=0.6966 |
| `v3_expand_s43` | `结果\diagnose_logs\v3_expand_s43.log` | 跑到 **epoch 15/30** 被手动停止 | `模型\_paused_v3_partial\v3_expand_s43.pth`：epoch14 EMA，holdout acc=72.15 / F1=0.6915 |
| `v3_expand_s44` | — | **未启动** | — |

> 注意：partial 权重已从 `模型\` 移到 `模型\_paused_v3_partial\`，**不会出现在演示页的权重选择里**；仅作断点记录，不可用。

> 注意：holdout 只有 1,501 张，acc 噪声比原 val（3,589）大；partial 数字仅说明"训练在正常收敛"，**不能拿来对比 v2 的 val**，更不能用于 test。

## 4. 重新开跑的命令（原样复制即可）

```powershell
$env:PYTHONIOENCODING='utf-8'
$py='python'
$p='<项目根>'

# seed 42
& $py "$p\代码\08_transfer_train.py" --model efficientnet_b0 --img-size 160 --epochs 30 `
  --batch-size 64 --optimizer lion --lr 1e-4 --weight-decay 1e-4 `
  --label-smoothing 0.1 --mixup 0.2 --ema-decay 0.999 --swa-last 5 --warmup-epochs 2 `
  --grad-clip 1.0 --amp --aug strong --seed 42 --patience 10 `
  --merge-val --val-holdout 1500 --split-seed 42 `
  --name v3_expand_s42 --note "A方案扩训练集：publicTest并入（留1500早停），v2配方 seed42"
# seed 43 同上，改 --seed 43 --name v3_expand_s43
# seed 44 同上，改 --seed 44 --name v3_expand_s44
```

- 建议：**seed42/43 并行**（约 55~65min/个，GPU 占用约 5.5GB），完成后单独跑 seed44；
- 全部完成后：
  ```powershell
  & $py "$p\代码\15_ensemble_eval.py" --weights "$p\模型\v3_expand_s42.pth" "$p\模型\v3_expand_s43.pth" "$p\模型\v3_expand_s44.pth" --name v3_expand_ensemble --tta --splits val
  # 先在 holdout 上不能直接用 15（它按官方 val 划分），需要确认；正式评估：
  & $py "$p\代码\15_ensemble_eval.py" --weights ... --name v3_expand_ensemble --tta --splits test
  ```
  （15_ensemble_eval 的 `--splits val` 仍会用旧 publicTest 做 val，但该数据已并入训练，**不要用 val 选模型/报结果**；最终只报 privateTest 一次。）
- 预计剩余时间：2 个并行 ~1h + seed44 ~45min + 集成评估 ~5min ≈ **1.5~2h**（不含 GPU 争用）。

## 5. 当前系统状态

- 默认模型/演示逻辑**未改变**：主模型仍是 `best_model.pth`（v2，test 75.93% / Macro-F1 0.7443）；
- **测试集评估次数仍为 1 次**（v2 最终），A 方案未跑 test；
- 演示服务在暂停训练时**已停止**（8501 无监听）；恢复命令：
  ```powershell
  & 'python' -m streamlit run "$p\代码\07_demo_app.py" --server.port 8501 --server.address 127.0.0.1 --server.headless true --browser.gatherUsageStats false
  ```
- GPU：已释放（nvidia-smi 仅桌面占用约 1.35GB）；
- 之前已完成并归档的内容不受影响：meme 扩库 58→66、多标签负结论、B/E 负结论（详见报告 §5.8/§6.10）。

## 6. 恢复后第一步建议

1. 先确认 `git`/备份？项目无 git；直接按上面命令重跑 3 seeds；
2. 3 seeds 完成前不要碰 `privateTest`；完成后只评一次并记录"test 第 2 次"；
3. 若集成结果 >75.93%，再决定是否替换默认模型（需更新 `07` 默认权重组、报告、PPT）；若没提升，A 方案同样归档为负结论，模型侧收工。

## 7. 最终结果（2026-11-07 20:05 完成，No-Go）

3 个 seed 全部重训完成（留出集 1,501 张，只用于早停/观测；未参与训练）：

| seed | 训练用时 | 最佳 epoch | 留出集 acc / F1 | 归档权重 |
|---|---|---|---|---|
| 42 | 43.5 min（21 轮，早停） | 11 (EMA) | 73.35% / 0.7156 | `模型\_archived_v3_expand\v3_expand_s42.pth` |
| 43 | 45.0 min（22 轮，早停） | 12 (EMA) | 72.55% / 0.6942 | `模型\_archived_v3_expand\v3_expand_s43.pth` |
| 44 | 50.3 min（28 轮，早停） | 18 (EMA) | 72.29% / 0.6988 | `模型\_archived_v3_expand\v3_expand_s44.pth` |
| **3-seed 集成** | — | — | **74.15% / 0.7224** | — |

**privateTest（唯一一次评估，本项目第 2 次 test）**：

| 模型 | test Acc | test Macro-F1 |
|---|---|---|
| v2 最终（3-seed，28,709 训练） | **75.93%** | **0.7443** |
| A 方案（3-seed 扩训练集，30,797） | 75.17% | 0.7400 |
| 差值 | **−0.76pp** | **−0.43pp** |

**结论**：
- A 方案（publicTest 并入训练，+7.3% 数据）**没有带来提升，反而略降**；留出集上的 74.15% 不能代表 privateTest（留出子集分布偏易）；
- 说明当前 FER2013 的瓶颈更可能是**标签噪声/类别定义模糊**，而不是训练样本数量；
- **不替换主模型、不进入默认集成**；v3 权重归档（`模型\_archived_v3_expand\`），默认仍是 v2；
- 测试集评估次数从 1 变为 2（如实记录：v2 最终、A 方案最终）；后续若再改模型，需谨慎控制 test 使用。
