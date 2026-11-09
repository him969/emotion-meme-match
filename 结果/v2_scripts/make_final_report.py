# -*- coding: utf-8 -*-
"""v2 最终汇总：读取 结果\\v2_records\\*.json，生成
  ① 控制台 markdown 表（阶段 1/2/seeds，含是否早停、权重来源）
  ② 结果\\v2_recommendation.md（推荐最优配置 + 3 seed 指标 + 产物路径 + 注意事项）
只读 val 结果，不做任何评估。
"""
import importlib.util
import json
import os
import pathlib
import sys

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = pathlib.Path(__file__).resolve().parents[1]
REC = ROOT / "结果" / "v2_records"
recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(REC.glob("*.json"))]
recs = [r for r in recs if r.get("group") in ("R", "P", "seed")]
by_name = {r["name"]: r for r in recs}


def row(r: dict) -> str:
    early = "是" if int(r.get("epochs_run") or 0) < int(r.get("epochs") or 0) else "否"
    t = r.get("total_time_s")
    tstr = f"{t/60:.1f}" if t else "—"
    return (f"| {r['name']} | {r['model']}@{r['img_size']} | "
            f"{r['best_val_acc']:.2%} | {r['best_val_macro_f1']:.4f} | {r['best_epoch']} | "
            f"{r.get('epochs_run')}/{r.get('epochs')} | {r.get('best_source')} | {tstr} | {early} |")


print("### 全部实验（按分组）")
print("| name | model@img | val_acc | val_macro_f1 | best_ep | 轮数(跑/设) | 权重来源 | 耗时(min) | 早停 |")
print("|---|---|---|---|---|---|---|---|---|")
for r in recs:
    print(row(r))

seeds = [by_name.get(f"v2_seed{s}") for s in (42, 43, 44)]
seeds = [s for s in seeds if s]
print("\n### seeds")
accs = [s["best_val_acc"] for s in seeds]
f1s = [s["best_val_macro_f1"] for s in seeds]
for s in seeds:
    print(row(s))
print(f"\nseed val_acc: mean={np.mean(accs):.2%} std={np.std(accs):.2%} min={min(accs):.2%} max={max(accs):.2%}")
print(f"seed macro_f1: mean={np.mean(f1s):.4f} std={np.std(f1s):.4f} min={min(f1s):.4f} max={max(f1s):.4f}")

best = max(recs, key=lambda r: (float(r.get("best_val_macro_f1", 0)),
                                float(r.get("best_val_acc", 0))))
print(f"\n综合最优：{best['name']}  F1={best['best_val_macro_f1']:.4f}  acc={best['best_val_acc']:.2%}")

# ---------------- 写 recommendation.md ----------------
lines = [
    "# v2 最优配置推荐与复现说明（train-v2 产出，只使用验证集 val，未评估 test）",
    "",
    "## 1. 推荐最优单模型",
    "",
    f"- 文件：`模型\\{best['name']}.pth`（已同时是 `模型\\v2_seed42.pth`）",
    f"- 模型/输入：`{best['model']}@{best['img_size']}`（timm，ImageNet 预训练，灰度 FER 放大到 {best['img_size']} 后复制 3 通道、ImageNet 归一化）",
    f"- val 指标：**val_acc = {best['best_val_acc']:.2%}，val_macro_f1 = {best['best_val_macro_f1']:.4f}**（best_epoch={best['best_epoch']}，来源={best['best_source']}）",
    f"- 对比 v1 历史基线 ResNet18@128 20ep：val 67.51% / macro-F1 0.6557 → v2 提升 "
    f"{best['best_val_acc']*100-67.51:+.2f}pp / {best['best_val_macro_f1']-0.6557:+.4f}",
    "- 权重包字段与 08 v1 完全兼容，06_evaluate.py / 07_demo_app.py 可直接加载（已实测通过）",
    "",
    "## 2. 最优训练配方（R4 = Lion 配方）",
    "",
    "```",
    "python 代码/08_transfer_train.py \\",
    f"  --model {best['model']} --img-size {best['img_size']} --epochs 30 --batch-size 64 \\",
    "  --amp --optimizer lion --lr 1e-4 --weight-decay 1e-4 \\",
    "  --label-smoothing 0.1 --aug strong --mixup 0.2 \\",
    "  --ema-decay 0.999 --swa-last 5 --warmup-epochs 2 --grad-clip 1.0 \\",
    "  --patience 10 --seed 42",
    "```",
    "",
    "关键组件收益（resnet18@128, 30ep, val macro-F1）：",
    "- R1 AMP+AdamW+basic aug+类别权重：0.6457",
    "- R2 +EMA(0.999)+SWA+warmup+clip：0.6546（EMA 收益最明显）",
    "- R3 +mixup0.2+strong aug：0.6587",
    "- R4 +Lion lr1e-4：**0.6722**（Lion 贡献最大 +0.0135）",
    "",
    "## 3. 三个 seed 的 val 指标",
    "",
    "| seed | 文件 | val_acc | val_macro_f1 | best_ep | 说明 |",
    "|---|---|---|---|---|---|",
]
for s in seeds:
    note = s.get("note", "")
    if s.get("timeout_fallback"):
        note = f"23:45 超时保护取最佳 ckpt（已跑 {s.get('epochs_run')} 轮）"
    lines.append(f"| seed{s.get('seed')} | 模型\\{s['name']}.pth | {s['best_val_acc']:.2%} | "
                 f"{s['best_val_macro_f1']:.4f} | {s['best_epoch']} | {note} |")
lines += [
    "",
    f"- 3 seed 均值：val_acc {np.mean(accs):.2%}（std {np.std(accs):.2%}），"
    f"macro-F1 {np.mean(f1s):.4f}（std {np.std(f1s):.4f}）",
    "- 集成增益由 Lead 用 06 或 15 脚本在 val 上计算；三个 seed ckpt 结构一致，可直接概率平均。",
    "",
    "## 4. 阶段 2 backbone 对比结论（均为 R4 配方, 30ep）",
    "",
    "| backbone@img | val_acc | val_macro_f1 | 说明 |",
    "|---|---|---|---|",
]
for nm in ("v2_resnet34_128", "v2_effb0_160"):
    r = by_name.get(nm)
    if r:
        lines.append(f"| {r['model']}@{r['img_size']} | {r['best_val_acc']:.2%} | "
                     f"{r['best_val_macro_f1']:.4f} | |")
lines += [
    "",
    "结论：efficientnet_b0@160 明显优于 resnet18/resnet34；resnet50@128 因 GPU 算力预算未跑"
    "（3 并发时 GPU 99% 饱和会拖垮其他实验，属计划中的可选项）。",
    "",
    "## 5. 产物路径",
    "",
    "- 权重：`模型\\v2_r1_amp.pth`、`v2_r2_ema.pth`、`v2_r3_mix.pth`、`v2_r4_lion.pth`、"
    "`v2_resnet34_128.pth`、`v2_effb0_160.pth`、`v2_seed42/43/44.pth`",
    "- 实验记录：`结果\\experiments_v2.jsonl`、`结果\\v2_experiment_summary.md`、`结果\\v2_records\\`",
    "- 训练历史/曲线：`结果\\train_v2_*.json`、`结果\\training_curves_v2_*.png`",
    "- 完整日志：`结果\\v2_logs\\*.log`",
    "",
    "## 6. 注意事项",
    "",
    "- 所有实验**只用 val 选模型，从未评估 test**；test 评估留给 Lead 最终决定。",
    "- 未覆盖 `模型\\best_model.pth`，未修改 06/07/12/13/14、报告、PROJECT、交接总结、数据。",
    "- 硬性时间约束下：R5（focal 平衡损失）与 P3（resnet50@128）未跑；"
    "seed43/44 若为超时保护 ckpt，其 epochs_run 少于 30。",
    "",
]
(ROOT / "结果" / "v2_recommendation.md").write_text("\n".join(lines), encoding="utf-8")
print(f"\n已写 {ROOT / '结果' / 'v2_recommendation.md'}")

# 重建聚合（复用 08 的函数）
spec = importlib.util.spec_from_file_location("t08", ROOT / "代码" / "08_transfer_train.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.rebuild_aggregates()
print("聚合文件已重建")
