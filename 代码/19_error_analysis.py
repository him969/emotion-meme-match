# -*- coding: utf-8 -*-
r"""
19_error_analysis.py —— 第 1 步：最终模型在验证集上的错误分析（只读推理，不改模型）

产出：
  结果\diagnose_v2_val.json          所有统计数字（便于报告直接引用）
  结果\diagnose_v2_val.md            可读报告（自动生成表格 + 解读）
  结果\diagnose_confusion_v2_val.png 混淆矩阵（计数 + 行归一化两联图）
  结果\diagnose_errors_v2_val.png    误分类样例网格（带 true/pred/FER+ 标签）

分析内容：
  ① 总体 Acc / Macro-F1 / 每类 P/R/F1 / 混淆矩阵；
  ② 最易混淆的类别对（近义情绪边界）；
  ③ 置信度分布：正确 vs 错误、分桶准确率、ECE、温度缩放（T*）对 NLL/ECE 的影响；
  ④ 标签噪声代理分析（用 FER+ 众包标签）：
     - 错误样本里有多少"模型预测恰好等于 FER+ 标签"（说明可能是原标签噪声）；
     - 在原标签与 FER+ 标签一致的"无争议子集"上的准确率（真实模型错误）；
  ⑤ top1-top2 概率间隔（margin）分布：错误样本是否只是"差一点"。

用法：
  $env:PYTHONIOENCODING='utf-8'
  python 代码\19_error_analysis.py
  # 可选 --weights 模型\best_model.pth --tta（默认单模型无 TTA；CPU 也能跑）
"""
import argparse
import importlib.util
import json
import os
import pathlib
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "代码"
DATA_DIR = ROOT / "数据"
MODEL_DIR = ROOT / "模型"
RESULT_DIR = ROOT / "结果"
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(ROOT / "_deps"))


def load_demo_module():
    spec = importlib.util.spec_from_file_location("demo07", CODE_DIR / "07_demo_app.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def predict_batch(model, xb, device, demo, tta: bool = False):
    """整批推理（可选 6 视角 TTA），返回 (logits, probs)，都是 numpy。"""
    with torch.no_grad():
        views = demo.tta_views(xb) if tta else [xb]
        probs = None
        for v in views:
            p = torch.softmax(model(v.to(device)), dim=1).cpu().numpy()
            probs = p if probs is None else probs + p
        probs = probs / len(views)
    logits = np.log(np.clip(probs, 1e-12, None))
    return logits, probs


def ece_score(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error。"""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc = (pred[mask] == y[mask]).mean()
        avg_conf = conf[mask].mean()
        ece += (mask.sum() / len(y)) * abs(acc - avg_conf)
    return float(ece)


def fit_temperature(probs: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """在验证集上网格搜索温度 T（最小化 NLL），返回 (T*, nll_before, nll_after)。"""
    logits = np.log(np.clip(probs, 1e-12, None))
    def nll(temp: float) -> float:
        z = logits / temp
        z = z - z.max(axis=1, keepdims=True)
        logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
        return float(-logp[np.arange(len(y)), y].mean())
    nll_before = nll(1.0)
    best_t, best_nll = 1.0, nll_before
    for t in np.linspace(0.5, 3.0, 51):
        v = nll(float(t))
        if v < best_nll:
            best_t, best_nll = float(t), v
    return best_t, nll_before, best_nll


def main() -> None:
    ap = argparse.ArgumentParser(description="最终模型验证集错误分析（只读）")
    ap.add_argument("--weights", type=str, default=str(MODEL_DIR / "best_model.pth"))
    ap.add_argument("--tta", action="store_true", help="是否用 6 视角 TTA（CPU 上会慢约 6 倍）")
    ap.add_argument("--out-prefix", type=str, default="diagnose_v2_val")
    ap.add_argument("--max-montage", type=int, default=24)
    args = ap.parse_args()

    demo = load_demo_module()
    CLASS = demo.CLASS_NAMES
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}（CUDA 可用={torch.cuda.is_available()}）；TTA={args.tta}")

    with np.load(DATA_DIR / "fer2013" / "fer2013.npz") as z:
        X_val = z["X_val"]
        y_orig = z["y_val"].astype(np.int64)
    fp_path = DATA_DIR / "fer2013" / "ferplus_labels.npz"
    if fp_path.exists():
        with np.load(fp_path) as z:
            fp_valid = z["valid_val"].astype(bool)
            y_fp = z["y_hard_val"].astype(np.int64)
    else:
        fp_valid = np.zeros(len(X_val), dtype=bool)
        y_fp = np.full(len(X_val), -1, dtype=np.int64)
        print("⚠ 缺少 FER+ 标签，跳过标签噪声分析")

    ckpt = demo.load_checkpoint(pathlib.Path(args.weights), device)
    model = demo.build_model_from_ckpt(ckpt, device)
    size, rgb = demo.model_input_spec(ckpt)
    print(f"权重：{pathlib.Path(args.weights).name} | {ckpt.get('profile')} @ {size}×{size} | "
          f"epoch={ckpt.get('epoch')} | val_acc={ckpt.get('val_acc', 0):.2%}")

    # ---- 推理 ----
    preds, probs_all = [], []
    bs = 128
    with torch.no_grad():
        for i in range(0, len(X_val), bs):
            batch = [demo.preprocess_image(Image.fromarray(x, mode="L"), ckpt["norm_mean"],
                                            ckpt["norm_std"], out_size=size, rgb=rgb)
                     for x in X_val[i:i + bs]]
            xb = torch.cat(batch, dim=0)
            _, p = predict_batch(model, xb, device, demo, tta=args.tta)
            probs_all.append(p)
            preds.append(p.argmax(axis=1))
            if (i // bs) % 5 == 0:
                print(f"  推理进度 {min(i+bs, len(X_val))}/{len(X_val)}", flush=True)
    probs = np.concatenate(probs_all)
    pred = np.concatenate(preds)
    n = len(y_orig)

    # ---- ① 总体/每类/混淆 ----
    acc = float((pred == y_orig).mean())
    macro_f1 = float(f1_score(y_orig, pred, average="macro"))
    p_r, r_r, f_r, sup = precision_recall_fscore_support(y_orig, pred, labels=list(range(7)))
    cm = confusion_matrix(y_orig, pred, labels=list(range(7)))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    per_class = {CLASS[i]: {"precision": round(float(p_r[i]), 4), "recall": round(float(r_r[i]), 4),
                            "f1": round(float(f_r[i]), 4), "support": int(sup[i])} for i in range(7)}

    # 最易混淆对（off-diagonal 计数最高）
    pairs = []
    for i in range(7):
        for j in range(7):
            if i != j and cm[i, j] > 0:
                pairs.append({"true": CLASS[i], "pred": CLASS[j], "count": int(cm[i, j]),
                              "rate": round(float(cm_norm[i, j]), 4)})
    pairs.sort(key=lambda d: -d["count"])
    top_pairs = pairs[:8]

    # ---- ③ 置信度 / margin / 校准 ----
    conf = probs.max(axis=1)
    margin = np.sort(probs, axis=1)[:, -1] - np.sort(probs, axis=1)[:, -2]
    correct = pred == y_orig
    bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0001)]
    bin_stats = []
    for lo, hi in bins:
        m = (conf >= lo) & (conf < hi)
        if m.sum():
            bin_stats.append({"range": f"{lo:.1f}-{min(hi, 1.0):.1f}", "n": int(m.sum()),
                              "acc": round(float((pred[m] == y_orig[m]).mean()), 4),
                              "avg_conf": round(float(conf[m].mean()), 4)})
    ece = ece_score(probs, y_orig)
    T, nll0, nll1 = fit_temperature(probs, y_orig)
    # 温度缩放后 ECE（温度不改变 argmax，只改变置信度）
    logits = np.log(np.clip(probs, 1e-12, None)) / T
    logits = logits - logits.max(axis=1, keepdims=True)
    p_cal = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    ece_cal = ece_score(p_cal, y_orig)

    # ---- ④ 标签噪声代理（FER+）----
    noise_stats = {}
    if fp_valid.any():
        vm = fp_valid
        noise_stats["ferplus_coverage"] = round(float(vm.mean()), 4)
        noise_stats["orig_ferplus_agree"] = round(float((y_orig[vm] == y_fp[vm]).mean()), 4)
        err = ~correct
        err_v = err & vm
        noise_stats["n_errors_all"] = int(err.sum())
        noise_stats["n_errors_valid"] = int(err_v.sum())
        # 错误样本中，模型预测恰好等于 FER+ 标签（即"按 FER+ 标准它是对的"）
        matches_fp = (pred[err_v] == y_fp[err_v])
        noise_stats["errors_matching_ferplus"] = int(matches_fp.sum())
        noise_stats["errors_matching_ferplus_rate"] = round(float(matches_fp.mean()), 4)
        # 无争议子集（原标签 == FER+）上的准确率 = 真实模型错误
        clear = vm & (y_orig == y_fp)
        noise_stats["n_clear"] = int(clear.sum())
        noise_stats["acc_on_clear"] = round(float((pred[clear] == y_orig[clear]).mean()), 4)
        noise_stats["n_errors_on_clear"] = int((pred[clear] != y_orig[clear]).sum())
        # 歧义子集（原标签 != FER+）上的准确率（按原标签评）
        amb = vm & (y_orig != y_fp)
        noise_stats["n_ambiguous"] = int(amb.sum())
        noise_stats["acc_on_ambiguous_vs_orig"] = round(float((pred[amb] == y_orig[amb]).mean()), 4)
        noise_stats["acc_on_ambiguous_vs_ferplus"] = round(float((pred[amb] == y_fp[amb]).mean()), 4)

    result = {
        "meta": {"weights": str(pathlib.Path(args.weights)), "device": str(device), "tta": bool(args.tta),
                 "n_val": n, "model_epoch": ckpt.get("epoch"),
                 "ckpt_val_acc": ckpt.get("val_acc"), "ckpt_val_macro_f1": ckpt.get("val_macro_f1")},
        "overall": {"acc": round(acc, 4), "macro_f1": round(macro_f1, 4)},
        "per_class": per_class,
        "confusion": {"matrix": cm.tolist(), "row_normalized": np.round(cm_norm, 4).tolist(),
                      "labels": CLASS},
        "top_confusions": top_pairs,
        "confidence": {
            "mean_conf_correct": round(float(conf[correct].mean()), 4),
            "mean_conf_wrong": round(float(conf[~correct].mean()), 4),
            "bins": bin_stats, "ece": round(ece, 4),
            "temperature": {"T_star": round(T, 3), "nll_before": round(nll0, 4),
                            "nll_after": round(nll1, 4), "ece_after": round(ece_cal, 4)},
            "margin": {"correct_mean": round(float(margin[correct].mean()), 4),
                       "wrong_mean": round(float(margin[~correct].mean()), 4),
                       "wrong_frac_margin_lt_0.10": round(float((margin[~correct] < 0.10).mean()), 4),
                       "wrong_frac_margin_lt_0.20": round(float((margin[~correct] < 0.20).mean()), 4)},
        },
        "label_noise_proxy": noise_stats,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULT_DIR / f"{args.out_prefix}.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 混淆矩阵图 ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    im0 = axes[0].imshow(cm, cmap="Blues")
    axes[0].set_title("Confusion (counts)")
    im1 = axes[1].imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Confusion (row-normalized)")
    for ax, mat, fmt in ((axes[0], cm, "d"), (axes[1], cm_norm, ".2f")):
        ax.set_xticks(range(7)); ax.set_yticks(range(7))
        ax.set_xticklabels(CLASS, rotation=45, ha="right"); ax.set_yticklabels(CLASS)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        for i in range(7):
            for j in range(7):
                ax.text(j, i, format(mat[i, j], fmt), ha="center", va="center",
                        color="white" if (mat[i, j] / (mat.max() + 1e-9)) > 0.5 else "black", fontsize=7)
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / f"{args.out_prefix.replace('diagnose_', 'diagnose_confusion_')}.png", dpi=150)
    plt.close(fig)

    # ---- 误分类样例网格 ----
    wrong_idx = np.flatnonzero(~correct)
    rng = np.random.default_rng(0)
    chosen = rng.choice(wrong_idx, size=min(args.max_montage, len(wrong_idx)), replace=False)
    cols = 6
    rows = (len(chosen) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.4))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, idx in zip(axes, chosen):
        ax.imshow(X_val[idx], cmap="gray")
        fp_txt = f" | FER+:{CLASS[y_fp[idx]]}" if (fp_valid[idx] and y_fp[idx] >= 0) else ""
        ax.set_title(f"true:{CLASS[y_orig[idx]]}\npred:{CLASS[pred[idx]]} ({conf[idx]:.2f}){fp_txt}",
                     fontsize=7)
        ax.axis("off")
    fig.suptitle(f"Misclassified val samples (n={len(chosen)}/{len(wrong_idx)})", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / f"{args.out_prefix.replace('diagnose_', 'diagnose_errors_')}.png", dpi=150)
    plt.close(fig)

    # ---- Markdown 报告 ----
    lines = [f"# 错误分析报告：{pathlib.Path(args.weights).name}（val {n} 张，TTA={args.tta}）", "",
             f"- 设备：`{device}`；模型：`{ckpt.get('profile')}` @ {size}×{size}；第 {ckpt.get('epoch')} 轮",
             f"- **总体：Acc {acc:.2%}，Macro-F1 {macro_f1:.4f}**", "",
             "## 每类指标（按真实类别）", "",
             "| 类别 | precision | recall | f1 | support |", "|---|---|---|---|---|"]
    for c in CLASS:
        d = per_class[c]
        lines.append(f"| {c} | {d['precision']:.4f} | {d['recall']:.4f} | {d['f1']:.4f} | {d['support']} |")
    lines += ["", "## Top 混淆对（真实→预测）", "", "| 真实 | 预测 | 数量 | 该类错误率 |", "|---|---|---|---|"]
    for p in top_pairs:
        lines.append(f"| {p['true']} | {p['pred']} | {p['count']} | {p['rate']:.2%} |")
    lines += ["", "## 置信度 / 校准", "",
              f"- 正确样本平均置信度：{result['confidence']['mean_conf_correct']:.3f}；"
              f"错误样本：{result['confidence']['mean_conf_wrong']:.3f}",
              f"- ECE：{ece:.4f}；温度缩放 T*={T:.2f}（NLL {nll0:.4f} → {nll1:.4f}，ECE→{ece_cal:.4f}）",
              f"- margin（top1-top2）：正确均值 {result['confidence']['margin']['correct_mean']:.3f}，"
              f"错误均值 {result['confidence']['margin']['wrong_mean']:.3f}；"
              f"错误中 margin<0.10 占 {result['confidence']['margin']['wrong_frac_margin_lt_0.10']:.1%}", ""]
    if noise_stats:
        lines += ["## 标签噪声代理（FER+ 参照）", "",
                  f"- FER+ 覆盖 {noise_stats['ferplus_coverage']:.1%}；与原标签一致率 {noise_stats['orig_ferplus_agree']:.1%}",
                  f"- 错误样本 {noise_stats['n_errors_valid']} 个（FER+ 有效子集）：其中 **{noise_stats['errors_matching_ferplus_rate']:.1%}** "
                  f"的模型预测恰好等于 FER+ 标签（可能是原标签噪声，而非模型错）",
                  f"- 无争议子集（原标签==FER+，n={noise_stats['n_clear']}）：准确率 **{noise_stats['acc_on_clear']:.2%}**，"
                  f"真实错误 {noise_stats['n_errors_on_clear']} 个",
                  f"- 歧义子集（原标签≠FER+，n={noise_stats['n_ambiguous']}）：按原标签 {noise_stats['acc_on_ambiguous_vs_orig']:.1%}，"
                  f"按 FER+ 标签 {noise_stats['acc_on_ambiguous_vs_ferplus']:.1%}", ""]
    lines += ["## 产物", "",
              f"- 数字：`{out_json.relative_to(ROOT)}`",
              f"- 混淆矩阵：`结果\\{args.out_prefix.replace('diagnose_', 'diagnose_confusion_')}.png`",
              f"- 错误样例：`结果\\{args.out_prefix.replace('diagnose_', 'diagnose_errors_')}.png`", ""]
    (RESULT_DIR / f"{args.out_prefix}.md").write_text("\n".join(lines), encoding="utf-8")

    # ---- 控制台摘要 ----
    print("\n" + "=" * 88)
    print(f"总体：Acc {acc:.2%}  Macro-F1 {macro_f1:.4f}")
    print("每类 recall：" + "；".join(f"{c} {per_class[c]['recall']:.1%}" for c in CLASS))
    print("\nTop 混淆对：")
    for p in top_pairs[:6]:
        print(f"  {p['true']:9s} -> {p['pred']:9s}  {p['count']:4d} 次（该真实类错误率 {p['rate']:.1%}）")
    print(f"\n置信度：正确均值 {result['confidence']['mean_conf_correct']:.3f} vs 错误 {result['confidence']['mean_conf_wrong']:.3f}")
    print(f"校准：ECE {ece:.4f}；T*={T:.2f} -> ECE {ece_cal:.4f}，NLL {nll0:.4f}->{nll1:.4f}")
    print(f"错误样本 margin：正确均值 {result['confidence']['margin']['correct_mean']:.3f} vs 错误 {result['confidence']['margin']['wrong_mean']:.3f}")
    if noise_stats:
        print(f"\nFER+ 标签噪声代理：错误中 {noise_stats['errors_matching_ferplus_rate']:.1%} 预测等于 FER+ 标签")
        print(f"无争议子集（{noise_stats['n_clear']} 张）准确率：{noise_stats['acc_on_clear']:.2%}（真实错误 {noise_stats['n_errors_on_clear']} 个）")
    print(f"\n报告：{RESULT_DIR / (args.out_prefix + '.md')}")
    print("=" * 88)


if __name__ == "__main__":
    main()
