# -*- coding: utf-8 -*-
r"""
21_match_variants.py —— 匹配层小实验（val-only，不训练、不改模型、不碰测试集）

对比 6 种"人脸情绪概率 → meme 情绪"的匹配方式：
  ① hard           : 取 argmax 情绪（硬匹配）
  ② soft           : 当前默认软匹配 score=Σ p_i·sim(i,j)（手工 7×7 矩阵）
  ③ soft_T         : 软匹配前先做温度缩放 T=0.65（F 方案的已知结果）
  ④ blend_α        : p_mix=(1-α)·onehot(argmax)+α·p，score=p_mix·SIM；α 在训练折上选
  ⑤ gated_θ        : maxp≥θ 时用 hard，否则用 soft；θ 在训练折上选
  ⑥ confusion_sim  : 用训练折的模型混淆矩阵估计有向相似度，再向手工矩阵收缩
                     （S=λ·S_conf+(1-λ)·SIM0）；λ 在训练折上选
评估两个口径：
  · 严格 label_hit：选中 meme 情绪 == 人脸真实情绪（历史指标）
  · 软命中率：严格命中 或 sim(选中, 真实) ≥ τ（τ=0.6/0.7，衡量"相似类算半对"）

所有超参只在训练折（val 的 4/5）上选，5 折交叉验证估计真实收益；测试集完全不参与。
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
import torch
from PIL import Image
from sklearn.model_selection import KFold

ROOT = pathlib.Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "代码"
DATA_DIR = ROOT / "数据"
MODEL_DIR = ROOT / "模型"
RESULT_DIR = ROOT / "结果"
sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(ROOT / "_deps"))


def load_demo():
    spec = importlib.util.spec_from_file_location("demo07", CODE_DIR / "07_demo_app.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def infer_probs(demo, ckpt, X, device, tta=True, bs=128):
    size, rgb = demo.model_input_spec(ckpt)
    model = demo.build_model_from_ckpt(ckpt, device)
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            batch = [demo.preprocess_image(Image.fromarray(x, mode="L"), ckpt["norm_mean"],
                                            ckpt["norm_std"], out_size=size, rgb=rgb)
                     for x in X[i:i + bs]]
            xb = torch.cat(batch, dim=0)
            views = demo.tta_views(xb) if tta else [xb]
            p = None
            for v in views:
                pp = torch.softmax(model(v.to(device)), dim=1).cpu().numpy()
                p = pp if p is None else p + pp
            out.append(p / len(views))
    return np.concatenate(out)


def pred_hard(p):
    return p.argmax(axis=1)


def pred_soft(p, sim):
    return (p @ sim).argmax(axis=1)


def onehot(p):
    out = np.zeros_like(p)
    out[np.arange(len(p)), p.argmax(axis=1)] = 1.0
    return out


def pred_blend(p, sim, alpha):
    return pred_soft((1.0 - alpha) * onehot(p) + alpha * p, sim)


def pred_gated(p, sim, theta):
    return np.where(p.max(axis=1) >= theta, pred_hard(p), pred_soft(p, sim))


def confusion_sim(y, pred_hard_, n=7):
    """用混淆矩阵构造对称相似度：S_ij=(C_ij+C_ji)/(C_ii+C_jj+C_ij+C_ji)。"""
    C = np.zeros((n, n), dtype=np.float64)
    for t, q in zip(y, pred_hard_):
        C[int(t), int(q)] += 1
    S = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            denom = C[i, i] + C[j, j] + C[i, j] + C[j, i]
            v = (C[i, j] + C[j, i]) / denom if denom > 0 else 0.0
            S[i, j] = S[j, i] = float(np.clip(v, 0.0, 1.0))
    return S


def strict_hit(pred, y):
    return float((pred == y).mean())


def soft_hit(pred, y, sim, tau):
    same = pred == y
    similar = sim[y, pred] >= tau
    return float((same | similar).mean())


def tune_alpha(p, y, sim, grid=np.arange(0.0, 1.001, 0.05)):
    best_a, best_hit = 1.0, strict_hit(pred_soft(p, sim), y)
    for a in grid:
        h = strict_hit(pred_blend(p, sim, float(a)), y)
        if h > best_hit + 1e-9:
            best_a, best_hit = float(a), h
    return best_a


def tune_theta(p, y, sim, grid=np.arange(0.40, 0.91, 0.05)):
    best_t, best_hit = 0.40, -1.0
    for t in grid:
        h = strict_hit(pred_gated(p, sim, float(t)), y)
        if h > best_hit + 1e-9:
            best_t, best_hit = float(t), h
    return best_t


def tune_lambda(p, y, sim0, grid=np.arange(0.0, 1.001, 0.1)):
    best_l, best_hit = 0.0, strict_hit(pred_soft(p, sim0), y)
    S = confusion_sim(y, pred_hard(p))
    for lam in grid:
        sim_mix = sim0 * (1.0 - lam) + S * lam
        np.fill_diagonal(sim_mix, 1.0)
        h = strict_hit(pred_soft(p, sim_mix), y)
        if h > best_hit + 1e-9:
            best_l, best_hit = float(lam), h
    return best_l


def main():
    ap = argparse.ArgumentParser(description="匹配层变体对比（val-only, 5 折 CV）")
    ap.add_argument("--weights", type=str, default=str(MODEL_DIR / "best_model.pth"))
    ap.add_argument("--out-prefix", type=str, default="match_variants")
    args = ap.parse_args()

    demo = load_demo()
    SIM0 = demo.EMOTION_SIM.copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")

    with np.load(DATA_DIR / "fer2013" / "fer2013.npz") as z:
        X_val, y_val = z["X_val"], z["y_val"].astype(np.int64)
    ckpt = demo.load_checkpoint(pathlib.Path(args.weights), device)
    probs = infer_probs(demo, ckpt, X_val, device, tta=True)
    print(f"推理完成：{probs.shape}（6 视角 TTA）；基线 val_acc={strict_hit(pred_hard(probs), y_val):.4f}")

    p = probs
    hard = pred_hard(p)
    soft = pred_soft(p, SIM0)
    p_T = demo.apply_temperature(p, demo.TEMP_FOR_MATCH)
    softT = pred_soft(p_T, SIM0)

    # ---- 全量 val 上的乐观结果 ----
    a_full, theta_full, lam_full = tune_alpha(p, y_val, SIM0), tune_theta(p, y_val, SIM0), tune_lambda(p, y_val, SIM0)
    S_full = confusion_sim(y_val, hard)
    sim_blend_full = SIM0 * (1 - lam_full) + S_full * lam_full
    np.fill_diagonal(sim_blend_full, 1.0)
    variants_full = {
        "hard": hard,
        "soft": soft,
        "soft_T0.65": softT,
        f"blend_a={a_full:.2f}": pred_blend(p, SIM0, a_full),
        f"gated_theta={theta_full:.2f}": pred_gated(p, SIM0, theta_full),
        f"conf_sim_lambda={lam_full:.1f}": pred_soft(p, sim_blend_full),
    }
    print("\n【全量 val（乐观）】")
    for name, pred in variants_full.items():
        print(f"  {name:28s} strict={strict_hit(pred, y_val):.4f} "
              f"soft@0.6={soft_hit(pred, y_val, SIM0, 0.6):.4f} "
              f"soft@0.7={soft_hit(pred, y_val, SIM0, 0.7):.4f}")

    # ---- 5 折 CV：超参只在训练折上选 ----
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    names = ["hard", "soft", "soft_T0.65", "blend_alpha", "gated_theta", "conf_sim"]
    vals = {k: [] for k in names}
    softs = {(k, tau): [] for k in names for tau in (0.6, 0.7)}
    chosen = {"alpha": [], "theta": [], "lambda": []}
    for tr, te in kf.split(p):
        ptr, ytr, pte, yte = p[tr], y_val[tr], p[te], y_val[te]
        hm = {"hard": pred_hard(pte), "soft": pred_soft(pte, SIM0),
              "soft_T0.65": pred_soft(demo.apply_temperature(pte, demo.TEMP_FOR_MATCH), SIM0)}
        a = tune_alpha(ptr, ytr, SIM0)
        th = tune_theta(ptr, ytr, SIM0)
        lam = tune_lambda(ptr, ytr, SIM0)
        S = confusion_sim(ytr, pred_hard(ptr))
        sim_mix = SIM0 * (1 - lam) + S * lam
        np.fill_diagonal(sim_mix, 1.0)
        hm["blend_alpha"] = pred_blend(pte, SIM0, a)
        hm["gated_theta"] = pred_gated(pte, SIM0, th)
        hm["conf_sim"] = pred_soft(pte, sim_mix)
        chosen["alpha"].append(a); chosen["theta"].append(th); chosen["lambda"].append(lam)
        for k in names:
            vals[k].append(strict_hit(hm[k], yte))
            for tau in (0.6, 0.7):
                softs[(k, tau)].append(soft_hit(hm[k], yte, SIM0, tau))

    print("\n【5 折交叉验证（诚实估计）】")
    cv = {}
    for k in names:
        m, s = float(np.mean(vals[k])), float(np.std(vals[k]))
        cv[k] = {"strict_mean": m, "strict_std": s,
                 "soft@0.6": float(np.mean(softs[(k, 0.6)])),
                 "soft@0.7": float(np.mean(softs[(k, 0.7)]))}
        print(f"  {k:14s} strict={m:.4f}±{s:.4f}  soft@0.6={cv[k]['soft@0.6']:.4f}  soft@0.7={cv[k]['soft@0.7']:.4f}")

    result = {
        "meta": {"weights": str(pathlib.Path(args.weights)), "device": str(device),
                 "n_val": len(y_val), "tta": True, "cv_folds": 5,
                 "note": "超参在训练折选，5 折 CV 估计；测试集未参与"},
        "full_val_optimistic": {k: {"strict": round(strict_hit(v, y_val), 4),
                                    "soft@0.6": round(soft_hit(v, y_val, SIM0, 0.6), 4),
                                    "soft@0.7": round(soft_hit(v, y_val, SIM0, 0.7), 4)}
                                for k, v in variants_full.items()},
        "cv": cv,
        "chosen_params_per_fold": chosen,
        "chosen_full": {"alpha": a_full, "theta": theta_full, "lambda": lam_full},
        "confusion_sim_full": np.round(S_full, 3).tolist(),
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / f"{args.out_prefix}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [f"# 匹配层变体对比（val {len(y_val)} 张，6 视角 TTA；val-only）", "",
             "| 方案 | 严格 label_hit（5 折 CV） | 软命中@0.6 | 软命中@0.7 |", "|---|---|---|---|"]
    for k in names:
        d = cv[k]
        lines.append(f"| {k} | {d['strict_mean']:.4f} ± {d['strict_std']:.4f} | "
                     f"{d['soft@0.6']:.4f} | {d['soft@0.7']:.4f} |")
    lines += ["", "全量 val 上的乐观结果：", ""]
    for k, d in result["full_val_optimistic"].items():
        lines.append(f"- {k}: strict={d['strict']:.4f}, soft@0.6={d['soft@0.6']:.4f}, soft@0.7={d['soft@0.7']:.4f}")
    lines += ["", "说明：",
              "- hard=取 argmax；soft=当前默认软匹配（手工相似矩阵）；soft_T0.65=温度缩放后软匹配；",
              "- blend_alpha 在 argmax one-hot 与概率之间插值；gated_theta 高置信用 hard、低置信用 soft；",
              "- conf_sim 用训练折混淆矩阵估计相似度并向手工矩阵收缩；",
              "- 所有超参只在训练折选，报告为 5 折 CV；测试集未参与。"]
    (RESULT_DIR / f"{args.out_prefix}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告：{RESULT_DIR / (args.out_prefix + '.md')}")


if __name__ == "__main__":
    main()
