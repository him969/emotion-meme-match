# -*- coding: utf-8 -*-
r"""
22_multi_label_match.py —— 多情绪 meme 匹配实验 v2（val-only；不改默认、不加 UI）

发现 v1 的 `score=p·SIM·q`（加权平均）在数学上**永远打不过 one-hot**（加权平均 ≤ 最大分量），
所以 v1 的 multi_rate=0。本版测试 3 类"能让多标签 meme 参与竞争"的打分方式：

  baseline    : one-hot 软匹配（当前默认）：score_m = s[dominant_m]
  weighted    : q·s（v1 的加权平均；预计仍然打不过 one-hot，保留作对照）
  cosine      : (q·s)/||q||（归一化对齐）
  maxlabel    : max_{j: q_j≥0.3} s_j（"它至少能表达其中一种"）
  gated_cos   : pmax<θ 时才允许多标签 meme 参与（用 cosine 打分），否则只允许 one-hot
  gated_max   : pmax<θ 时允许多标签 meme（用 maxlabel 打分），否则只允许 one-hot

其中 s = p·SIM（7 类得分），pmax=max(p)。θ 在 5 折训练折上选（最大化 coverage，其次 strict）。

指标（全量 + 歧义子集 pmax<0.5 / <0.6）：
  strict    : 选中 meme 的 dominant == 真实情绪（历史口径）
  coverage  : 真实情绪 ∈ 选中 meme 的标签集合（q≥0.3）
  multi_rate: 选中多标签 meme 的比例（必须一起看，防止"模糊 meme 霸榜"）
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


def evaluate(pred_idx, y, q, dom_idx, multi_mask):
    strict = dom_idx[pred_idx] == y
    cover = np.array([bool(q[pred_idx[k], int(y[k])] >= 0.3) for k in range(len(y))])
    mrate = multi_mask[pred_idx]
    return {"strict": strict, "coverage": cover, "multi_rate": mrate}


def summ(m, mask=None):
    if mask is not None:
        m = {k: v[mask] for k, v in m.items()}
    return {"n": int(len(m["strict"])), "strict": float(m["strict"].mean()),
            "coverage": float(m["coverage"].mean()), "multi_rate": float(m["multi_rate"].mean())}


def main():
    ap = argparse.ArgumentParser(description="多情绪 meme 匹配实验 v2（val-only）")
    ap.add_argument("--weights", type=str, default=str(MODEL_DIR / "best_model.pth"))
    ap.add_argument("--out-prefix", type=str, default="multi_label_match")
    args = ap.parse_args()

    demo = load_demo()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")

    with np.load(DATA_DIR / "fer2013" / "fer2013.npz") as z:
        X_val, y_val = z["X_val"], z["y_val"].astype(np.int64)
    ckpt = demo.load_checkpoint(pathlib.Path(args.weights), device)
    memes = demo.load_memes()
    q, info = demo.load_multi_labels(memes)
    cls2idx = {n: i for i, n in enumerate(demo.CLASS_NAMES)}
    dom_idx = memes["emotion"].map(cls2idx).to_numpy()
    multi_mask = np.array([f in info for f in memes["file"]], dtype=bool)
    print(f"meme：{len(memes)} 张；多标签 {len(info)} 张；q 行和="
          f"{q.sum(axis=1).min():.3f}~{q.sum(axis=1).max():.3f}")

    probs = infer_probs(demo, ckpt, X_val, device, tta=True)
    S = probs @ demo.EMOTION_SIM                       # (N,7) 每类得分
    pmax = probs.max(axis=1)
    amb5, amb6 = pmax < 0.5, pmax < 0.6

    onehot = S[:, dom_idx]                              # (N,M) 当前软匹配
    weighted = S @ q.T                                  # (N,M) 加权平均（v1 方式）
    qnorm = np.linalg.norm(q, axis=1)                   # (M,)
    cosine = weighted / qnorm[None, :]
    labels_mask = q >= 0.3                              # (M,7)
    maxlabel = np.where(labels_mask[None, :, :], S[:, None, :], -np.inf).max(axis=2)

    def gated_score(score, theta):
        """pmax≥θ（明确照片）时把多标签 meme 的分数置为 -inf；pmax<θ 时全部参与。"""
        out = score.copy()
        hi = pmax >= theta
        out[np.ix_(hi, multi_mask)] = -np.inf
        return out

    def pick(score):
        return score.argmax(axis=1)

    # θ 网格 + 每折选择（最大化训练折 coverage，其次 strict）
    grid = np.round(np.arange(0.50, 0.91, 0.05), 2)
    rules = {"baseline": onehot, "weighted": weighted, "cosine": cosine, "maxlabel": maxlabel}
    full = {}
    for name, sc in rules.items():
        m = evaluate(pick(sc), y_val, q, dom_idx, multi_mask)
        full[name] = {"all": summ(m), "amb<0.5": summ(m, amb5), "amb<0.6": summ(m, amb6)}

    for rule_name, base_sc in (("cosine", cosine), ("maxlabel", maxlabel)):
        for theta in grid:
            sc = gated_score(base_sc, float(theta))
            m = evaluate(pick(sc), y_val, q, dom_idx, multi_mask)
            full[f"gated_{rule_name}_theta={theta:.2f}"] = {
                "all": summ(m), "amb<0.5": summ(m, amb5), "amb<0.6": summ(m, amb6)}

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    cv = {}
    for rule_name, base_sc in (("gated_cosine", cosine), ("gated_maxlabel", maxlabel)):
        rows = []
        for tr, te in kf.split(probs):
            best_theta, best_key = grid[0], (-1.0, -1.0)
            for theta in grid:
                sc = gated_score(base_sc, float(theta))
                m = evaluate(pick(sc[tr]), y_val[tr], q, dom_idx, multi_mask)
                key = (float(m["coverage"].mean()), float(m["strict"].mean()))
                if key > best_key:
                    best_key, best_theta = key, float(theta)
            sc_te = gated_score(base_sc, best_theta)[te]
            m_te = evaluate(pick(sc_te), y_val[te], q, dom_idx, multi_mask)
            rows.append({"theta": best_theta,
                         "strict": float(m_te["strict"].mean()),
                         "coverage": float(m_te["coverage"].mean()),
                         "multi_rate": float(m_te["multi_rate"].mean())})
        df = pd.DataFrame(rows)
        cv[rule_name] = {"theta_per_fold": df["theta"].tolist(),
                         "strict": float(df["strict"].mean()),
                         "coverage": float(df["coverage"].mean()),
                         "multi_rate": float(df["multi_rate"].mean())}

    result = {"meta": {"weights": str(pathlib.Path(args.weights)), "device": str(device),
                       "n_val": int(len(y_val)), "tta": True, "n_memes": int(len(memes)),
                       "n_multi_memes": int(len(info)), "cv_folds": 5,
                       "note": "val-only；测试集未参与；默认行为未修改；"
                               "v1 的加权平均在数学上被 one-hot 支配，故引入 cosine/maxlabel/gated"},
              "n_ambiguous": {"pmax<0.5": int(amb5.sum()), "pmax<0.6": int(amb6.sum())},
              "full_val": full, "cv": cv, "multi_label_memes": info}
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / f"{args.out_prefix}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = [f"# 多情绪 meme 匹配实验 v2（val {len(y_val)} 张，6 视角 TTA；val-only）", "",
             f"- 多标签 meme：**{len(info)}** 张；歧义照片：pmax<0.5 **{int(amb5.sum())}** 张、"
             f"pmax<0.6 **{int(amb6.sum())}** 张", "",
             "## 全量 val（θ 为 gated 门限；真实增益看 CV）", "",
             "| 方案 | strict | coverage | multi_rate |", "|---|---|---|---|"]
    for k, d in full.items():
        lines.append(f"| {k} | {d['all']['strict']:.4f} | {d['all']['coverage']:.4f} | {d['all']['multi_rate']:.4f} |")
    lines += ["", "## 歧义子集（pmax<0.5）", "",
              "| 方案 | strict | coverage | multi_rate |", "|---|---|---|---|"]
    for k, d in full.items():
        lines.append(f"| {k} | {d['amb<0.5']['strict']:.4f} | {d['amb<0.5']['coverage']:.4f} | "
                     f"{d['amb<0.5']['multi_rate']:.4f} |")
    lines += ["", "## 5 折交叉验证（θ 只在训练折选）", ""]
    for k, d in cv.items():
        lines.append(f"- **{k}**：strict **{d['strict']:.4f}**，coverage **{d['coverage']:.4f}**，"
                     f"multi_rate {d['multi_rate']:.4f}，θ/折={d['theta_per_fold']}")
    lines += ["", "口径说明：strict=dominant 命中；coverage=真实情绪在选中 meme 的标签集合（q≥0.3）；"
              "multi_rate=选中多标签 meme 比例。多标签会天然抬高 coverage，因此必须结合 multi_rate 与目检判断。"]
    (RESULT_DIR / f"{args.out_prefix}.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n全量 val：")
    for k, d in full.items():
        print(f"  {k:28s} strict={d['all']['strict']:.4f} coverage={d['all']['coverage']:.4f} "
              f"multi_rate={d['all']['multi_rate']:.4f}")
    print("\n5 折 CV：")
    for k, d in cv.items():
        print(f"  {k:14s} strict={d['strict']:.4f} coverage={d['coverage']:.4f} "
              f"multi_rate={d['multi_rate']:.4f} θ={d['theta_per_fold']}")
    print(f"\n报告：{RESULT_DIR / (args.out_prefix + '.md')}")


if __name__ == "__main__":
    main()
