# -*- coding: utf-8 -*-
r"""
20_match_calibration.py —— F 方案：温度缩放 + 情绪相似度矩阵调优（不改模型权重、只改匹配层）

目标：在不重新训练模型的前提下，提高软匹配的 label_hit
      （label_hit = 选中的 meme 情绪 == 人脸真实情绪；计算见 09_meme_softmatch_proto.py）
做法：
  ① 用当前模型在验证集上算 7 类概率（可含 6 视角 TTA）；
  ② 温度缩放：p_T = softmax(logits / T)，网格搜索 T，使 label_hit 最大；
  ③ 情绪相似度矩阵：对 7×7 EMOTION_SIM 的非对角元素做坐标上升搜索（最大化 label_hit）；
  ④ 用 5 折交叉验证估计"真实增益"，避免在验证集上过拟合；
  ⑤ 输出：结果\match_calibration.json/.md + 可选的优化后矩阵/T。

注意：所有参数只在 val 上拟合/评估；测试集不参与。原模型权重完全不动。
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


def label_hit(probs: np.ndarray, y: np.ndarray, sim: np.ndarray, T: float = 1.0) -> float:
    """软匹配的 label_hit：s_j = Σ_i p_i·sim(i,j)，取 max 的 j 与真实情绪比较。

    温度缩放通过 logits/T 再 softmax 实现；温度不改变 p 的 argmax，但会改变软匹配加权后的 argmax。
    """
    if T != 1.0:
        logits = np.log(np.clip(probs, 1e-12, None)) / T
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    scores = probs @ sim
    pred = scores.argmax(axis=1)
    return float((pred == y).mean())


def fit_temperature(probs: np.ndarray, y: np.ndarray, sim: np.ndarray) -> tuple[float, float]:
    """网格搜 T ∈ [0.50, 2.00]，最大化 label_hit；返回 (T*, hit)。"""
    grid = np.round(np.arange(0.50, 2.01, 0.05), 2)
    best_t, best_hit = 1.0, label_hit(probs, y, sim, 1.0)
    for t in grid:
        h = label_hit(probs, y, sim, float(t))
        if h > best_hit + 1e-9:
            best_t, best_hit = float(t), h
    return best_t, best_hit


def fit_sim(probs: np.ndarray, y: np.ndarray, sim0: np.ndarray, T: float = 1.0,
            iters: int = 4, grid: int = 11) -> np.ndarray:
    """坐标上升：逐个优化非对角元素（0~1 网格），最大化 label_hit；对角线固定 1。"""
    sim = sim0.copy()
    values = np.linspace(0.0, 1.0, grid)
    for _ in range(iters):
        improved = False
        for i in range(sim.shape[0]):
            for j in range(sim.shape[1]):
                if i == j:
                    continue
                cur = sim[i, j]
                best_v, best_h = cur, label_hit(probs, y, sim, T)
                for v in values:
                    if abs(v - cur) < 1e-9:
                        continue
                    sim[i, j] = v
                    h = label_hit(probs, y, sim, T)
                    if h > best_h + 1e-9:
                        best_h, best_v = h, v
                sim[i, j] = best_v
                if abs(best_v - cur) > 1e-9:
                    improved = True
        if not improved:
            break
    return sim


def cross_validate(probs: np.ndarray, y: np.ndarray, sim0: np.ndarray,
                   n_splits: int = 5, seed: int = 0) -> dict:
    """5 折 CV：分别评估 基线 / 只调 T / 只调 SIM / 两者联合 的 label_hit。"""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    hits = {"baseline": [], "temperature": [], "sim": [], "both": []}
    params = {"T": [], "sim": []}
    for tr, te in kf.split(probs):
        p_tr, y_tr, p_te, y_te = probs[tr], y[tr], probs[te], y[te]
        hits["baseline"].append(label_hit(p_te, y_te, sim0, 1.0))
        # 只调 T
        t, _ = fit_temperature(p_tr, y_tr, sim0)
        params["T"].append(t)
        hits["temperature"].append(label_hit(p_te, y_te, sim0, t))
        # 只调 SIM
        sim = fit_sim(p_tr, y_tr, sim0, T=1.0)
        params["sim"].append(sim.tolist())
        hits["sim"].append(label_hit(p_te, y_te, sim, 1.0))
        # 联合：先 SIM（T=1），再在拟合后的 SIM 上调 T
        t2, _ = fit_temperature(p_tr, y_tr, sim)
        hits["both"].append(label_hit(p_te, y_te, sim, t2))
    return {"hits": {k: [float(x) for x in v] for k, v in hits.items()},
            "mean": {k: float(np.mean(v)) for k, v in hits.items()},
            "std": {k: float(np.std(v)) for k, v in hits.items()},
            "folds": n_splits, "seed": seed,
            "fold_temperatures": params["T"]}


def main() -> None:
    ap = argparse.ArgumentParser(description="温度缩放 + 情绪相似度矩阵调优（val-only）")
    ap.add_argument("--weights", type=str, default=str(MODEL_DIR / "best_model.pth"))
    ap.add_argument("--tta", action="store_true", help="用 6 视角 TTA 计算概率（推荐，与历史 label_hit 口径一致）")
    ap.add_argument("--out-prefix", type=str, default="match_calibration_v2")
    args = ap.parse_args()

    demo = load_demo()
    CLASS = demo.CLASS_NAMES
    sim0 = demo.EMOTION_SIM.copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device} | TTA={args.tta}")

    with np.load(DATA_DIR / "fer2013" / "fer2013.npz") as z:
        X_val = z["X_val"]
        y_val = z["y_val"].astype(np.int64)
    ckpt = demo.load_checkpoint(pathlib.Path(args.weights), device)
    model = demo.build_model_from_ckpt(ckpt, device)
    size, rgb = demo.model_input_spec(ckpt)
    print(f"权重：{pathlib.Path(args.weights).name} | {ckpt.get('profile')} @ {size}×{size} | "
          f"epoch={ckpt.get('epoch')} | val_acc={ckpt.get('val_acc', 0):.2%}")

    probs_list = []
    with torch.no_grad():
        for i in range(0, len(X_val), 128):
            batch = [demo.preprocess_image(Image.fromarray(x, mode="L"), ckpt["norm_mean"],
                                            ckpt["norm_std"], out_size=size, rgb=rgb)
                     for x in X_val[i:i + 128]]
            xb = torch.cat(batch, dim=0)
            views = demo.tta_views(xb) if args.tta else [xb]
            p = None
            for v in views:
                pp = torch.softmax(model(v.to(device)), dim=1).cpu().numpy()
                p = pp if p is None else p + pp
            probs_list.append(p / len(views))
            if (i // 128) % 5 == 0:
                print(f"  推理进度 {min(i+128, len(X_val))}/{len(X_val)}", flush=True)
    probs = np.concatenate(probs_list)

    base_hit = label_hit(probs, y_val, sim0, 1.0)
    print(f"\n基线 label_hit（TTA={args.tta}）：{base_hit:.4f}")

    # 全量 val 上的"乐观"最优参数
    t_full, hit_t_full = fit_temperature(probs, y_val, sim0)
    sim_full = fit_sim(probs, y_val, sim0, T=1.0)
    hit_sim_full = label_hit(probs, y_val, sim_full, 1.0)
    t_both, hit_both_full = fit_temperature(probs, y_val, sim_full)
    print(f"全量 val 上：T*={t_full:.2f} → hit {hit_t_full:.4f}；SIM 调优 → {hit_sim_full:.4f}；"
          f"联合 T={t_both:.2f} → {hit_both_full:.4f}（这些是乐观估计）")

    # 5 折交叉验证（诚实估计）
    print("\n5 折交叉验证（估计真实增益）…")
    cv = cross_validate(probs, y_val, sim0)
    for k in ("baseline", "temperature", "sim", "both"):
        print(f"  {k:12s} CV label_hit = {cv['mean'][k]:.4f} ± {cv['std'][k]:.4f}")

    result = {
        "meta": {"weights": str(pathlib.Path(args.weights)), "tta": bool(args.tta),
                 "device": str(device), "n_val": len(y_val),
                 "ckpt_epoch": ckpt.get("epoch"), "ckpt_val_acc": ckpt.get("val_acc")},
        "baseline_label_hit": round(base_hit, 4),
        "full_val_optimistic": {"T": t_full, "hit_with_T": round(hit_t_full, 4),
                                "hit_with_sim": round(hit_sim_full, 4),
                                "T_with_sim": t_both, "hit_with_both": round(hit_both_full, 4)},
        "cross_validation": cv,
        "optimized_sim": np.round(sim_full, 3).tolist(),
        "optimized_T": round(t_both, 3),
        "emotion_sim_original": np.round(sim0, 3).tolist(),
        "note": "全量 val 拟合为乐观估计；真实增益看 5 折 CV。参数只用于 val，测试集未参与。",
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / f"{args.out_prefix}.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    # Markdown 报告
    lines = [f"# F 方案：温度缩放 + 情绪相似度矩阵调优（val {len(y_val)} 张，TTA={args.tta}）", "",
             f"- 模型：`{pathlib.Path(args.weights).name}`（{ckpt.get('profile')} @ {size}×{size}，第 {ckpt.get('epoch')} 轮）",
             f"- **基线 label_hit = {base_hit:.4f}**", "",
             "## 5 折交叉验证（诚实估计）", "",
             "| 方案 | label_hit 均值 | 标准差 |", "|---|---|---|"]
    for k, label in (("baseline", "基线（原 T=1, 原相似度矩阵）"),
                     ("temperature", "只做温度缩放"),
                     ("sim", "只调相似度矩阵"),
                     ("both", "温度 + 矩阵联合")):
        lines.append(f"| {label} | {cv['mean'][k]:.4f} | {cv['std'][k]:.4f} |")
    lines += ["", "## 全量 val 上拟合的参数（乐观估计）", "",
              f"- 温度 T* = {t_full:.2f}（label_hit {hit_t_full:.4f}）",
              f"- 相似度矩阵调优后 label_hit = {hit_sim_full:.4f}",
              f"- 联合（先矩阵后温度）T = {t_both:.2f}，label_hit = {hit_both_full:.4f}", "",
              "## 调优后的情绪相似度矩阵（行/列顺序：angry, disgust, fear, happy, sad, surprise, neutral）", "",
              "```"]
    for i, row in enumerate(np.round(sim_full, 2)):
        lines.append(f"{CLASS[i]:8s} " + " ".join(f"{v:.2f}" for v in row))
    lines += ["```", "", "> 注意：全量 val 拟合的矩阵/温度是乐观估计；若上线需用 CV 参数（各折 T 见 json）。",
              "> 测试集不参与；原模型权重未改动。"]
    (RESULT_DIR / f"{args.out_prefix}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告：{RESULT_DIR / (args.out_prefix + '.md')}")
    print(f"JSON：{RESULT_DIR / (args.out_prefix + '.json')}")


if __name__ == "__main__":
    main()
