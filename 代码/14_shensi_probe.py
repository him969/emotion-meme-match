# -*- coding: utf-8 -*-
r"""
14_shensi_probe.py —— "神似模式"B2 离线验证 + 素材生成（只用验证集，不训练、不碰测试集）

做什么
------
1. 用 mediapipe 对 FER 验证集 3589 张算人脸关键点几何，产出验证集分位统计
   （模型\shensi_geometry_stats.json，供 13_shensi_visual.py 的百分位映射使用）；
2. 逐类统计几何量（嘴开合、眼睁开、嘴角上扬、眉眼距）→ 证明"关键点确实带神态信息"；
3. 用 13 模块在"已选情绪类内"比较 4 种视觉信号：
     random / attr-only / geo-only / fused（默认权重 0.5/0.35/0.15）：
   输出多样性、典型例子与对照拼图（供目检打分）；
4. 生成 `结果\proto_shensi_*`：picks CSV、summary JSON、landmark 拼图、A/B 对照拼图。

运行
----
  $env:PYTHONIOENCODING='utf-8'
  python 代码\14_shensi_probe.py
"""
import json
import os
import pathlib
import random
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "代码"))

import importlib.util

spec = importlib.util.spec_from_file_location("shensi_visual", ROOT / "代码" / "13_shensi_visual.py")
sv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sv)

RESULT_DIR = ROOT / "结果"
FER_NPZ = ROOT / "数据" / "fer2013" / "fer2013.npz"
PROBS_NPZ = RESULT_DIR / "proto_face_probs_val.npz"
GEOM_STATS_JSON = ROOT / "模型" / "shensi_geometry_stats.json"
FACE_CACHE_NPZ = RESULT_DIR / "proto_shensi_face_cache.npz"
PICKS_CSV = RESULT_DIR / "proto_shensi_picks_val.csv"
SUMMARY_JSON = RESULT_DIR / "proto_shensi_summary.json"
LANDMARK_PNG = RESULT_DIR / "proto_shensi_landmarks.png"
CLASS_NAMES = sv.CLASS_NAMES
RNG = np.random.default_rng(2026)
PyRandom = random.Random(2026)

GEOM_FEATURES = sv.GEOM_FEATURES

# 7×7 情绪相似度（与 07_demo_app.py 的 EMOTION_SIM 完全一致）
EMOTION_SIM = np.array([
    [1.00, 0.80, 0.50, 0.05, 0.50, 0.20, 0.25],
    [0.80, 1.00, 0.30, 0.02, 0.45, 0.10, 0.20],
    [0.50, 0.30, 1.00, 0.05, 0.40, 0.80, 0.20],
    [0.05, 0.02, 0.05, 1.00, 0.05, 0.60, 0.20],
    [0.50, 0.45, 0.40, 0.05, 1.00, 0.10, 0.60],
    [0.20, 0.10, 0.80, 0.60, 0.10, 1.00, 0.30],
    [0.25, 0.20, 0.20, 0.20, 0.60, 0.30, 1.00],
], dtype=np.float64)


def load_font(size):
    for name in ["arial.ttf", "segoeui.ttf", "msyh.ttc"]:
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


F = load_font(13)
FB = load_font(16)


def load_fer_val():
    with np.load(FER_NPZ) as z:
        return z["X_val"], z["y_val"]


def load_probs():
    with np.load(PROBS_NPZ) as z:
        return z["p_tta"].astype(np.float64), z["y_val"].astype(np.int64)


# ---------------- 1. 验证集几何 ----------------

def compute_geometry(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """对每张 FER 人脸算几何量；返回 (features [N,4], ok_mask)。"""
    lm = sv.get_landmarker()
    if lm is None:
        raise SystemExit("没有可用的人脸关键点模型（mediapipe/onnx 都失败）")
    feats = np.full((len(X), len(GEOM_FEATURES)), np.nan, dtype=np.float32)
    ok = np.zeros(len(X), dtype=bool)
    t0 = time.time()
    for i, x48 in enumerate(X):
        img = Image.fromarray(x48, mode="L").convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
        try:
            geo = sv.face_geometry(img)
        except Exception:  # noqa: BLE001
            geo = None
        if geo is not None:
            for j, name in enumerate(GEOM_FEATURES):
                feats[i, j] = geo[name]
            ok[i] = True
        if (i + 1) % 500 == 0:
            print(f"  几何 {i + 1}/{len(X)}，已检出 {ok.sum()}，用时 {time.time() - t0:.0f}s", flush=True)
    print(f"  几何完成：{ok.sum()}/{len(X)} 张检出人脸，用时 {time.time() - t0:.0f}s")
    return feats, ok


def geom_stats(feats: np.ndarray, ok: np.ndarray) -> dict:
    """每特征取 101 个分位点，供 demo 的百分位映射。"""
    stats = {}
    for j, name in enumerate(GEOM_FEATURES):
        v = feats[ok, j]
        v = v[np.isfinite(v)]
        qs = np.linspace(0, 100, 101)
        stats[name] = {"percentiles": qs.tolist(),
                       "values": np.percentile(v, qs).tolist(),
                       "mean": float(np.mean(v)), "std": float(np.std(v))}
    return stats


# ---------------- 2. 人脸视觉证据（几何 + CLIP 属性画像） ----------------

def compute_face_evidence(X: np.ndarray, feats: np.ndarray, ok: np.ndarray,
                          geo_stats: dict) -> dict:
    """算每张验证集人脸的属性画像与 CLIP 特征（缓存 attr/clip；几何每次用本次新算的）。

    注意：几何必须用**本次运行的关键点后端**重算，不能复用缓存——mediapipe 与 ONNX 备份
    在输入裁切上不同，几何数值会系统性偏移（实测同图嘴开合 0.01 vs 0.36）。
    """
    if FACE_CACHE_NPZ.exists():
        with np.load(FACE_CACHE_NPZ) as z:
            if z["attr"].shape[0] == len(X):
                print(f"  [缓存] 人脸属性画像 {FACE_CACHE_NPZ.name}（几何用本次重算）")
                return {"attr": z["attr"].astype(np.float64), "clip": z["clip"].astype(np.float64),
                        "geom": feats, "ok": ok}
    imgs = [Image.fromarray(x, mode="L").convert("RGB") for x in X]
    clip_feat = sv.encode_images_clip(imgs)
    attr = sv.attr_profiles(clip_feat, face_domain=True)
    np.savez_compressed(FACE_CACHE_NPZ, attr=attr.astype(np.float32), clip=clip_feat.astype(np.float32),
                        geom=feats, ok=ok)
    print(f"  已缓存人脸属性画像：{FACE_CACHE_NPZ.relative_to(ROOT)}")
    return {"attr": attr, "clip": clip_feat, "geom": feats, "ok": ok}


# ---------------- 3. 类内选择（4 种信号） ----------------

def pick_with_assets(p: np.ndarray, evidence: dict, assets, variant: str,
                     weights=(0.50, 0.35, 0.15)) -> tuple[np.ndarray, np.ndarray]:
    """对每张人脸：先按 FER 概率选类，再在类内按 variant 挑一张。

    返回 (chosen 的 meme 位置下标, chosen 的综合分)。variant ∈ random/attr/geo/clip/fused。
    """
    pred = p.argmax(1)
    meme_emotion = assets.memes["emotion"].map({e: i for i, e in enumerate(CLASS_NAMES)}).to_numpy()
    n_meme = len(meme_emotion)
    chosen = np.empty(len(p), dtype=int)
    score = np.empty(len(p), dtype=float)
    for i in range(len(p)):
        cand = np.flatnonzero(meme_emotion == pred[i])
        if variant == "random":
            k = PyRandom.randrange(len(cand))
            chosen[i] = cand[k]
            score[i] = 0.5
            continue
        e = {"attr_prof": evidence["attr"][i] if evidence.get("attr") is not None else None,
             "geometry": (None if not evidence["ok"][i] else
                          {name: float(evidence["geom"][i, j]) for j, name in enumerate(GEOM_FEATURES)}),
             "clip_feat": evidence["clip"][i] if evidence.get("clip") is not None else None}
        ranked = sv.rank_candidates(e, assets.memes, list(cand), weights=weights)
        if variant == "fused":
            best = ranked[0]
        elif variant == "attr":
            best = max(ranked, key=lambda r: r["attr_sim"])
        elif variant == "geo":
            best = max(ranked, key=lambda r: r["geo_sim"])
        elif variant == "clip":
            best = max(ranked, key=lambda r: r["clip_sim"])
        else:
            raise ValueError(variant)
        chosen[i] = best["position"]
        score[i] = best["score"]
    return chosen, score


def diversity_stats(chosen: np.ndarray, meme_emotion: np.ndarray) -> dict:
    files = chosen
    counts = np.bincount(files, minlength=len(meme_emotion))
    return {
        "unique_memes": int(len(np.unique(files))),
        "total_memes": int(len(meme_emotion)),
        "max_single_share": float(counts.max() / len(files)),
        "consecutive_repeats": int(np.sum(files[1:] == files[:-1])),
    }


# ---------------- 4. 对照拼图（目检用） ----------------

def make_ab_sheet(X, y, p, assets, picks_map: dict[str, np.ndarray], rows: list[int],
                  out_path: pathlib.Path, cell: int = 165) -> None:
    """每行：人脸 | random | attr-only | fused；标签写文件名 + 分数。"""
    cols = ["face", "random", "attr", "fused"]
    label_h = 40
    width = len(cols) * (cell + 10) + 10
    height = 40 + len(rows) * (cell + label_h + 8)
    canvas = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(canvas)
    d.text((8, 8), "神似模式验收：人脸 | 基线随机 | 纯属性桥 | 融合(属性+关键点+CLIP)", fill="black", font=FB)
    memes = assets.memes
    for r, i in enumerate(rows):
        yy = 40 + r * (cell + label_h + 8)
        face = Image.fromarray(X[i], mode="L").convert("RGB").resize((cell, cell), Image.Resampling.LANCZOS)
        canvas.paste(face, (8, yy))
        d.text((8, yy + cell + 2), f"#{i} 真={CLASS_NAMES[int(y[i])]} 预测={CLASS_NAMES[int(p.argmax(1)[i])]}",
               fill="black", font=F)
        for c, variant in enumerate(cols[1:], start=1):
            x = 8 + c * (cell + 10)
            pos = int(picks_map[variant][i])
            path = str(ROOT / pathlib.Path(memes.iloc[pos]["processed"]))
            with Image.open(path) as im:
                m = im.convert("RGB")
                m.thumbnail((cell, cell))
                canvas.paste(m, (x + (cell - m.width) // 2, yy + (cell - m.height) // 2))
            d.text((x, yy + cell + 2), f"{variant}:{str(memes.iloc[pos]['file'])[:24]}", fill="black", font=F)
            d.text((x, yy + cell + 18), f"情绪={memes.iloc[pos]['emotion']}",
                   fill="gray", font=F)
    canvas.save(out_path)
    print(f"写出 {out_path.relative_to(ROOT)}  {canvas.size}")


def make_landmark_sheet(X, y, out_path: pathlib.Path, n: int = 12) -> None:
    """关键点可视化拼图（证明几何提取正确落在眼/鼻/唇上）。"""
    lm = sv.get_landmarker()
    idx = np.sort(RNG.choice(len(X), n, replace=False))
    cell = 180
    cols = 4
    rows = (n + cols - 1) // cols
    canvas = Image.new("RGB", (cols * (cell + 10) + 10, 40 + rows * (cell + 30)), "white")
    d = ImageDraw.Draw(canvas)
    d.text((8, 8), "神似模式：人脸 478 关键点提取（验证集样例）", fill="black", font=FB)
    for k, i in enumerate(idx):
        r, c = divmod(k, cols)
        x0, y0 = 10 + c * (cell + 10), 40 + r * (cell + 30)
        img = Image.fromarray(X[i], mode="L").convert("RGB").resize((cell, cell), Image.Resampling.LANCZOS)
        pts = lm.detect(img)
        canvas.paste(img, (x0, y0))
        if pts is not None:
            for idxs, color in [(sv.I_LEFT_EYE, (0, 255, 0)), (sv.I_RIGHT_EYE, (0, 255, 0)),
                                (sv.I_MOUTH, (255, 40, 40)), (sv.I_BROW, (0, 200, 255))]:
                xy = [(x0 + float(pts[j, 0]), y0 + float(pts[j, 1])) for j in idxs]
                d.line(xy, fill=color, width=2, joint="curve")
        geo = sv.face_geometry(img)
        txt = f"#{i} {CLASS_NAMES[int(y[i])]}"
        if geo:
            txt += f" 嘴{geo['mouth_aspect']:.2f} 眼{geo['eye_open']:.2f}"
        d.text((x0, y0 + cell + 2), txt, fill="black", font=F)
    canvas.save(out_path)
    print(f"写出 {out_path.relative_to(ROOT)}  {canvas.size}")


# ---------------- 5. 主流程 ----------------

def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    X, y = load_fer_val()
    p, y_p = load_probs()
    assert np.array_equal(y, y_p)
    print(f"验证集 {len(X)} 张")

    print("\n[1/5] 人脸关键点几何 ...")
    feats, ok = compute_geometry(X)
    stats = geom_stats(feats, ok)
    GEOM_STATS_JSON.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  几何分位统计已存：{GEOM_STATS_JSON.relative_to(ROOT)}")

    print("\n[2/5] 逐类几何均值（证明关键点带神态信息）...")
    per_class = {}
    for c, name in enumerate(CLASS_NAMES):
        m = ok & (y == c)
        per_class[name] = {f: round(float(np.nanmean(feats[m, j])), 4) for j, f in enumerate(GEOM_FEATURES)}
        print(f"  {name:9s}" + "  ".join(f"{f}={per_class[name][f]:.3f}" for f in GEOM_FEATURES))

    print("\n[3/5] 人脸视觉证据（CLIP 属性画像）...")
    evidence = compute_face_evidence(X, feats, ok, stats)

    print("\n[4/5] 类内选择（random / attr / geo / clip / fused）...")
    assets = sv.get_assets()
    t0 = time.time()
    picks_map, scores = {}, {}
    for variant in ["random", "attr", "geo", "clip", "fused"]:
        picks_map[variant], scores[variant] = pick_with_assets(p, evidence, assets, variant)
        div = diversity_stats(picks_map[variant], assets.memes["emotion"].map(
            {e: i for i, e in enumerate(CLASS_NAMES)}).to_numpy())
        print(f"  {variant:7s} 用时 {time.time() - t0:.1f}s  多样性：{div}")
        t0 = time.time()
    meme_emotion = assets.memes["emotion"].map({e: i for i, e in enumerate(CLASS_NAMES)}).to_numpy()

    # ---------------- β 加权扫描：情感分 + 视觉分 ----------------
    print("\n[4b/5] β 加权扫描（F = (1-β)·情感分 + β·视觉分，看情绪命中率/多样性/视觉分）...")
    meme_emotion_idx = assets.memes["emotion"].map({e: i for i, e in enumerate(CLASS_NAMES)}).to_numpy()
    s_emo = p @ EMOTION_SIM[:, meme_emotion_idx]              # (N,58) 软匹配分
    def _minmax_rows(a):
        lo = a.min(axis=1, keepdims=True)
        hi = a.max(axis=1, keepdims=True)
        return np.where(hi - lo < 1e-9, 0.5, (a - lo) / (hi - lo + 1e-9))
    s_emo_n = _minmax_rows(s_emo)
    # 每张人脸的视觉分量（58 维）
    vis_rows = np.empty((len(p), len(meme_emotion_idx)), dtype=np.float32)
    lat_ms = []
    for i in range(len(p)):
        e = {"attr_prof": evidence["attr"][i],
             "geometry": (None if not evidence["ok"][i] else
                          {name: float(evidence["geom"][i, j]) for j, name in enumerate(GEOM_FEATURES)}),
             "clip_feat": evidence["clip"][i]}
        t0 = time.time()
        vis_rows[i] = sv.visual_components(e, assets.memes)["visual"]
        lat_ms.append((time.time() - t0) * 1000)
    print(f"  视觉分量计算：均值 {np.mean(lat_ms):.2f}ms/张，P95 {np.percentile(lat_ms, 95):.2f}ms")
    beta_rows = []
    for beta in [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6]:
        fused = (1.0 - beta) * s_emo_n + beta * vis_rows
        chosen = fused.argmax(axis=1)
        hit = float((meme_emotion_idx[chosen] == y).mean())
        div = diversity_stats(chosen, meme_emotion_idx)
        mean_vis = float(vis_rows[np.arange(len(p)), chosen].mean())
        beta_rows.append({"beta": beta, "emotion_hit": hit, "mean_visual": mean_vis, **div})
        print(f"  β={beta:.2f}  情绪命中={hit:.4f}  视觉分={mean_vis:.4f}  "
              f"用到 {div['unique_memes']}/58 张，最大占比 {div['max_single_share']:.1%}，连续重复 {div['consecutive_repeats']}")
    beta_arr = np.array([r["emotion_hit"] for r in beta_rows])
    print(f"  对照：模型 argmax 情绪准确率 = {(p.argmax(1) == y).mean():.4f}；"
          f"历史软匹配 label_hit = 0.6773（β=0 行）")

    # 保存 picks
    df = pd.DataFrame({"val_idx": np.arange(len(p)),
                       "y_true": [CLASS_NAMES[v] for v in y],
                       "pred": [CLASS_NAMES[v] for v in p.argmax(1)],
                       "p_pred": p.max(1)})
    for variant, arr in picks_map.items():
        df[f"{variant}_file"] = assets.memes.iloc[arr]["file"].to_numpy()
        df[f"{variant}_emotion"] = assets.memes.iloc[arr]["emotion"].to_numpy()
    df["beta035_file"] = assets.memes.iloc[((1 - 0.35) * s_emo_n + 0.35 * vis_rows).argmax(1)]["file"].to_numpy()
    df.to_csv(PICKS_CSV, index=False, encoding="utf-8-sig")
    print(f"  明细：{PICKS_CSV.relative_to(ROOT)}")

    print("\n[5/5] 目检素材 ...")
    # 采样：每类取 2 个高置信样本
    rows = []
    for c in range(7):
        idx = np.flatnonzero((p.argmax(1) == c) & (y == c))
        idx = idx[np.argsort(-p[idx, c])][:2]
        rows.extend(idx.tolist())
    make_ab_sheet(X, y, p, assets, picks_map, rows[:10], RESULT_DIR / "proto_shensi_ab_1.png")
    make_ab_sheet(X, y, p, assets, picks_map, rows[10:], RESULT_DIR / "proto_shensi_ab_2.png")
    make_landmark_sheet(X, y, LANDMARK_PNG)

    summary = {"n_val": int(len(X)), "geometry_ok": int(ok.sum()),
               "geometry_ok_rate": float(ok.mean()),
               "per_class_geometry": per_class,
               "diversity": {v: diversity_stats(picks_map[v], meme_emotion) for v in picks_map},
               "beta_sweep": beta_rows,
               "baseline_soft_match_hit": float((meme_emotion_idx[s_emo.argmax(1)] == y).mean()),
               "weights": {"attr": sv.DEFAULT_WEIGHTS[0], "geo": sv.DEFAULT_WEIGHTS[1],
                           "clip": sv.DEFAULT_WEIGHTS[2]},
               "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n汇总：{SUMMARY_JSON.relative_to(ROOT)}")
    print("完成")


if __name__ == "__main__":
    main()
