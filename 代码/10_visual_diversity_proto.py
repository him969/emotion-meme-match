# -*- coding: utf-8 -*-
r"""
10_visual_diversity_proto.py —— 建议 3 的离线验证：1 主推 + 2 视觉备选（不改模型、不改演示）

背景：
  软匹配"同分随机"修复后，"重复问题"已解决，但同一种情绪下展示的图仍可能"风格很像"。
  这个原型验证：用 meme↔meme 视觉相似度挑 2 张"视觉上尽量不同"的备选，效果如何。

本脚本只读数据与权重，产出对比结果，**不改 07_demo_app.py、不改任何模型**。

评估矩阵（3 种备选范围 × 2 种视觉特征 = 6 种组合）：
  备选范围：
    ① same     仅同情绪类（默认）
    ② majority 同类为主（候选 = 同类 + 相似情绪；至少 1 张同类）
    ③ free     自由补充（候选 = 同类 + 相似情绪；无同类数量要求）
  视觉特征：
    ① imgnet   ImageNet 原始 ResNet18 特征（看"外观像不像"，默认）
    ② fer      FER 微调 ResNet18 特征（看"表情特征像不像"）

主推逻辑保持不变：软匹配选出情绪类别 → 同类随机一张（与 07 修复后的行为一致）。
备选选择：贪心 max-min —— 备选1 离主推最远；备选2 到（主推、备选1）的最小距离最大。
"换一批"：枚举主推固定下的合法备选对，保留"多样性分 ≥ 最优 90%"的对随机抽，
          并记住最近几批不重复。

评估（验证集 3589 张，复用 09 缓存的 proto_face_probs_val.npz）：
  · 主推不变性：每种组合的主推情绪必须与现状完全一致（预期 0 差异）
  · 备选标签安全性：备选与主推同情绪比例、备选情绪的 S 加权相似度
  · 视觉多样性：三张图两两余弦距离均值、近重复对数量（距离 < 0.05 即相似度 > 0.95）
  · 换一批：模拟抽 6 批（1 初始 + 5 次换一批），统计不同组合数 / 是否变化
  · 与"随机备选"基线对比（同样范围约束下随机挑 2 张）

产出（结果\ 下）：
  proto_visual_diversity_report.md / .csv    6 组合指标表 + 结论
  proto_visual_examples_imgnet.png           样例拼图（ImageNet 特征，同类为主模式）
  proto_visual_examples_fer.png              样例拼图（FER 特征，同类为主模式）

运行（PowerShell）：
    $env:PYTHONIOENCODING='utf-8'
    python <项目根>/代码/10_visual_diversity_proto.py
"""
import os
import pathlib
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, pathlib.Path(r"<项目根>/_deps").as_posix())

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import importlib.util
import itertools
import random
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "代码"
RESULT_DIR = ROOT / "结果"
DEPS_DIR = ROOT / "_deps"
MODEL_DIR = ROOT / "模型"
if str(DEPS_DIR) not in sys.path:
    sys.path.insert(0, str(DEPS_DIR))

import timm
from safetensors.torch import load_file

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
WEIGHTS = MODEL_DIR / "best_model.pth"

# 与 07_demo_app.py 保持一致的情绪相似度矩阵（行/列顺序 = CLASS_NAMES）
EMOTION_SIM = np.array([
    [1.00, 0.80, 0.50, 0.05, 0.50, 0.20, 0.25],
    [0.80, 1.00, 0.30, 0.02, 0.45, 0.10, 0.20],
    [0.50, 0.30, 1.00, 0.05, 0.40, 0.80, 0.20],
    [0.05, 0.02, 0.05, 1.00, 0.05, 0.60, 0.20],
    [0.50, 0.45, 0.40, 0.05, 1.00, 0.10, 0.60],
    [0.20, 0.10, 0.80, 0.60, 0.10, 1.00, 0.30],
    [0.25, 0.20, 0.20, 0.20, 0.60, 0.30, 1.00],
], dtype=np.float64)


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, CODE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ds = load_module("fer_dataset", "03_dataset.py")
demo = load_module("demo_app", "07_demo_app.py")


# ---------------- 1. meme 视觉特征 ----------------

@torch.no_grad()
def extract_features(feature_kind: str, meme_tensors: torch.Tensor, device) -> np.ndarray:
    """提取 58 张 meme 的 512 维全局特征（L2 归一化后返回），两种特征可切换。"""
    if feature_kind == "fer":
        ckpt = demo.load_checkpoint(WEIGHTS, device)
        model = demo.build_model_from_ckpt(ckpt, device)
        feats = model.forward_features(meme_tensors.to(device))
        feats = model.global_pool(feats)
        feats = torch.flatten(feats, 1)
    elif feature_kind == "imgnet":
        model = timm.create_model("resnet18", pretrained=False, num_classes=0)
        sd = load_file(str(DEPS_DIR / "weights" / "resnet18.a1_in1k.safetensors"))
        state = {k: v for k, v in sd.items() if not k.startswith("fc.")}
        model.load_state_dict(state, strict=False)
        model = model.to(device).eval()
        feats = model.forward_features(meme_tensors.to(device))
        feats = model.global_pool(feats)
        feats = torch.flatten(feats, 1)
    else:
        raise ValueError(feature_kind)
    feats = feats.float().cpu().numpy()
    feats = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    return feats


def build_meme_tensor(memes: pd.DataFrame) -> torch.Tensor:
    """把 58 张处理后的 meme 图变成 (58,3,128,128) 的 ImageNet 归一化张量。"""
    tensors = []
    for p in memes["path"]:
        with Image.open(p) as im:
            tensors.append(demo.preprocess_image(im, IMAGENET_MEAN, IMAGENET_STD, out_size=128, rgb=True))
    return torch.cat(tensors, dim=0)


# ---------------- 2. 候选池与备选选择 ----------------

def similar_emotions(main_emotion: int, k: int = 2) -> list[int]:
    """按相似度矩阵找出与主情绪最相似的 k 个其他情绪（下标）。"""
    row = EMOTION_SIM[main_emotion].copy()
    row[main_emotion] = -1
    return [int(i) for i in np.argsort(-row)[:k]]


def candidate_pool(memes: pd.DataFrame, label_idx: np.ndarray, main_idx: int,
                   main_emotion: int, mode: str) -> np.ndarray:
    """按备选范围模式返回候选 meme 下标（不含主推）。

    mode = same      → 只取同情绪类
           majority  → 同类 + 相似情绪
           free      → 同类 + 相似情绪（与 majority 的差别在约束，不在候选池）
    """
    same = np.flatnonzero((label_idx == main_emotion) & (np.arange(len(memes)) != main_idx))
    if mode == "same":
        return same
    adj = similar_emotions(main_emotion, k=2)
    extra = np.flatnonzero(np.isin(label_idx, adj) & (np.arange(len(memes)) != main_idx))
    return np.concatenate([same, extra])


def greedy_pair(main_idx: int, pool: np.ndarray, feat: np.ndarray,
                label_idx: np.ndarray, main_emotion: int, mode: str,
                rng: random.Random | None = None) -> tuple[int, int]:
    """贪心 max-min 选两张备选；majority 模式保证至少 1 张同类。

    feat 是 L2 归一化后的特征，距离 = 1 - cos。
    """
    if len(pool) < 1:
        return -1, -1
    d_main = 1.0 - feat[pool] @ feat[main_idx]
    if len(pool) == 1:
        alt1 = int(pool[0])
        return alt1, -1
    # 备选 1：离主推最远
    alt1_pos = int(np.argmax(d_main))
    alt1 = int(pool[alt1_pos])
    rest = pool[pool != alt1]
    d1 = 1.0 - feat[rest] @ feat[alt1]
    score2 = np.minimum(d_main[pool != alt1], d1)
    alt2 = int(rest[int(np.argmax(score2))])
    # majority 模式：两张都不同类时，把第二张换成"最优同类候选"
    if mode == "majority":
        if label_idx[alt1] != main_emotion and label_idx[alt2] != main_emotion:
            same = np.flatnonzero((label_idx == main_emotion) & (np.arange(len(label_idx)) != main_idx))
            same = same[same != alt1]
            if len(same) > 0:
                d_main_same = 1.0 - feat[same] @ feat[main_idx]
                d_same = 1.0 - feat[same] @ feat[alt1]
                alt2 = int(same[int(np.argmax(np.minimum(d_main_same, d_same)))])
    return alt1, alt2


def pair_diversity_score(main_idx: int, a: int, b: int, feat: np.ndarray) -> float:
    """一对备选的多样性分：三张图两两余弦距离的最小值（越大越"不重样"）。"""
    dab = float(1.0 - feat[a] @ feat[b])
    dma = float(1.0 - feat[main_idx] @ feat[a])
    dmb = float(1.0 - feat[main_idx] @ feat[b])
    return min(dma, dmb, dab)


def legal_pair_count(pool: np.ndarray, label_idx: np.ndarray, main_emotion: int, mode: str) -> int:
    """可枚举的合法备选对数量（majority 要求至少一张同类）。"""
    n = 0
    for a, b in itertools.combinations(pool, 2):
        if mode == "majority" and label_idx[a] != main_emotion and label_idx[b] != main_emotion:
            continue
        n += 1
    return n


def reroll_pairs(main_idx: int, pool: np.ndarray, feat: np.ndarray, label_idx: np.ndarray,
                 main_emotion: int, mode: str, rng: random.Random,
                 recent: list[frozenset], keep_ratio: float = 0.9) -> tuple[int, int]:
    """换一批：枚举合法备选对，保留多样性分 >= 最优 90% 的对，随机抽一个没出现过的。"""
    pairs = []
    for a, b in itertools.combinations(pool, 2):
        if mode == "majority" and label_idx[a] != main_emotion and label_idx[b] != main_emotion:
            continue
        pairs.append((a, b, pair_diversity_score(main_idx, a, b, feat)))
    if not pairs:
        a, b = greedy_pair(main_idx, pool, feat, label_idx, main_emotion, mode, rng)
        return a, b
    best = max(p[2] for p in pairs)
    pool_top = [(a, b) for a, b, s in pairs if s >= keep_ratio * best]
    fresh = [(a, b) for a, b in pool_top if frozenset((a, b)) not in recent]
    cand = fresh if fresh else pool_top
    return cand[rng.randrange(len(cand))]


# ---------------- 3. 主推逻辑（与 07 修复后行为一致） ----------------

def soft_scores(p: np.ndarray, label_idx: np.ndarray) -> np.ndarray:
    return EMOTION_SIM[:, label_idx].T @ p


def pick_main(p: np.ndarray, label_idx: np.ndarray, rng: random.Random) -> tuple[int, int]:
    """返回 (主推 meme 下标, 主推情绪下标)。选中情绪 = 旧逻辑（软匹配分最高的情绪类），类内随机。"""
    scores = soft_scores(p, label_idx)
    # 旧逻辑：sort_values 取第一张 → 该情绪类
    order = np.argsort(-scores, kind="stable")
    winner_emotion = int(label_idx[order[0]])
    cand = np.flatnonzero(label_idx == winner_emotion)
    main_idx = int(cand[rng.randrange(len(cand))])
    return main_idx, winner_emotion


# ---------------- 4. 评估主流程 ----------------

def eval_combo(memes, label_idx, feat, probs, y_val, mode: str, feature_kind: str,
               main_rng_seed: int = 42, n_reroll_queries: int = 200) -> dict:
    """对一种"范围 × 特征"组合跑完整评估。"""
    n = len(probs)
    metrics = {
        "mode": mode, "feature": feature_kind,
        "main_emotion_mismatch": 0,
        "same_emotion_alt_ratio": [], "alt_s_weighted": [],
        "pairwise_dist": [], "near_dup_pairs": 0,
    }
    # 主推随机（每张查询独立种子，保证可复现）
    main_picks = []
    for i in range(n):
        rng = random.Random(main_rng_seed * 100003 + i)
        main_idx, winner_emotion = pick_main(probs[i], label_idx, rng)
        # 与旧逻辑比较选中情绪（应完全一致）
        old_emotion = int(label_idx[int(np.argsort(-soft_scores(probs[i], label_idx), kind="stable")[0])])
        metrics["main_emotion_mismatch"] += int(winner_emotion != old_emotion)
        main_picks.append((main_idx, winner_emotion))

        pool = candidate_pool(memes, label_idx, main_idx, winner_emotion, mode)
        alt1, alt2 = greedy_pair(main_idx, pool, feat, label_idx, winner_emotion, mode,
                                 random.Random(i))
        if alt1 < 0:
            continue
        alts = [alt1] + ([alt2] if alt2 >= 0 else [])
        metrics["same_emotion_alt_ratio"].append(
            np.mean([label_idx[a] == winner_emotion for a in alts]))
        metrics["alt_s_weighted"].append(
            np.mean([EMOTION_SIM[winner_emotion, label_idx[a]] for a in alts]))
        dma = 1.0 - feat[main_idx] @ feat[alt1]
        dmb = 1.0 - feat[main_idx] @ feat[alt2]
        dab = 1.0 - feat[alt1] @ feat[alt2]
        metrics["pairwise_dist"].append(np.mean([dma, dmb, dab]))
        metrics["near_dup_pairs"] += int(dma < 0.05) + int(dmb < 0.05) + int(dab < 0.05)

    # 换一批统计（抽样查询，减少耗时）
    reroll_sets_total, reroll_distinct, reroll_changed = [], [], []
    sample_idx = np.random.default_rng(0).choice(n, size=min(n_reroll_queries, n), replace=False)
    for i in sample_idx:
        main_idx, winner_emotion = main_picks[int(i)]
        pool = candidate_pool(memes, label_idx, main_idx, winner_emotion, mode)
        rng = random.Random(7)
        a, b = greedy_pair(main_idx, pool, feat, label_idx, winner_emotion, mode, rng)
        recent = [frozenset((a, b))]
        combos = {frozenset((a, b))}
        changed = 0
        for _ in range(5):
            a2, b2 = reroll_pairs(main_idx, pool, feat, label_idx, winner_emotion, mode,
                                  random.Random(rng.randrange(10**9)), recent)
            if frozenset((a2, b2)) != frozenset((a, b)):
                changed += 1
            recent.append(frozenset((a2, b2)))
            combos.add(frozenset((a2, b2)))
        reroll_sets_total.append(len(combos))
        reroll_changed.append(changed)

    # 附：同范围约束下"随机选两张"的基线（同样抽样 200 个查询）
    rand_dist = []
    for i in sample_idx:
        main_idx, winner_emotion = main_picks[int(i)]
        pool = candidate_pool(memes, label_idx, main_idx, winner_emotion, mode)
        rng = random.Random(int(i))
        idxs = list(pool)
        rng.shuffle(idxs)
        picked = []
        for a in idxs:
            if mode == "majority" and len(picked) == 0 and label_idx[a] != winner_emotion:
                continue
            if mode == "majority" and len(picked) == 1 and label_idx[picked[0]] != winner_emotion and label_idx[a] != winner_emotion:
                continue
            picked.append(a)
            if len(picked) == 2:
                break
        if len(picked) == 2:
            rand_dist.append(np.mean([1 - feat[main_idx] @ feat[picked[0]],
                                      1 - feat[main_idx] @ feat[picked[1]],
                                      1 - feat[picked[0]] @ feat[picked[1]]]))

    out = {
        "mode": mode, "feature": feature_kind,
        "main_emotion_mismatch": metrics["main_emotion_mismatch"],
        "same_emotion_alt_ratio": float(np.mean(metrics["same_emotion_alt_ratio"])),
        "alt_s_weighted": float(np.mean(metrics["alt_s_weighted"])),
        "pairwise_dist_mean": float(np.mean(metrics["pairwise_dist"])),
        "near_dup_pairs": int(metrics["near_dup_pairs"]),
        "random_pairwise_dist_mean": float(np.mean(rand_dist)) if rand_dist else float("nan"),
        "reroll_mean_distinct_sets": float(np.mean(reroll_sets_total)),
        "reroll_changed_ratio": float(np.sum(reroll_changed) / (5 * len(reroll_changed))),
    }
    return out


def make_example_figure(memes, feat, probs, label_idx, mode: str, feature_kind: str, path: pathlib.Path,
                        n_examples: int = 4) -> None:
    """把几个查询的"主推 + 2 备选"拼成图，人工目检视觉多样性。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for _font in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if any(_font.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [_font]
            break
    plt.rcParams["axes.unicode_minus"] = False

    rng = np.random.default_rng(3)
    idxs = rng.choice(len(probs), size=n_examples, replace=False)
    fig, axes = plt.subplots(n_examples, 3, figsize=(9, n_examples * 3.1))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(idxs):
        main_idx, winner_emotion = pick_main(probs[int(i)], label_idx, random.Random(int(i)))
        pool = candidate_pool(memes, label_idx, main_idx, winner_emotion, mode)
        alt1, alt2 = greedy_pair(main_idx, pool, feat, label_idx, winner_emotion, mode,
                                 random.Random(int(i)))
        picks = [main_idx, alt1, alt2]
        titles = [
            f"主推 {CLASS_NAMES[label_idx[main_idx]]}",
            f"备选1 {CLASS_NAMES[label_idx[alt1]]}\n差异 {1 - feat[main_idx] @ feat[alt1]:.2f}",
            f"备选2 {CLASS_NAMES[label_idx[alt2]]}\n差异 {1 - feat[main_idx] @ feat[alt2]:.2f}·{1 - feat[alt1] @ feat[alt2]:.2f}",
        ]
        for c, (m_idx, title) in enumerate(zip(picks, titles)):
            ax = axes[r, c]
            with Image.open(memes["path"].iloc[m_idx]) as im:
                ax.imshow(im)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    fig.suptitle(f"1 主推 + 2 视觉备选（范围={mode}，特征={feature_kind}）", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def to_md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """DataFrame → Markdown 表格（不依赖 tabulate）。"""
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            cells.append(f"{float(v):{floatfmt}}" if isinstance(v, (float, np.floating)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 92)
    print("建议 3 离线验证：1 主推 + 2 视觉备选（不改模型、不改演示）")
    print("=" * 92)
    print(f"  设备：{device}")

    memes = demo.load_memes()
    label_idx = memes["emotion"].map({e: i for i, e in enumerate(CLASS_NAMES)}).to_numpy()
    print(f"  meme 数量：{len(memes)}")

    z = np.load(RESULT_DIR / "proto_face_probs_val.npz")
    probs, y_val = z["p_tta"], z["y_val"].astype(np.int64)
    print(f"  验证集：{len(probs)} 张（复用 09 缓存的概率）")

    print("\n① 提取 meme 视觉特征（两套）")
    meme_tensors = build_meme_tensor(memes)
    feats = {}
    for kind in ("imgnet", "fer"):
        t = time.time()
        feats[kind] = extract_features(kind, meme_tensors, device)
        print(f"  {kind:7s}: {feats[kind].shape}  用时 {time.time()-t:.1f}s")

    print("\n② 评估 3 种备选范围 × 2 种视觉特征")
    rows = []
    for feature_kind in ("imgnet", "fer"):
        for mode in ("same", "majority", "free"):
            t = time.time()
            r = eval_combo(memes, label_idx, feats[feature_kind], probs, y_val, mode, feature_kind)
            rows.append(r)
            print(f"  {feature_kind:7s} × {mode:9s} 完成（{time.time()-t:.1f}s）")
    df = pd.DataFrame(rows).sort_values(["feature", "pairwise_dist_mean"], ascending=[True, False])
    df.to_csv(RESULT_DIR / "proto_visual_diversity_results.csv", index=False, encoding="utf-8-sig")

    # 打印对比表
    show = ["feature", "mode", "main_emotion_mismatch", "same_emotion_alt_ratio", "alt_s_weighted",
            "pairwise_dist_mean", "random_pairwise_dist_mean", "near_dup_pairs",
            "reroll_mean_distinct_sets", "reroll_changed_ratio"]
    print("\n  组合对比（pairwise_dist_mean 越大越“不重样”；random 列为同约束随机备选基线）")
    print("  " + df[show].to_string(index=False, float_format=lambda x: f"{x:.3f}").replace("\n", "\n  "))

    # 结论
    print("\n③ 结论要点")
    base = df[df["mode"] == "same"].iloc[0]
    best = df.sort_values("pairwise_dist_mean", ascending=False).iloc[0]
    for kind in ("imgnet", "fer"):
        sub = df[df["feature"] == kind]
        best_k = sub.sort_values("pairwise_dist_mean", ascending=False).iloc[0]
        print(f"  {kind:7s}: 主推情绪不一致条数={int(best_k['main_emotion_mismatch'])}（应为 0）；"
              f"最好组合 {best_k['mode']:9s} 两两距离 {best_k['pairwise_dist_mean']:.3f}，"
              f"随机基线 {best_k['random_pairwise_dist_mean']:.3f}；"
              f"备选同类比例 {best_k['same_emotion_alt_ratio']:.2f}；近重复对 {int(best_k['near_dup_pairs'])}")
    print(f"  整体最佳：feature={best['feature']} mode={best['mode']} "
          f"pairwise_dist={best['pairwise_dist_mean']:.3f}（随机基线 {best['random_pairwise_dist_mean']:.3f}）")

    print("\n④ 生成样例拼图")
    make_example_figure(memes, feats["imgnet"], probs, label_idx, "majority", "imgnet",
                        RESULT_DIR / "proto_visual_examples_imgnet.png")
    make_example_figure(memes, feats["fer"], probs, label_idx, "majority", "fer",
                        RESULT_DIR / "proto_visual_examples_fer.png")
    print("  proto_visual_examples_imgnet.png / proto_visual_examples_fer.png")

    # 报告
    report = []
    report.append("# 建议 3 离线验证：1 主推 + 2 视觉备选\n")
    report.append("- 数据：58 张 meme；验证集 3589 张（复用 09 缓存概率）")
    report.append("- 主推逻辑：软匹配选中情绪 + 同类随机（与 07 修复后一致）")
    report.append("- 备选：贪心 max-min（备选1 离主推最远；备选2 到已选两张的最小距离最大）")
    report.append("- 特征：imgnet = ImageNet ResNet18；fer = FER 微调 ResNet18")
    report.append("- 距离 = 1 - 余弦相似度（0 = 一样，2 = 完全相反）")
    report.append("")
    report.append("## 1. 6 组合对比\n")
    report.append(to_md_table(df[show], floatfmt=".3f"))
    report.append("")
    report.append("## 2. 解读\n")
    report.append("- `main_emotion_mismatch` 应全为 0：备选策略没有改变主推情绪；")
    report.append("- `pairwise_dist_mean` 越大说明三张越“不重样”；与 `random_pairwise_dist_mean` 对比看视觉选择是否有增量；")
    report.append("- `same_emotion_alt_ratio`：备选里同情绪比例（仅同类模式应=1.00）；")
    report.append("- `near_dup_pairs`：三张两两距离 <0.05（相似度>0.95）的对数，越少越好；")
    report.append("- `reroll_*`：模拟 1 次初始 + 5 次换一批，平均不同组合数 / 换一批成功率；")
    report.append("- 结论之后再定默认值（当前默认：same + imgnet）。")
    (RESULT_DIR / "proto_visual_diversity_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\n完成，总用时 {time.time()-t0:.1f}s")
    print("产物：结果\\proto_visual_diversity_report.md / .csv / proto_visual_examples_*.png")


if __name__ == "__main__":
    main()
