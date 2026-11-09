# -*- coding: utf-8 -*-
r"""
09_meme_softmatch_proto.py —— 离线原型：猫 meme 的 7 维软标签 + 距离度量/聚类匹配（不改模型、不改演示）

这是"软标签 + 聚类匹配"想法的离线验证脚本。它只读现有数据/权重，产出对比结果，
**不会修改 07_demo_app.py 的匹配逻辑，也不会改动任何模型权重**。

核心思路（与现在软匹配的关系）：
  现在的软匹配：score(meme) = Σ_i p_i × S[i, meme 的情绪] = p · (S·e_meme)
  即每张 meme 隐式对应 7 维向量 v = S·e。本脚本把 v 换成"自动生成的独立 7 维向量"，
  并用多种"距离/相似度"去匹配人脸概率向量 p，另外尝试 KMeans 聚类做标签平滑/多样性。

7 维向量的 3 种全自动来源（不需要人工标注）：
  A. rule   ：onehot 情绪 × 相似度矩阵 S 平滑并归一化（等价于现在效果的向量形式，作为 baseline）
  B. model  ：用当前最终模型（ResNet18 迁移学习 + TTA）直接对 meme 图片预测 7 类概率
              （注意：猫 meme 对 FER 人脸模型是"域外输入"，预测可能有偏差，所以还做混合）
  C. hybrid ：α·model + (1-α)·rule，α 取 0.3/0.5/0.7 扫描

匹配度量：点积 / 余弦 / L1（曼哈顿）/ L2（欧氏）/ JS 散度；聚类做质心平滑。

评估（关键难点：没有"标准答案"，所以用代理指标 + 真实标签代理）：
  用验证集 3589 张人脸的"真实情绪标签"当代理：
    · label_hit      选中的 meme 情绪 == 人脸真实情绪（越低不一定越差，见报告解释）
    · sim_weighted   S[真实情绪, 选中 meme 情绪]（允许"近义"得分；比硬命中更公平）
    · top3_hit       得分 Top-3 的 meme 里是否包含真实情绪
    · stability      p 普通推理 vs TTA 推理，两次选中的 meme 是否一致（越高越稳）
    · unique_memes   3589 次查询一共用到多少张不同的 meme（多样性）
    · argmax_agree   选中 meme 情绪 == 模型预测情绪 的比例（策略偏离模型的程度）
    · hit_if_correct 只统计"模型预测正确"的样本上的 label_hit（看匹配层是否拖后腿）
    · lowconf_*      只统计"模型置信度 < 0.35"的难样本
  说明：匹配层无法修正模型错误，所以 label_hit 的理论上限就是模型准确率（约 69.7%）。

运行（PowerShell）：
    $env:PYTHONIOENCODING='utf-8'
    python <项目根>/代码/09_meme_softmatch_proto.py

产出（都在 结果\ 下）：
    proto_meme_vectors.csv        58 张 meme 的 7 维向量（rule/model/hybrid）+ 聚类编号
    proto_face_probs_val.npz      验证集人脸概率（普通/TTA）+ 真实标签（可复用，省推理时间）
    proto_matching_results.csv    所有策略的指标对比表
    proto_matching_report.md      可读版对比报告
    proto_meme_clusters.png       聚类 PCA 可视化
    proto_metric_bars.png         关键指标柱状图
"""
import os
import pathlib
import sys

# 本机 conda 与 torch 各带一份 Intel OpenMP；torch + matplotlib 一起用会报 OMP Error #15
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# 迁移学习权重走 HuggingFace 镜像（本脚本不下载权重，但保持和 08 一致的环境变量）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import importlib.util
import json
import time

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "代码"
RESULT_DIR = ROOT / "结果"
DATA_DIR = ROOT / "数据"
MODEL_DIR = ROOT / "模型"
DEPS_DIR = ROOT / "_deps"          # timm 等第三方库装在这里（迁移学习模型需要）
if str(DEPS_DIR) not in sys.path:
    sys.path.insert(0, str(DEPS_DIR))

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
WEIGHTS = MODEL_DIR / "best_model.pth"

# 与 07_demo_app.py 里的 EMOTION_SIM 保持一致（同一个 7×7 情绪相似度矩阵）
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
    """按路径加载同目录脚本模块（文件名数字开头，不能直接 import）。"""
    spec = importlib.util.spec_from_file_location(name, CODE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ds = load_module("fer_dataset", "03_dataset.py")     # FERTransferDataset（128 RGB ImageNet 归一化）
demo = load_module("demo_app", "07_demo_app.py")     # 复用：load_memes / preprocess_image / build_model_from_ckpt / tta_views


# ---------------- 1. 生成 meme 的 7 维向量 ----------------

def normalize_rows(a: np.ndarray) -> np.ndarray:
    """把每行归一化成概率分布（和为 1），避免全 0 行。"""
    a = np.clip(a.astype(np.float64), 0, None)
    s = a.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return a / s


def build_rule_vectors(memes: pd.DataFrame) -> np.ndarray:
    """A. 规则向量：v = S[:, meme 情绪]，再归一化。这就是现在软匹配的隐式向量。"""
    emo_idx = memes["emotion"].map({e: i for i, e in enumerate(CLASS_NAMES)}).to_numpy()
    v = EMOTION_SIM[:, emo_idx].T          # (M,7)，每一行 S[:,k]
    return normalize_rows(v)


def build_onehot_vectors(memes: pd.DataFrame) -> np.ndarray:
    """onehot 向量（作为"不加工"的对照）。"""
    emo_idx = memes["emotion"].map({e: i for i, e in enumerate(CLASS_NAMES)}).to_numpy()
    v = np.zeros((len(memes), len(CLASS_NAMES)), dtype=np.float64)
    v[np.arange(len(memes)), emo_idx] = 1.0
    return v


@torch.no_grad()
def infer_batch(model, xb: torch.Tensor, device: torch.device, tta: bool = True) -> np.ndarray:
    """对一批图做前向，返回 softmax 概率 (N,7)；tta=True 时对 6 个视角取平均。"""
    model.eval()
    xb = xb.to(device)
    views = demo.tta_views(xb) if tta else [xb]
    p = torch.zeros(len(xb), len(CLASS_NAMES), device=device)
    for v in views:
        p += torch.softmax(model(v), dim=1)
    return (p / len(views)).cpu().numpy()


def build_model_vectors(model, memes: pd.DataFrame, device) -> np.ndarray:
    """B. 模型向量：用当前分类器对每张 meme 图预测 7 类概率（多视角平均，降低随机性）。"""
    tensors = []
    for path in memes["path"]:
        with Image.open(path) as im:
            x = demo.preprocess_image(im, IMAGENET_MEAN, IMAGENET_STD, out_size=128, rgb=True)
        tensors.append(x)
    xb = torch.cat(tensors, dim=0)                 # (M,3,128,128)
    out = []
    for i in range(0, len(xb), 16):
        out.append(infer_batch(model, xb[i:i+16], device, tta=True))
    return np.concatenate(out, axis=0)


def build_hybrid_vectors(v_rule: np.ndarray, v_model: np.ndarray, alpha: float) -> np.ndarray:
    """C. 混合向量：α·模型向量 + (1-α)·规则向量；α 越大越"相信模型对猫图的感觉"。"""
    return normalize_rows(alpha * v_model + (1 - alpha) * v_rule)


# ---------------- 2. 距离/相似度 ----------------

def js_divergence(p: np.ndarray, v: np.ndarray) -> np.ndarray:
    """JS 散度（对称版 KL 平均），p:(N,7) v:(M,7) → (N,M) 距离矩阵。"""
    p = p[:, None, :]
    v = v[None, :, :]
    m = 0.5 * (p + v)
    eps = 1e-12

    def kl(a, b):
        return np.sum(np.where(a > 0, a * np.log((a + eps) / (b + eps)), 0.0), axis=-1)

    return 0.5 * kl(p, m) + 0.5 * kl(v, m)


def score_matrix(p: np.ndarray, v: np.ndarray, metric: str) -> np.ndarray:
    """计算人脸概率 p 与 meme 向量 v 的匹配得分（越大越好），返回 (N, M)。"""
    if metric in ("dot", "cosine"):
        dot = p @ v.T
        if metric == "dot":
            return dot
        norm = np.linalg.norm(p, axis=1, keepdims=True) @ np.linalg.norm(v, axis=1, keepdims=True).T
        return dot / np.maximum(norm, 1e-12)
    if metric == "l1":
        return -np.abs(p[:, None, :] - v[None, :, :]).sum(axis=-1)
    if metric == "l2":
        return -np.sqrt(np.maximum(((p[:, None, :] - v[None, :, :]) ** 2).sum(axis=-1), 0))
    if metric == "js":
        return -js_divergence(p, v)
    raise ValueError(f"未知 metric: {metric}")


# ---------------- 3. 聚类（KMeans，自动选 k） ----------------

def fit_kmeans(vectors: np.ndarray, k_min: int = 2, k_max: int = 12, seed: int = 42):
    """在 7 维向量上跑 KMeans，用轮廓系数选最佳 k；返回 (labels, kmeans, k, silhouette)。"""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    best = {"score": -1.0, "k": k_min, "labels": None, "km": None}
    for k in range(k_min, min(k_max, len(vectors) - 1) + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(vectors)
        if len(set(km.labels_)) < 2:
            continue
        s = silhouette_score(vectors, km.labels_)
        if s > best["score"]:
            best = {"score": float(s), "k": k, "labels": km.labels_, "km": km}
    return best["labels"], best["km"], best["k"], best["score"]


def smooth_by_cluster(vectors: np.ndarray, labels: np.ndarray, beta: float = 0.3) -> np.ndarray:
    """聚类质心平滑：v' = (1-β)·v + β·簇质心，再归一化。用来缓解个别向量噪声。"""
    centroids = np.zeros_like(vectors)
    for c in np.unique(labels):
        mask = labels == c
        centroids[mask] = vectors[mask].mean(axis=0, keepdims=True)
    return normalize_rows((1 - beta) * vectors + beta * centroids)


# ---------------- 4. 选图策略 ----------------

def select_argmax(scores: np.ndarray) -> np.ndarray:
    """每个查询选得分最高的 meme；并列时返回靠前的（确定性，方便复现）。"""
    return scores.argmax(axis=1)


def select_hard(p: np.ndarray, memes: pd.DataFrame, seed: int = 2026) -> np.ndarray:
    """硬匹配 baseline：取 argmax 情绪 → 在该类候选里随机选一张（多次取平均，本函数只返回一次）。"""
    rng = np.random.default_rng(seed)
    labels = memes["emotion_idx"].to_numpy()
    pred = p.argmax(axis=1)
    out = np.empty(len(p), dtype=np.int64)
    for i, c in enumerate(pred):
        cand = np.flatnonzero(labels == c)
        out[i] = cand[rng.integers(len(cand))] if len(cand) else 0
    return out


def select_with_diversity(scores: np.ndarray, cluster_of_meme: np.ndarray,
                          tolerance: float = 0.95) -> np.ndarray:
    """多样性后处理：如果这次选中的 meme 和上一次相同，则从"不同簇且得分接近"的候选里换一张。

    用途：演示中同一张图反复出现观感差；离线指标里看"多样性提升换多少命中率"。
    """
    out = np.empty(len(scores), dtype=np.int64)
    last = -1
    for i in range(len(scores)):
        order = np.argsort(scores[i])[::-1]
        pick = int(order[0])
        if pick == last:
            best = scores[i, pick]
            for j in order[1:30]:
                j = int(j)
                if cluster_of_meme[j] != cluster_of_meme[pick] and scores[i, j] >= tolerance * best:
                    pick = j
                    break
        out[i] = pick
        last = pick
    return out


# ---------------- 5. 评估指标 ----------------

def to_md_table(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """把 DataFrame 转成 Markdown 表格（不依赖 tabulate，pandas.to_markdown 需要额外依赖）。"""
    headers = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in df.columns:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append(f"{float(v):{floatfmt}}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def evaluate_selections(sel: np.ndarray, p: np.ndarray, y: np.ndarray,
                        memes: pd.DataFrame,
                        sel_tta: np.ndarray | None = None,
                        low_conf_thr: float = 0.35) -> dict:
    """计算一个策略的全部代理指标。"""
    meme_labels = memes["emotion"].to_numpy()
    chosen_label = meme_labels[sel]                       # 选中的 meme 情绪名
    chosen_idx = memes["emotion_idx"].to_numpy()[sel]     # 选中的 meme 情绪下标
    pred = p.argmax(axis=1)

    label_hit = (chosen_label == np.array(CLASS_NAMES)[y]).mean()
    sim_weighted = EMOTION_SIM[y, chosen_idx].mean()
    argmax_agree = (chosen_idx == pred).mean()
    correct_mask = pred == y
    hit_if_correct = ((chosen_label[correct_mask] == np.array(CLASS_NAMES)[y][correct_mask]).mean()
                      if correct_mask.any() else float("nan"))

    # 多样性：用到的不同 meme 数、选择分布熵、每类平均用到多少张不同 meme
    counts = np.bincount(sel, minlength=len(memes))
    unique_memes = int((counts > 0).sum())
    share = counts / counts.sum()
    entropy = float(-(share[share > 0] * np.log(share[share > 0])).sum())

    per_class_div = {}
    for ci, cname in enumerate(CLASS_NAMES):
        m = y == ci
        if m.any():
            per_class_div[cname] = int(len(np.unique(sel[m])))
    mean_class_div = float(np.mean(list(per_class_div.values())))

    # 低置信度子集
    low = p.max(axis=1) < low_conf_thr
    low_metrics = {}
    if low.sum() > 0:
        low_metrics = {
            "n": int(low.sum()),
            "label_hit": float((chosen_label[low] == np.array(CLASS_NAMES)[y][low]).mean()),
            "sim_weighted": float(EMOTION_SIM[y[low], chosen_idx[low]].mean()),
            "argmax_agree": float((chosen_idx[low] == pred[low]).mean()),
        }

    # 稳定性：普通推理 vs TTA 推理 选中的 meme 是否一致
    stability = float((sel == sel_tta).mean()) if sel_tta is not None else float("nan")

    return {
        "label_hit": float(label_hit),
        "sim_weighted": float(sim_weighted),
        "argmax_agree": float(argmax_agree),
        "hit_if_correct": float(hit_if_correct),
        "stability": stability,
        "unique_memes": unique_memes,
        "diversity_entropy": entropy,
        "mean_class_diversity": mean_class_div,
        "low_conf_n": low_metrics.get("n", 0),
        "low_conf_label_hit": low_metrics.get("label_hit", float("nan")),
        "low_conf_sim_weighted": low_metrics.get("sim_weighted", float("nan")),
    }


def topk_hit(scores: np.ndarray, y: np.ndarray, memes: pd.DataFrame, k: int = 3) -> float:
    """Top-k 候选里是否包含"真实情绪"的 meme（k=1 时等于 label_hit）。"""
    meme_idx = memes["emotion_idx"].to_numpy()
    topk = np.argsort(-scores, axis=1)[:, :k]
    hit = 0
    for i in range(len(y)):
        if np.any(meme_idx[topk[i]] == y[i]):
            hit += 1
    return hit / len(y)


# ---------------- 6. 主流程 ----------------

def main() -> None:
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 96)
    print("离线原型：猫 meme 7 维软标签 + 距离/聚类匹配（不改模型、不改演示）")
    print("=" * 96)
    print(f"  设备：{device}   权重：{WEIGHTS}")
    if not WEIGHTS.exists():
        raise SystemExit(f"找不到权重：{WEIGHTS}")

    # ---- 6.1 加载模型 + meme ----
    ckpt = demo.load_checkpoint(WEIGHTS, device)
    model = demo.build_model_from_ckpt(ckpt, device)
    memes = demo.load_memes()
    memes["emotion_idx"] = memes["emotion"].map({e: i for i, e in enumerate(CLASS_NAMES)})
    print(f"  模型：{ckpt.get('profile')}   meme 数量：{len(memes)}")
    print("  meme 每类数量：" + ", ".join(f"{e}={n}" for e, n in memes["emotion"].value_counts().items()))

    # ---- 6.2 生成三种 7 维向量 ----
    print("\n① 生成 meme 的 7 维向量（全自动）")
    v_onehot = build_onehot_vectors(memes)
    v_rule = build_rule_vectors(memes)
    v_model = build_model_vectors(model, memes, device)
    hybrids = {a: build_hybrid_vectors(v_rule, v_model, a) for a in (0.3, 0.5, 0.7)}
    print("  rule 向量示例（angry 类前 3 张）：", np.round(v_rule[:3], 2).tolist())
    print("  model 向量示例（前 3 张）：", np.round(v_model[:3], 2).tolist())

    # 模型对猫 meme 的预测分布：按 meme 真实情绪分组看"模型觉得它们像什么"
    pred_model = v_model.argmax(axis=1)
    crosstab = pd.crosstab(memes["emotion"], [CLASS_NAMES[i] for i in pred_model])
    print("\n  模型对猫 meme 的预测交叉表（行=人工情绪，列=模型预测）：")
    print("  " + crosstab.to_string().replace("\n", "\n  "))

    # ---- 6.3 聚类（在 hybrid0.5 上做，供可视化/平滑/多样性用） ----
    print("\n② KMeans 聚类（自动选 k）")
    v_cluster_base = hybrids[0.5]
    cluster_labels, km, k_best, sil = fit_kmeans(v_cluster_base)
    print(f"  最佳 k={k_best}，轮廓系数={sil:.3f}")
    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index().to_dict()
    print(f"  各簇大小：{cluster_sizes}")
    v_smooth = smooth_by_cluster(v_cluster_base, cluster_labels, beta=0.3)

    # 保存 meme 向量表（CSV 里放完整 7 维；npz 里存全部向量，供后续复用）
    vec_df = memes[["file", "emotion", "source"]].copy()
    for i, c in enumerate(CLASS_NAMES):
        vec_df[f"rule_{c}"] = np.round(v_rule[:, i], 4)
    for i, c in enumerate(CLASS_NAMES):
        vec_df[f"model_{c}"] = np.round(v_model[:, i], 4)
    for i, c in enumerate(CLASS_NAMES):
        vec_df[f"hybrid05_{c}"] = np.round(hybrids[0.5][:, i], 4)
    vec_df["cluster"] = cluster_labels
    vec_df.to_csv(RESULT_DIR / "proto_meme_vectors.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(RESULT_DIR / "proto_meme_vectors.npz",
                        files=memes["file"].to_numpy(), emotion=memes["emotion"].to_numpy(),
                        v_onehot=v_onehot, v_rule=v_rule, v_model=v_model,
                        **{f"v_hybrid{a}": v for a, v in hybrids.items()},
                        v_smooth=v_smooth, cluster=cluster_labels)
    print(f"  已保存：结果\\proto_meme_vectors.csv / .npz")

    # ---- 6.4 验证集人脸概率（只跑一次，之后复用） ----
    probs_path = RESULT_DIR / "proto_face_probs_val.npz"
    if probs_path.exists():
        z = np.load(probs_path)
        p_norm, p_tta, y_val = z["p_norm"], z["p_tta"], z["y_val"]
        print(f"\n③ 复用已缓存的人脸概率：{probs_path.name}（{len(y_val)} 张）")
    else:
        print("\n③ 用最终模型对验证集 3589 张人脸做推理（普通 + TTA，只跑一次）")
        data = ds.load_fer2013()
        X_val, y_val = data["val"]
        Xr = ds.resize_split(X_val, 128)
        dset = ds.FERTransferDataset(Xr, y_val, out_size=128, augment=False)
        loader = torch.utils.data.DataLoader(dset, batch_size=256, shuffle=False, num_workers=0,
                                             pin_memory=torch.cuda.is_available())
        norm_list, tta_list = [], []
        t0 = time.time()
        for xb, _ in loader:
            norm_list.append(infer_batch(model, xb, device, tta=False))
            tta_list.append(infer_batch(model, xb, device, tta=True))
        p_norm = np.concatenate(norm_list)
        p_tta = np.concatenate(tta_list)
        y_val = y_val.astype(np.int64)
        np.savez_compressed(probs_path, p_norm=p_norm, p_tta=p_tta, y_val=y_val)
        print(f"  完成，用时 {time.time()-t0:.1f}s；普通准确率 {(p_norm.argmax(1)==y_val).mean():.4f}，"
              f"TTA 准确率 {(p_tta.argmax(1)==y_val).mean():.4f}")

    # ---- 6.5 策略对比 ----
    print("\n④ 匹配策略对比（验证集 3589 张，真实情绪标签作代理指标）")
    strategies = []

    # 基线 1：硬匹配（规则：argmax 类里随机挑；稳定性用普通/TTA 各跑一次比较）
    hard_sel = select_hard(p_norm, memes)
    hard_sel_tta = select_hard(p_tta, memes)
    m = evaluate_selections(hard_sel, p_norm, y_val, memes, sel_tta=hard_sel_tta)
    m.update({"strategy": "hard（硬匹配 baseline）", "note": "argmax 类内随机", "top3_hit": float("nan")})
    strategies.append(m)

    # 基线 2：现方案 = p · rule（软匹配-矩阵）
    scores_old = score_matrix(p_norm, v_rule, "dot")
    sel_old = select_argmax(scores_old)
    sel_old_tta = select_argmax(score_matrix(p_tta, v_rule, "dot"))
    m = evaluate_selections(sel_old, p_norm, y_val, memes, sel_tta=sel_old_tta)
    m.update({"strategy": "soft-matrix（现方案：p·S·e）", "note": "点积", "top3_hit": topk_hit(scores_old, y_val, memes, 3)})
    strategies.append(m)

    # 其他策略：向量来源 × 度量
    vector_sets = {
        "onehot": v_onehot,
        "rule": v_rule,
        "model": v_model,
        "hybrid0.3": hybrids[0.3],
        "hybrid0.5": hybrids[0.5],
        "hybrid0.7": hybrids[0.7],
        "hybrid0.5_smooth": v_smooth,
    }
    metrics = ["dot", "cosine", "l1", "l2", "js"]
    for v_name, v in vector_sets.items():
        for metric in metrics:
            # dot 只有概率向量才有明确意义；onehot/rule 上的 dot 也保留作对照
            name = f"{v_name} + {metric}"
            s_norm = score_matrix(p_norm, v, metric)
            s_tta = score_matrix(p_tta, v, metric)
            sel = select_argmax(s_norm)
            sel_tta = select_argmax(s_tta)
            m = evaluate_selections(sel, p_norm, y_val, memes, sel_tta=sel_tta)
            m.update({"strategy": name, "note": "", "top3_hit": topk_hit(s_norm, y_val, memes, 3)})
            strategies.append(m)

    # 多样性后处理版本（在 hybrid0.5+cosine 上做示范）
    s_best_base = score_matrix(p_norm, hybrids[0.5], "cosine")
    sel_div = select_with_diversity(s_best_base, cluster_of_meme=cluster_labels)
    sel_div_tta = select_with_diversity(score_matrix(p_tta, hybrids[0.5], "cosine"), cluster_labels)
    m = evaluate_selections(sel_div, p_norm, y_val, memes, sel_tta=sel_div_tta)
    m.update({"strategy": "hybrid0.5 + cosine + 多样性后处理", "note": "避免连续同图/同簇",
              "top3_hit": topk_hit(s_best_base, y_val, memes, 3)})
    strategies.append(m)

    # 低置信回退：p 置信度低时用 rule 向量兜底，否则用 hybrid0.5+cosine
    s_hyb = score_matrix(p_norm, hybrids[0.5], "cosine")
    low_mask = p_norm.max(axis=1) < 0.35
    sel_fb = s_hyb.argmax(axis=1)
    sel_fb[low_mask] = select_argmax(score_matrix(p_norm[low_mask], v_rule, "dot"))
    sel_fb_tta = s_hyb.argmax(axis=1)
    low_tta = p_tta.max(axis=1) < 0.35
    sel_fb_tta[low_tta] = select_argmax(score_matrix(p_tta[low_tta], v_rule, "dot"))
    m = evaluate_selections(sel_fb, p_norm, y_val, memes, sel_tta=sel_fb_tta)
    m.update({"strategy": "hybrid0.5 + cosine + 低置信回退", "note": "置信度<0.35 用 rule",
              "top3_hit": topk_hit(s_hyb, y_val, memes, 3)})
    strategies.append(m)

    df = pd.DataFrame(strategies)
    cols = ["strategy", "label_hit", "sim_weighted", "top3_hit", "argmax_agree", "stability",
            "unique_memes", "diversity_entropy", "low_conf_sim_weighted", "hit_if_correct", "note"]
    df = df[cols].sort_values(["sim_weighted", "label_hit"], ascending=False).reset_index(drop=True)
    df.to_csv(RESULT_DIR / "proto_matching_results.csv", index=False, encoding="utf-8-sig")

    # 打印精简表（只打印关键列）
    show_cols = ["strategy", "label_hit", "sim_weighted", "top3_hit", "stability",
                 "unique_memes", "low_conf_sim_weighted", "argmax_agree"]
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 46)
    print("\n  " + df[show_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}").replace("\n", "\n  "))

    # 基线/最优对比摘要
    base = df[df["strategy"].str.startswith("soft-matrix")].iloc[0]
    best_sim = df.iloc[0]
    hard = df[df["strategy"].str.startswith("hard")].iloc[0]
    print("\n  关键对比（验证集）：")
    print(f"    hard        : label_hit={hard['label_hit']:.3f}  sim_weighted={hard['sim_weighted']:.3f}  "
          f"unique={hard['unique_memes']}")
    print(f"    现方案 soft-matrix: label_hit={base['label_hit']:.3f}  sim_weighted={base['sim_weighted']:.3f}  "
          f"unique={base['unique_memes']}  stability={base['stability']:.3f}")
    print(f"    sim_weighted 最优策略: {best_sim['strategy']}  label_hit={best_sim['label_hit']:.3f}  "
          f"sim_weighted={best_sim['sim_weighted']:.3f}  unique={best_sim['unique_memes']}")

    # ---- 6.6 保存报告 ----
    report = []
    report.append("# 猫 meme 软标签 + 距离/聚类匹配 —— 离线原型结果\n")
    report.append(f"- 权重：`{WEIGHTS.name}`（{ckpt.get('profile')}）")
    report.append(f"- 验证集：3589 张；普通准确率 {(p_norm.argmax(1)==y_val).mean():.4f}，"
                  f"TTA 准确率 {(p_tta.argmax(1)==y_val).mean():.4f}")
    report.append(f"- meme：{len(memes)} 张；KMeans 最佳 k={k_best}（轮廓系数 {sil:.3f}）")
    report.append("")
    report.append("## 1. 指标对比（按 sim_weighted 排序，完整表见 proto_matching_results.csv）\n")
    report.append(to_md_table(df[show_cols], floatfmt=".3f"))
    report.append("")
    report.append("## 2. 模型对猫 meme 的预测交叉表（行=人工情绪，列=模型预测）\n")
    report.append(to_md_table(crosstab.reset_index(), floatfmt="d"))
    report.append("")
    report.append("## 3. 结论与解读\n")
    # 数据驱动的关键结论（全部来自本次运行结果）
    model_agreement = float((v_model.argmax(axis=1) == memes["emotion_idx"].to_numpy()).mean())
    hi_label = df.sort_values("label_hit", ascending=False).iloc[0]
    rule_l2 = df[df["strategy"] == "rule + l2"].iloc[0]
    report.append(f"1. **模型向量不可用**：用 FER 人脸模型去预测猫 meme，与人工情绪标签的一致率只有 "
                  f"**{model_agreement*100:.1f}%**（{int(round(model_agreement*len(memes)))}/{len(memes)}）；"
                  f"`model` 策略 label_hit 只有 {df[df['strategy'].str.startswith('model')]['label_hit'].max():.3f}，"
                  f"比现方案 {base['label_hit']:.3f} 差很多。**猫 meme 对 FER 模型是域外输入，不能当自动标注器。**")
    report.append(f"2. **全自动软标签没有实质提升**：`rule`（=现方案的向量形式）与 `onehot` 各种距离度量的 "
                  f"label_hit 都在 {min(base['label_hit'], rule_l2['label_hit']):.3f}~{hi_label['label_hit']:.3f}，"
                  f"sim_weighted 在 {min(base['sim_weighted'], rule_l2['sim_weighted']):.3f}~"
                  f"{max(base['sim_weighted'], rule_l2['sim_weighted']):.3f} 之间，差异不到 1 个百分点；"
                  f"距离度量（L1/L2/余弦/JS）之间也没有稳定优劣。")
    report.append(f"3. **现方案和 rule 策略有个隐藏问题：多样性极差**：同一情绪的所有 meme 向量完全相同，"
                  f"argmax 永远选中第一张 → 3589 次查询只用到 **{int(base['unique_memes'])} 张不同 meme**"
                  f"（hard 匹配用随机类内选择，用到 {int(hard['unique_memes'])} 张）。演示中会反复出现同一张图，"
                  f"这比“换距离度量”更值得修。")
    report.append(f"4. **聚类没有帮助**：KMeans k={k_best} 的轮廓系数只有 {sil:.2f}（结构很弱），"
                  f"聚类质心平滑后指标反而下降（hybrid0.5_smooth 系列 label_hit 明显低于 hybrid0.5）。")
    report.append(f"5. **Top-3 思路值得单独评估**：软向量策略可以把“真实情绪进入 Top-3”的比例提到约 "
                  f"{df['top3_hit'].max():.3f}，但对应 Top-1 命中只有 "
                  f"{df.loc[df['top3_hit'].idxmax(), 'label_hit']:.3f}；"
                  f"如果演示改成“1 张主推 + 2 张备选”，需要单独做用户研究，不能直接说更准。")
    report.append("")
    report.append("## 4. 总体判断\n")
    report.append("- 按“全自动标注”的路线，这个想法**目前不成立**：没有可利用的新信息，模型对猫图不可靠；")
    report.append("- 若要让软标签真正有价值，需要有信息量的向量来源：人工多情绪标注、CLIP/多模态零样本打标、"
                  "或表情专用模型；这些都不属于“全自动”范畴（至少要人工抽检）；")
    report.append("- 低成本可做的两个改进（与软标签无关）："
                  "① 软匹配加“类内随机/加权随机”，解决只反复用 7~8 张图的问题；"
                  "② 演示展示 Top-3 候选（1 主 + 2 备选），主观满意度可能比 Top-1 更高。")
    (RESULT_DIR / "proto_matching_report.md").write_text("\n".join(report), encoding="utf-8")

    # ---- 6.7 可视化 ----
    print("\n⑤ 生成图表")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from sklearn.decomposition import PCA

    # 中文字体：Windows 上一般有 Microsoft YaHei / SimHei，否则图上中文会变成方块
    for _font in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        if any(_font.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [_font]
            break
    plt.rcParams["axes.unicode_minus"] = False

    # 聚类 PCA 图
    pca = PCA(n_components=2).fit(v_cluster_base)
    xy = pca.transform(v_cluster_base)
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.get_cmap("tab10")
    for c in np.unique(cluster_labels):
        m = cluster_labels == c
        ax.scatter(xy[m, 0], xy[m, 1], s=40, color=cmap(c % 10), label=f"cluster {c}")
    for i, row in memes.iterrows():
        ax.annotate(row["emotion"][:3], (xy[i, 0], xy[i, 1]), fontsize=6, alpha=0.7)
    ax.set_title(f"KMeans on meme hybrid vectors (k={k_best}, silhouette={sil:.2f})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "proto_meme_clusters.png", dpi=140)
    plt.close(fig)

    # 关键指标柱状图：比较 hard / soft-matrix / 若干候选
    pick_names = ["hard（硬匹配 baseline）", "soft-matrix（现方案：p·S·e）"]
    for n in ["hybrid0.5 + cosine", "hybrid0.5 + l1", "hybrid0.7 + cosine",
              "model + cosine", "hybrid0.5_smooth + cosine"]:
        if n in set(df["strategy"]):
            pick_names.append(n)
    sub = df[df["strategy"].isin(pick_names)].copy()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, col, title in zip(axes, ["label_hit", "sim_weighted", "unique_memes"],
                              ["Label hit (真实情绪命中)", "S-weighted similarity", "Unique memes used (多样性)"]):
        y = np.arange(len(sub))
        ax.barh(y, sub[col].values, color="tab:blue", alpha=0.75)
        ax.set_yticks(y)
        ax.set_yticklabels([s[:34] for s in sub["strategy"]], fontsize=8)
        ax.set_title(title)
        ax.invert_yaxis()
        for i, v in enumerate(sub[col].values):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "proto_metric_bars.png", dpi=140)
    plt.close(fig)

    print(f"\n完成，总用时 {time.time()-t_start:.1f}s")
    print("产物：")
    print("  结果\\proto_matching_report.md      对比报告（含指标表 + 交叉表 + 结论）")
    print("  结果\\proto_matching_results.csv   全部策略指标")
    print("  结果\\proto_meme_vectors.csv/.npz  58 张 meme 的 7 维向量")
    print("  结果\\proto_face_probs_val.npz     验证集人脸概率缓存")
    print("  结果\\proto_meme_clusters.png      聚类 PCA 图")
    print("  结果\\proto_metric_bars.png        关键指标柱状图")


if __name__ == "__main__":
    main()
