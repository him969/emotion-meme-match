# -*- coding: utf-8 -*-
r"""
12_clip_probe.py —— 建议2 B1 阶段：CLIP 跨模态"神似"探针（只读验证集，不训练）

目的
----
在**不改模型、不碰测试集**的前提下，回答三个问题：
  1. CLIP 的视觉特征里，到底有没有"人脸表情 ↔ 猫 meme 情绪"的跨域信号？
  2. 如果有，信号有多强（对照：随机水平、以及现有 ImageNet ResNet18 特征）？
  3. 能不能把"同情绪类内按视觉相似度挑一张"做得比"随机一张"更"神似"？

数据卫生
--------
- 只用验证集 X_val / y_val（3589 张），**测试集一次都不碰**；
- FER 模型概率读 `结果\proto_face_probs_val.npz`（09 原型产出的 TTA 概率），当条件输入用；
- 不训练任何参数，CLIP 全部冻结（eval 模式）。

探针（决策门来自交接文档 §9，先定后测）
--------------------------------------
  P1 跨模态质心对齐：7 类人脸质心 vs 7 类 meme 质心的余弦矩阵，对角均值 − 非对角均值 ≥ +0.03
  P2 跨模态检索：每张人脸检索 58 张 meme 的 top-1 命中率（命中=meme 人工标签==人脸真实标签）≥ 0.33
  P3 零样本软标签：用 FER 人脸 7 个情绪原型给 meme 打标，与人工标签一致率 ≥ 40%（对照 22.4%）
  P4 对照组：同样的 P1/P2 用现有 ImageNet ResNet18 特征跑一遍（不设门槛，用于解释）
  P5 同类内视觉重排：按 FER 预测情绪挑类，再类内按 CLIP 相似度取最像的一张（不改变情绪命中率）
  P6 属性桥（需要 CLIP 文本编码器）：把"眼神/嘴型/气质"拆成属性，人脸与 meme 各自算属性画像再比

产物
----
  结果\proto_clip_features.npz   CLIP 特征（人脸 3589×512、meme 58×512、标签、文件名）
  结果\proto_clip_attr_meme.csv  meme 的属性画像（P6 成功时）
  结果\proto_clip_picks_val.csv  P5 每张验证集人脸的类内首选（供弱指标目检）
  结果\proto_clip_probe.txt      人读报告（门槛结论 + 表格）
  结果\proto_clip_probe.json     机器可读结果

运行（PowerShell）
------------------
  $env:PYTHONIOENCODING='utf-8'
  python 代码\12_clip_probe.py
  # 可选：--recompute 重算特征；--batch 128；--n-perm 10000；--n-boot 1000；--skip-attr 跳过 P6
"""
import os

# 同一进程里 torch + matplotlib 会撞两份 Intel OpenMP（见 PROJECT.md 3.2 节第 5 条）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# tokenizer 走 hf-mirror（直连 huggingface.co 会超时，见 PROJECT.md 3.2 节第 8 条）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json
import pathlib
import random
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEPS_DIR = ROOT / "_deps"
if str(DEPS_DIR) not in sys.path:
    sys.path.insert(0, str(DEPS_DIR))

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
FER_NPZ = ROOT / "数据" / "fer2013" / "fer2013.npz"
MEME_CSV = ROOT / "数据" / "meme_index.csv"
PROBS_NPZ = ROOT / "结果" / "proto_face_probs_val.npz"       # 09 原型产出：p_norm / p_tta / y_val
RESULT_DIR = ROOT / "结果"
CLIP_WEIGHTS = DEPS_DIR / "weights" / "open_clip_vitb32_laion2b.bin"
CLIP_TOKENIZER_DIR = DEPS_DIR / "weights" / "clip_tokenizer"
FEAT_NPZ = RESULT_DIR / "proto_clip_features.npz"
ATTR_CSV = RESULT_DIR / "proto_clip_attr_meme.csv"
PICKS_CSV = RESULT_DIR / "proto_clip_picks_val.csv"
REPORT_TXT = RESULT_DIR / "proto_clip_probe.txt"
REPORT_JSON = RESULT_DIR / "proto_clip_probe.json"
IMGNET_WEIGHTS = DEPS_DIR / "weights" / "resnet18.a1_in1k.safetensors"

# ---------------- 交接文档 §9 的决策门（先定死，跑完不改） ----------------
GATE_CENTROID = 0.03      # P1：对角均值 − 非对角均值
GATE_RETRIEVAL = 0.33     # P2：top-1 命中率
GATE_ZEROSHOT = 0.40      # P3：与人工标签一致率
CONTROL_ZEROSHOT = 0.224  # P3 对照：FER 模型自动打标的历史一致率（09 原型）

IMGNET_MEAN = (0.485, 0.456, 0.406)
IMGNET_STD = (0.229, 0.224, 0.225)

RNG = np.random.default_rng(2026)
RANDOM = random.Random(2026)


# =============================================================================
# 0. 通用工具
# =============================================================================

def l2norm(a: np.ndarray) -> np.ndarray:
    """按行 L2 归一化。"""
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)


def center_rows(a: np.ndarray) -> np.ndarray:
    """减去整体均值（去掉模态共同的"偏置方向"，再做余弦才有意义）。"""
    return a - a.mean(axis=0, keepdims=True)


def cos_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,d) × (M,d) → (N,M) 余弦相似度（a、b 需已 L2 归一化）。"""
    return a @ b.T


def class_centroids(feats: np.ndarray, labels: np.ndarray, centered: bool = True) -> np.ndarray:
    """按 7 类标签算质心并 L2 归一化；centered=True 时先做整体去均值。"""
    f = center_rows(feats) if centered else feats
    cents = np.stack([f[labels == c].mean(axis=0) for c in range(len(CLASS_NAMES))])
    return l2norm(cents)


def bootstrap_ci(values: np.ndarray, n_boot: int, rng: np.random.Generator,
                 stat=np.mean) -> tuple[float, float]:
    """对一组逐样本指标做 bootstrap 95% 置信区间。"""
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = np.array([stat(values[i]) for i in idx])
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def permutation_pvalue(observed: float, null: np.ndarray) -> float:
    """单侧置换检验 p 值。"""
    null = np.asarray(null)
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


# =============================================================================
# 1. 数据读取
# =============================================================================

def load_fer_val() -> tuple[np.ndarray, np.ndarray]:
    """读 FER2013 验证集（48×48 uint8 灰度 + int64 标签）。"""
    with np.load(FER_NPZ) as z:
        return z["X_val"], z["y_val"]


def load_val_probs() -> tuple[np.ndarray, np.ndarray]:
    """读 09 原型缓存的验证集 TTA 概率（p_norm / p_tta / y_val）。"""
    with np.load(PROBS_NPZ) as z:
        return z["p_tta"].astype(np.float64), z["y_val"].astype(np.int64)


def load_memes() -> pd.DataFrame:
    """读 meme 元数据表，并把 processed 相对路径转成绝对路径与 file 主干名。"""
    memes = pd.read_csv(MEME_CSV)
    memes["path"] = memes["processed"].map(lambda p: str(ROOT / pathlib.Path(p)))
    memes["stem"] = memes["file"].str.replace(r"\.[^.]+$", "", regex=True)
    return memes


# =============================================================================
# 2. 特征提取：CLIP（主） + ImageNet ResNet18（对照）
# =============================================================================

def load_clip(device: str):
    r"""加载项目内 laion CLIP ViT-B/32（open_clip 格式，离线本地权重）。

    权重与 config 已核对：vision 224/12 层/宽 768/patch 32，预处理 mean/std 为 OpenAI 值。
    """
    import open_clip
    import torch

    if not CLIP_WEIGHTS.exists():
        raise SystemExit(f"找不到 CLIP 权重：{CLIP_WEIGHTS}\n"
                         f"请先让 clip-env 下载（见 _probe_clip/clip_env_report.md）")
    t0 = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=str(CLIP_WEIGHTS), device=device
    )
    model.eval()
    infer_sec = time.time() - t0
    print(f"  [CLIP] 加载成功：ViT-B-32（laion2b） {infer_sec:.1f}s  参数={sum(p.numel() for p in model.parameters()):,}")
    return open_clip, model, preprocess


def load_imgnet_control(device: str):
    """对照用 ImageNet ResNet18（与 11_visual_diversity.py 的 imgnet 特征同款加载方式）。"""
    import timm
    import torch
    from safetensors.torch import load_file

    m = timm.create_model("resnet18", pretrained=False, num_classes=0)
    sd = load_file(str(IMGNET_WEIGHTS))
    m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("fc.")}, strict=False)
    return m.to(device).eval()


def encode_images_with_clip(model, preprocess, images: list[Image.Image],
                            device: str, batch: int = 128) -> np.ndarray:
    """PIL 图像列表 → CLIP 图像特征（已 L2 归一化，float32 numpy）。"""
    import torch

    out = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(images), batch):
            chunk = images[i:i + batch]
            tens = torch.stack([preprocess(im) for im in chunk]).to(device)
            feat = model.encode_image(tens).float()
            out.append(feat.cpu().numpy())
    feats = np.concatenate(out, axis=0)
    print(f"  [CLIP] 编码 {len(images)} 张，用时 {time.time() - t0:.1f}s（batch={batch}）")
    return l2norm(feats)


def fer_gray_to_pil(x48: np.ndarray) -> Image.Image:
    """48×48 uint8 灰度 → RGB PIL（CLIP 预处理内部会放大到 224）。"""
    return Image.fromarray(x48, mode="L").convert("RGB")


def meme_to_pil(path: str) -> Image.Image:
    """读 meme_ready 下的 256×256 PNG（白底 RGB）。"""
    with Image.open(path) as im:
        return im.convert("RGB")


def build_clip_features(device: str, batch: int, recompute: bool) -> dict:
    """计算/读取 CLIP 特征。缓存到 结果\\proto_clip_features.npz，重跑不重复编码。"""
    X, y = load_fer_val()
    memes = load_memes()
    if FEAT_NPZ.exists() and not recompute:
        with np.load(FEAT_NPZ) as z:
            if z["face"].shape[0] == len(X) and z["meme"].shape[0] == len(memes):
                print(f"  [缓存] 读 {FEAT_NPZ.name}（--recompute 可强制重算）")
                return {"face": z["face"], "meme": z["meme"],
                        "meme_emotion": z["meme_emotion"].astype(int),
                        "meme_file": z["meme_file"].astype(str),
                        "y": z["y"].astype(int)}

    open_clip, model, preprocess = load_clip(device)
    print(f"  人脸：{len(X)} 张（48×48 灰度 → 224 RGB，带放大信息损失）")
    face = encode_images_with_clip(model, preprocess, [fer_gray_to_pil(x) for x in X], device, batch)
    print(f"  meme：{len(memes)} 张（256×256 白底 → 224）")
    meme = encode_images_with_clip(model, preprocess, [meme_to_pil(p) for p in memes["path"]], device, batch)

    meme_emotion = memes["emotion"].map({e: i for i, e in enumerate(CLASS_NAMES)}).to_numpy(dtype=np.int64)
    meme_file = memes["file"].to_numpy(dtype=str)
    np.savez_compressed(FEAT_NPZ, face=face.astype(np.float32), meme=meme.astype(np.float32),
                        meme_emotion=meme_emotion, meme_file=meme_file, y=y.astype(np.int64))
    print(f"  特征已缓存：{FEAT_NPZ.relative_to(ROOT)}")
    return {"face": face, "meme": meme, "meme_emotion": meme_emotion, "meme_file": meme_file, "y": y}


def build_imgnet_features(device: str) -> dict:
    """对照特征：ImageNet ResNet18 全局池化（512 维）——人脸 128×128、meme 128×128。"""
    import torch

    X, y = load_fer_val()
    memes = load_memes()
    model = load_imgnet_control(device)

    def to_tensor(im: Image.Image, size: int = 128):
        im = im.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        arr = (arr - np.array(IMGNET_MEAN, dtype=np.float32)) / np.array(IMGNET_STD, dtype=np.float32)
        return torch.from_numpy(arr).permute(2, 0, 1)

    def extract(images: list[Image.Image], batch: int = 128) -> np.ndarray:
        out = []
        with torch.no_grad():
            for i in range(0, len(images), batch):
                tens = torch.stack([to_tensor(im) for im in images[i:i + batch]]).to(device)
                f = model.forward_features(tens)
                f = model.global_pool(f)
                out.append(torch.flatten(f, 1).float().cpu().numpy())
        return l2norm(np.concatenate(out, axis=0))

    # 人脸：48×48 灰度 → RGB（与 11 的 imgnet 处理一致：meme 是中心裁剪后 128）
    face = extract([Image.fromarray(x, mode="L").convert("RGB") for x in X])
    meme_imgs = []
    for p in memes["path"]:
        with Image.open(p) as im:
            im = im.convert("RGB")
            w, h = im.size
            side = min(w, h)
            left, top = (w - side) // 2, (h - side) // 2
            meme_imgs.append(im.crop((left, top, left + side, top + side)))
    meme = extract(meme_imgs)
    meme_emotion = memes["emotion"].map({e: i for i, e in enumerate(CLASS_NAMES)}).to_numpy(dtype=np.int64)
    return {"face": face, "meme": meme, "meme_emotion": meme_emotion}


# =============================================================================
# 3. P1 跨模态质心对齐
# =============================================================================

def probe_centroid(face: np.ndarray, y: np.ndarray, meme: np.ndarray, meme_emotion: np.ndarray,
                   n_perm: int = 10000, n_boot: int = 1000) -> dict:
    """7×7 质心余弦矩阵：对角均值 − 非对角均值；置换检验 + bootstrap CI。"""
    fc = class_centroids(face, y, centered=True)
    mc = class_centroids(meme, meme_emotion, centered=True)
    mat = cos_matrix(fc, mc)
    diag = np.diag(mat).mean()
    off = (mat.sum() - np.trace(mat)) / (49 - 7)
    observed = float(diag - off)

    # 置换：打乱 meme 的情绪标签（保持类大小），重算 meme 质心与矩阵
    null = []
    for _ in range(n_perm):
        lab = RNG.permutation(meme_emotion)
        mcp = l2norm(np.stack([center_rows(meme)[lab == c].mean(axis=0) for c in range(7)]))
        mp = cos_matrix(fc, mcp)
        null.append(float(np.diag(mp).mean() - (mp.sum() - np.trace(mp)) / 42))
    pval = permutation_pvalue(observed, null)

    # bootstrap：类内重采样人脸/meme
    fc_center = center_rows(face)
    mc_center = center_rows(meme)
    boots = []
    for _ in range(n_boot):
        f_cent = []
        m_cent = []
        for c in range(7):
            fi = np.flatnonzero(y == c)
            mi = np.flatnonzero(meme_emotion == c)
            f_cent.append(fc_center[RNG.choice(fi, size=len(fi), replace=True)].mean(axis=0))
            m_cent.append(mc_center[RNG.choice(mi, size=len(mi), replace=True)].mean(axis=0))
        fc_b = l2norm(np.stack(f_cent))
        mc_b = l2norm(np.stack(m_cent))
        mb = cos_matrix(fc_b, mc_b)
        boots.append(float(np.diag(mb).mean() - (mb.sum() - np.trace(mb)) / 42))
    lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)

    return {
        "matrix": mat, "diag_mean": float(diag), "offdiag_mean": float(off),
        "stat": observed, "p_value": pval, "ci95": [float(lo), float(hi)],
        "gate": GATE_CENTROID, "passed": observed >= GATE_CENTROID,
    }


# =============================================================================
# 4. P2 跨模态检索
# =============================================================================

def _retrieval_metrics(sim: np.ndarray, y: np.ndarray, meme_emotion: np.ndarray) -> dict:
    """给定人脸×meme 相似度矩阵，算 top-1/3/5 命中率与精确随机基线。"""
    order = np.argsort(-sim, axis=1)
    hit1 = (meme_emotion[order[:, 0]] == y)
    top3 = np.array([np.any(meme_emotion[order[i, :3]] == y[i]) for i in range(len(y))])
    top5 = np.array([np.any(meme_emotion[order[i, :5]] == y[i]) for i in range(len(y))])
    # 精确随机基线：每条人脸按真实类别，从 58 张 meme 里随机抽一张命中的概率
    p_face = np.bincount(y, minlength=7) / len(y)
    p_meme = np.bincount(meme_emotion, minlength=7) / len(meme_emotion)
    chance = float((p_face * p_meme).sum())
    return {"top1": float(hit1.mean()), "top3": float(top3.mean()), "top5": float(top5.mean()),
            "hit1": hit1, "top3_per": top3, "top5_per": top5, "chance": chance,
            "order": order}


def probe_retrieval(face: np.ndarray, y: np.ndarray, meme: np.ndarray, meme_emotion: np.ndarray,
                    n_perm: int = 10000, n_boot: int = 1000, label: str = "") -> dict:
    """跨模态检索：每张人脸在 58 张 meme 里找 top-k，命中=meme 标签==人脸真实标签。"""
    sim = cos_matrix(face, meme)
    m = _retrieval_metrics(sim, y, meme_emotion)
    lo, hi = bootstrap_ci(m["hit1"].astype(float), n_boot, RNG)
    lo3, hi3 = bootstrap_ci(m["top3_per"].astype(float), n_boot, RNG)

    # 置换：打乱 meme 情绪标签 → top-1 索引固定，只改标签比对
    null = []
    for _ in range(n_perm):
        lab = RNG.permutation(meme_emotion)
        null.append(float((lab[m["order"][:, 0]] == y).mean()))
    pval = permutation_pvalue(m["top1"], null)

    # 中心化相似度（去掉共同偏置）作为稳健性对照
    sim_c = cos_matrix(l2norm(center_rows(face)), l2norm(center_rows(meme)))
    mc = _retrieval_metrics(sim_c, y, meme_emotion)

    return {"label": label, "top1": m["top1"], "top3": m["top3"], "top5": m["top5"],
            "chance": m["chance"], "lift": float(m["top1"] - m["chance"]),
            "ci95_top1": [float(lo), float(hi)], "ci95_top3": [float(lo3), float(hi3)],
            "p_value": pval, "top1_centered": mc["top1"], "top3_centered": mc["top3"],
            "gate": GATE_RETRIEVAL, "passed": m["top1"] >= GATE_RETRIEVAL}


# =============================================================================
# 5. P3 零样本软标签（FER 人脸原型 → meme）
# =============================================================================

def probe_zeroshot(face: np.ndarray, y: np.ndarray, meme: np.ndarray,
                   meme_emotion: np.ndarray, n_perm: int = 10000, n_boot: int = 1000) -> dict:
    """用 7 个 FER 人脸情绪原型给 58 张 meme 打标，与人工标签比对。"""
    fc = class_centroids(face, y, centered=True)
    meme_c = l2norm(center_rows(meme))
    pred = np.argmax(cos_matrix(meme_c, fc), axis=1)
    agree = (pred == meme_emotion)
    acc = float(agree.mean())

    lo, hi = bootstrap_ci(agree.astype(float), n_boot, RNG)
    # 随机/多数类基线
    majority = float(np.bincount(meme_emotion, minlength=7).max() / len(meme_emotion))
    uniform = 1.0 / 7
    null = []
    for _ in range(n_perm):
        lab = RNG.permutation(meme_emotion)
        null.append(float((pred == lab).mean()))
    pval = permutation_pvalue(acc, null)

    conf = np.zeros((7, 7), dtype=int)     # 行=人工标签，列=CLIP 预测
    for t, p in zip(meme_emotion, pred):
        conf[t, p] += 1

    return {"agreement": acc, "ci95": [float(lo), float(hi)], "majority_baseline": majority,
            "uniform_baseline": uniform, "p_value": pval, "confusion": conf,
            "gate": GATE_ZEROSHOT, "control": CONTROL_ZEROSHOT, "passed": acc >= GATE_ZEROSHOT}


# =============================================================================
# 6. P4 对照组（ImageNet ResNet18 同协议）
# =============================================================================

def probe_control() -> dict:
    """用 ImageNet ResNet18 特征跑 P1/P2 同协议，作为"CLIP 是否真的更强"的对照。"""
    print("  [对照] ImageNet ResNet18 特征（同协议 P1/P2）...")
    feats = build_imgnet_features(DEVICE)
    c = probe_centroid(feats["face"], y_val, feats["meme"], feats["meme_emotion"])
    r = probe_retrieval(feats["face"], y_val, feats["meme"], feats["meme_emotion"], label="imgnet")
    return {"centroid_stat": c["stat"], "centroid_p": c["p_value"],
            "top1": r["top1"], "top3": r["top3"], "chance": r["chance"], "p_value": r["p_value"]}


# =============================================================================
# 7. P5 同类内视觉重排（神似模式的原型）
# =============================================================================

def probe_rerank(face: np.ndarray, meme: np.ndarray, meme_emotion: np.ndarray,
                 meme_file: np.ndarray, p_tta: np.ndarray) -> dict:
    """按 FER 预测情绪选类 → 类内按 CLIP 相似度取最像的一张。

    - 情绪命中率与"软匹配"完全一致（选类逻辑没改，只是类内从随机变最像）；
    - 输出 结果\\proto_clip_picks_val.csv 供后续目检/弱指标验收。
    """
    pred = p_tta.argmax(axis=1)
    pick_idx = np.empty(len(pred), dtype=int)
    img_sim = np.empty(len(pred), dtype=float)
    rand_idx = np.empty(len(pred), dtype=int)
    for i in range(len(pred)):
        cand = np.flatnonzero(meme_emotion == pred[i])
        s = face[i] @ meme[cand].T
        k = int(np.argmax(s))
        pick_idx[i] = cand[k]
        img_sim[i] = s[k]
        rand_idx[i] = cand[RANDOM.randrange(len(cand))]

    mean_sim = float(img_sim.mean())
    # 随机同类的期望相似度
    expected = float(np.mean([np.mean(face[i] @ meme[meme_emotion == pred[i]].T)
                              for i in range(0, len(pred), 7)]))  # 抽样加速
    unique = int(len(np.unique(pick_idx)))
    counts = np.bincount(pick_idx, minlength=len(meme_emotion))
    top_share = float(counts.max() / len(pred))
    # 连续重复（按验证集顺序，用实际展示的 avoid 逻辑会再降）
    consec = int(np.sum(pick_idx[1:] == pick_idx[:-1]))

    df = pd.DataFrame({
        "val_idx": np.arange(len(pred)),
        "y_true": [CLASS_NAMES[v] for v in y_val],
        "pred": [CLASS_NAMES[v] for v in pred],
        "p_pred": p_tta.max(axis=1),
        "chosen_file": meme_file[pick_idx],
        "chosen_emotion": [CLASS_NAMES[v] for v in meme_emotion[pick_idx]],
        "img_sim": img_sim,
        "baseline_random_file": meme_file[rand_idx],
    })
    df.to_csv(PICKS_CSV, index=False, encoding="utf-8-sig")
    print(f"  [P5] 类内视觉重排：mean cos={mean_sim:.4f}（随机同类≈{expected:.4f}），"
          f"用到 {unique}/{len(meme_emotion)} 张，最大单图占比 {top_share:.1%}，连续重复 {consec} 次")
    return {"mean_img_sim": mean_sim, "random_same_class_sim": expected,
            "unique_memes": unique, "total_memes": int(len(meme_emotion)),
            "max_single_share": top_share, "consecutive_repeats": consec,
            "csv": str(PICKS_CSV.relative_to(ROOT))}


# =============================================================================
# 8. P6 属性桥（眼神/嘴型/气质；需要文本编码器）
# =============================================================================

ATTRS = [
    # (属性名, 中文, 人脸正/负提示词, meme 正/负提示词)
    ("mouth_open", "张嘴", "a photo of a person whose mouth is open", "a photo of a person whose mouth is closed",
     "an image of a cat whose mouth is open", "an image of a cat whose mouth is closed"),
    ("teeth", "露齿", "a photo of a person showing their teeth", "a photo of a person with lips closed",
     "an image of a cat showing teeth", "an image of a cat with mouth closed"),
    ("tongue", "吐舌", "a photo of a person with their tongue out", "a photo of a person with tongue inside",
     "an image of a cat with its tongue out", "an image of a cat with tongue inside"),
    ("eyes_wide", "瞪眼", "a photo of a person with wide open eyes", "a photo of a person with relaxed eyes",
     "an image of a cat with wide open eyes", "an image of a cat with relaxed eyes"),
    ("eyes_narrow", "眯眼", "a photo of a person with narrowed squinting eyes", "a photo of a person with wide eyes",
     "an image of a cat with narrowed squinting eyes", "an image of a cat with wide eyes"),
    ("eyes_closed", "闭眼", "a photo of a person with closed eyes", "a photo of a person with open eyes",
     "an image of a cat with closed eyes", "an image of a cat with open eyes"),
    ("brows_furrowed", "皱眉", "a photo of a person with furrowed angry brows", "a photo of a person with relaxed brows",
     "an image of a cat with angry furrowed brows", "an image of a cat with relaxed brows"),
    ("smile", "微笑", "a photo of a smiling person", "a photo of a non-smiling person",
     "an image of a smiling cat", "an image of a non-smiling cat"),
    ("head_tilt", "歪头", "a photo of a person tilting their head", "a photo of a person looking straight",
     "an image of a cat tilting its head", "an image of a cat looking straight"),
    ("grumpy", "高冷嫌弃", "a photo of a grumpy disdainful person", "a photo of a cheerful friendly person",
     "an image of a grumpy disdainful cat", "an image of a cheerful friendly cat"),
    ("silly", "沙雕搞怪", "a photo of a silly goofy person", "a photo of a serious calm person",
     "an image of a silly goofy cat", "an image of a serious calm cat"),
    ("cute", "可爱卖萌", "a photo of a cute endearing person", "a photo of a plain ordinary person",
     "an image of a cute endearing cat", "an image of a plain ordinary cat"),
    ("pitiful", "委屈可怜", "a photo of a sad pitiful teary person", "a photo of a happy confident person",
     "an image of a sad pitiful teary cat", "an image of a happy confident cat"),
    ("shocked", "震惊", "a photo of a shocked astonished person", "a photo of an unsurprised person",
     "an image of a shocked astonished cat", "an image of an unsurprised cat"),
    ("smug", "得意", "a photo of a smug pleased person", "a photo of a humble modest person",
     "an image of a smug pleased cat", "an image of a humble modest cat"),
    ("closeup", "特写", "a close-up photo of a face", "a wide shot showing a whole body",
     "a close-up image of a cat's face", "a wide shot showing a whole cat body"),
]


def load_text_encoder(open_clip, model):
    """尽量拿到能 encode_text 的 tokenizer；失败返回 (None, 原因)。"""
    try:
        tok = open_clip.get_tokenizer("ViT-B-32")
        if tok is None:
            return None, "get_tokenizer 返回 None"
        return tok, "open_clip.get_tokenizer('ViT-B-32')"
    except Exception as e1:
        try:
            from open_clip.tokenizer import SimpleTokenizer
            tok = SimpleTokenizer()
            return tok, f"SimpleTokenizer 兜底（get_tokenizer 失败：{repr(e1)[:80]}）"
        except Exception as e2:
            return None, f"tokenizer 不可用：get_tokenizer={repr(e1)[:80]}；SimpleTokenizer={repr(e2)[:80]}"


def probe_attr_bridge(open_clip, model, tok, face: np.ndarray, y: np.ndarray,
                      meme: np.ndarray, meme_emotion: np.ndarray, meme_file: np.ndarray,
                      device: str) -> dict | None:
    """属性画像：每类属性用人脸/meme 各自的提示词算对比概率，再比较画像相似度。

    属性画像相似度 = 1 − 各属性概率差的均值（0=完全不像，1=一模一样）。
    """
    import torch

    prompts, meta = [], []
    for attr in ATTRS:
        name, zh = attr[0], attr[1]
        pf, nf, pm, nm = attr[2], attr[3], attr[4], attr[5]
        prompts += [pf, nf, pm, nm]
        meta.append((name, zh))
    with torch.no_grad():
        tokens = tok(prompts).to(device)
        text_feat = model.encode_text(tokens).float()
    text_feat = l2norm(text_feat.cpu().numpy())
    logit_scale = float(model.logit_scale.detach().exp().clamp(max=100.0)) if hasattr(model, "logit_scale") else 100.0

    def profiles(img_feat: np.ndarray, face_domain: bool) -> np.ndarray:
        """img_feat (N,512) → (N, A) 每属性"正提示词"概率。"""
        cols = []
        for a, attr in enumerate(ATTRS):
            base = 4 * a + (0 if face_domain else 2)
            pair = text_feat[[base, base + 1]]                 # (2,512)
            logits = logit_scale * (img_feat @ pair.T)          # (N,2)
            logits = logits - logits.max(axis=1, keepdims=True)
            p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
            cols.append(p[:, 0])
        return np.stack(cols, axis=1)

    face_prof = profiles(face, True)
    meme_prof = profiles(meme, False)

    def mad_sim(face_prof_: np.ndarray, meme_prof_: np.ndarray) -> np.ndarray:
        """(N,A) vs (M,A) → (N,M) 画像相似度。"""
        sim = np.zeros((len(face_prof_), len(meme_prof_)))
        for a in range(face_prof_.shape[1]):
            sim += 1.0 - np.abs(face_prof_[:, a][:, None] - meme_prof_[None, :, a])
        return sim / face_prof_.shape[1]

    attr_sim = mad_sim(face_prof, meme_prof)
    hit = (meme_emotion[attr_sim.argmax(axis=1)] == y)
    # 同情绪 vs 不同情绪的画像相似度
    same_mask = (np.array([CLASS_NAMES[v] for v in meme_emotion])[None, :] ==
                 np.array([CLASS_NAMES[v] for v in y])[:, None])
    same_mean = float(attr_sim[same_mask].mean())
    diff_mean = float(attr_sim[~same_mask].mean())

    # 与图像相似度融合（在检索命中率上试权重）
    img_sim = cos_matrix(face, meme)
    grid = {}
    for w in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        fused = w * img_sim + (1 - w) * attr_sim
        grid[round(w, 1)] = float((meme_emotion[fused.argmax(axis=1)] == y).mean())

    df = pd.DataFrame({"file": meme_file, "emotion": [CLASS_NAMES[v] for v in meme_emotion]})
    for a, (name, zh) in enumerate(meta):
        df[f"p_{name}"] = meme_prof[:, a]
    df.to_csv(ATTR_CSV, index=False, encoding="utf-8-sig")
    print(f"  [P6] 属性桥：同情绪画像相似 {same_mean:.4f} vs 不同情绪 {diff_mean:.4f}；"
          f"纯属性检索 top1={hit.mean():.4f}")
    return {"same_emotion_attr_sim": same_mean, "diff_emotion_attr_sim": diff_mean,
            "attr_only_top1": float(hit.mean()), "fusion_grid_top1": grid,
            "csv": str(ATTR_CSV.relative_to(ROOT))}


# =============================================================================
# 9. 报告输出
# =============================================================================

def write_report(results: dict) -> None:
    """把结果写成 txt（人读）+ json（机器读）。"""
    lines = []
    add = lines.append
    add("=" * 86)
    add("建议2 B1：CLIP 跨模态'神似'探针报告")
    add("=" * 86)
    add(f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}   设备：{DEVICE}")
    add(f"数据：FER2013 验证集 {len(y_val)} 张（测试集未使用）；meme {results['meta']['n_meme']} 张（人工标签）")
    add(f"CLIP：ViT-B-32（laion2b_s34b_b79k，本地权重 {CLIP_WEIGHTS.name}）")
    add("")

    p1 = results["p1"]
    add("【P1 跨模态质心对齐】（决策门 ≥ +%.2f）" % GATE_CENTROID)
    add(f"  对角均值={p1['diag_mean']:+.4f}  非对角均值={p1['offdiag_mean']:+.4f}  "
        f"统计量={p1['stat']:+.4f}  95%CI=[{p1['ci95'][0]:+.4f}, {p1['ci95'][1]:+.4f}]  "
        f"置换 p={p1['p_value']:.4f}")
    add(f"  结论：{'✅ 通过' if p1['passed'] else '❌ 未通过'}")
    add("")
    add("  7×7 余弦相似矩阵（行=人脸真实情绪，列=meme 人工标签）：")
    add("        " + "".join(f"{c:>10s}" for c in CLASS_NAMES))
    for i, c in enumerate(CLASS_NAMES):
        add(f"  {c:>8s}" + "".join(f"{p1['matrix'][i, j]:>10.3f}" for j in range(7)))
    add("")

    rep = results["p2"]
    add("【P2 跨模态检索】（决策门 top-1 ≥ %.2f）" % GATE_RETRIEVAL)
    add(f"  CLIP top1={rep['top1']:.4f}  top3={rep['top3']:.4f}  top5={rep['top5']:.4f}  "
        f"精确随机基线={rep['chance']:.4f}  提升={rep['lift']:+.4f}")
    add(f"  95%CI(top1)=[{rep['ci95_top1'][0]:.4f}, {rep['ci95_top1'][1]:.4f}]  "
        f"置换 p={rep['p_value']:.4f}  中心化后 top1={rep['top1_centered']:.4f}")
    add(f"  结论：{'✅ 通过' if rep['passed'] else '❌ 未通过'}")
    add("")

    zs = results["p3"]
    add("【P3 零样本软标签】（决策门一致率 ≥ %.2f；历史对照：FER 模型自动打标 22.4%%）" % GATE_ZEROSHOT)
    add(f"  一致率={zs['agreement']:.4f}  95%CI=[{zs['ci95'][0]:.4f}, {zs['ci95'][1]:.4f}]  "
        f"多数类基线={zs['majority_baseline']:.4f}  均匀基线={zs['uniform_baseline']:.4f}  "
        f"置换 p={zs['p_value']:.4f}")
    add(f"  结论：{'✅ 通过' if zs['passed'] else '❌ 未通过'}")
    add("")
    add("  混淆矩阵（行=人工标签，列=CLIP 最近原型预测）：")
    add("        " + "".join(f"{c:>10s}" for c in CLASS_NAMES))
    for i, c in enumerate(CLASS_NAMES):
        add(f"  {c:>8s}" + "".join(f"{zs['confusion'][i, j]:>10d}" for j in range(7)))
    add("")

    if results.get("p4"):
        p4 = results["p4"]
        add("【P4 对照组：ImageNet ResNet18（同协议，不设门槛）】")
        add(f"  质心对齐={p4['centroid_stat']:+.4f}（p={p4['centroid_p']:.4f}）  "
            f"检索 top1={p4['top1']:.4f} / top3={p4['top3']:.4f}（随机基线 {p4['chance']:.4f}，p={p4['p_value']:.4f}）")
        add("")

    p5 = results["p5"]
    add("【P5 同类内视觉重排（神似模式原型）】")
    add(f"  类内首选平均 CLIP 相似度={p5['mean_img_sim']:.4f}（随机同类≈{p5['random_same_class_sim']:.4f}）")
    add(f"  用到不同 meme：{p5['unique_memes']}/{p5['total_memes']} 张；最大单图占比={p5['max_single_share']:.1%}；"
        f"按验证集顺序连续重复={p5['consecutive_repeats']} 次")
    add(f"  明细：{p5['csv']}")
    add("")

    p6 = results.get("p6")
    if p6:
        add("【P6 属性桥（眼神/嘴型/气质）】")
        add(f"  同情绪属性画像相似度={p6['same_emotion_attr_sim']:.4f}  vs  "
            f"不同情绪={p6['diff_emotion_attr_sim']:.4f}")
        add(f"  纯属性检索 top1={p6['attr_only_top1']:.4f}；图像+属性融合权重扫描 top1："
            + "  ".join(f"w={w}:{v:.3f}" for w, v in p6["fusion_grid_top1"].items()))
        add(f"  明细：{p6['csv']}")
        add("")

    passed = [results["p1"]["passed"], results["p2"]["passed"], results["p3"]["passed"]]
    add("【总结论】")
    add(f"  P1={passed[0]}  P2={passed[1]}  P3={passed[2]}  →  "
        f"{'✅ 三门全过，建议进入 B2' if all(passed) else '❌ 未全部通过，按预案留负结论/降级'}")
    if not all(passed):
        add("  说明：交接文档规定探针未通过则终止 CLIP 主匹配融合路线，负结论写入报告，不安装/不交付 CLIP 权重。")
    add("")
    add("=" * 86)

    text = "\n".join(lines)
    REPORT_TXT.write_text(text, encoding="utf-8")
    REPORT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=_json_default),
                           encoding="utf-8")
    print(text)
    print(f"\n已写入：{REPORT_TXT.relative_to(ROOT)} / {REPORT_JSON.relative_to(ROOT)}")


def _json_default(o):
    """json 序列化 numpy 类型。"""
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"不可序列化：{type(o)}")


# =============================================================================
# 10. 主流程
# =============================================================================

DEVICE = "cuda"
y_val: np.ndarray | None = None


def main() -> None:
    parser = argparse.ArgumentParser(description="建议2 B1：CLIP 跨模态神似探针")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--n-perm", type=int, default=10000)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--recompute", action="store_true", help="强制重算 CLIP 特征")
    parser.add_argument("--skip-attr", action="store_true", help="跳过 P6 属性桥")
    parser.add_argument("--skip-control", action="store_true", help="跳过 P4 对照")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    global DEVICE, y_val
    DEVICE = args.device
    import torch
    if DEVICE == "cuda" and not torch.cuda.is_available():
        DEVICE = "cpu"
        print("CUDA 不可用，退回 CPU（会慢一些）")
    print(f"设备：{DEVICE}")

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n[1/6] 准备数据与 CLIP 特征 ...")
    feats = build_clip_features(DEVICE, args.batch, args.recompute)
    face, meme = feats["face"], feats["meme"]
    meme_emotion, meme_file = feats["meme_emotion"], feats["meme_file"]
    y_val = feats["y"]
    p_tta, y_p = load_val_probs()
    assert np.array_equal(y_p, y_val), "prob 缓存的 y_val 与 FER npz 不一致，请检查数据划分"
    print(f"  CLIP 特征：人脸 {face.shape}，meme {meme.shape}")

    results = {"meta": {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "device": DEVICE,
                        "n_val": int(len(y_val)), "n_meme": int(len(meme_emotion)),
                        "clip": "ViT-B-32 laion2b_s34b_b79k", "gates": {
                            "centroid": GATE_CENTROID, "retrieval": GATE_RETRIEVAL,
                            "zeroshot": GATE_ZEROSHOT, "control_zeroshot": CONTROL_ZEROSHOT}}}

    print("\n[2/6] P1 跨模态质心对齐 ...")
    results["p1"] = probe_centroid(face, y_val, meme, meme_emotion, args.n_perm, args.n_boot)
    print(f"  统计量={results['p1']['stat']:+.4f}  门槛={GATE_CENTROID}  "
          f"p={results['p1']['p_value']:.4f}  → {'通过' if results['p1']['passed'] else '未通过'}")

    print("\n[3/6] P2 跨模态检索 ...")
    results["p2"] = probe_retrieval(face, y_val, meme, meme_emotion, args.n_perm, args.n_boot)
    print(f"  top1={results['p2']['top1']:.4f}（随机 {results['p2']['chance']:.4f}）  "
          f"p={results['p2']['p_value']:.4f}  → {'通过' if results['p2']['passed'] else '未通过'}")

    print("\n[4/6] P3 零样本软标签 ...")
    results["p3"] = probe_zeroshot(face, y_val, meme, meme_emotion, args.n_perm, args.n_boot)
    print(f"  一致率={results['p3']['agreement']:.4f}  门槛={GATE_ZEROSHOT}  "
          f"p={results['p3']['p_value']:.4f}  → {'通过' if results['p3']['passed'] else '未通过'}")

    if not args.skip_control:
        print("\n[5/6] P4 对照组 ImageNet ResNet18 ...")
        results["p4"] = probe_control()
        print(f"  质心={results['p4']['centroid_stat']:+.4f}  top1={results['p4']['top1']:.4f}")

    print("\n[6/6] P5 同类内视觉重排 / P6 属性桥 ...")
    results["p5"] = probe_rerank(face, meme, meme_emotion, meme_file, p_tta)

    if not args.skip_attr:
        open_clip, model, _ = load_clip(DEVICE)   # 权重已缓存，二次加载很快
        tok, why = load_text_encoder(open_clip, model)
        print(f"  tokenizer：{why}")
        if tok is not None:
            try:
                results["p6"] = probe_attr_bridge(open_clip, model, tok, face, y_val,
                                                  meme, meme_emotion, meme_file, DEVICE)
            except Exception as e:
                print(f"  P6 属性桥失败（不影响 P1~P5）：{repr(e)[:200]}")
        else:
            print("  跳过 P6（无文本编码器）")

    write_report(results)


if __name__ == "__main__":
    main()
