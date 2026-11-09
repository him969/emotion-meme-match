# -*- coding: utf-8 -*-
r"""
11_visual_diversity.py —— 建议 3：1 主推 + 2 视觉备选的共享模块

本模块给 07_demo_app.py 提供"视觉多样性备选"能力，保持 07 的 UI 代码干净。
**不修改模型、不修改数据**，只做：
  ① 提取 58 张 meme 的两套视觉特征（ImageNet ResNet18 / FER 微调 ResNet18），进程内缓存；
  ② 按"备选范围"构造候选池（仅同情绪类 / 同类为主 / 自由补充）；
  ③ 用"距离分位带 + 贪心 max-min"挑 2 张视觉上尽量不同的备选；
     为什么不用纯最大距离：离线验证发现纯最大距离会挑到"只有一团绒毛纹理"这种怪图，
     分位带（60%~95%）只损失约 0.02 多样性，但目检明显更稳；
  ④ "换一批"：枚举主推固定下的合法备选对，保留多样性分 >= 最优 90% 的对，随机抽没出现过的；
  ⑤ 提供余弦距离，供页面显示"视觉差异分"。

依赖：timm 与 ImageNet 预训练权重在项目内 `_deps\`（和 08_transfer_train.py 共用），
       FER 特征直接用当前主模型（07 已经加载，传入 fer_model 即可，不用重复加载）。

用法（07 里）：
    vd = load_module(...)
    feats = vd.get_meme_features("imgnet", memes, device)       # (58,512) L2 归一化
    main_idx = memes.index[memes["file"] == picked["file"]][0]
    main_emotion = CLASS_NAMES.index(picked["emotion"])
    pool = vd.candidate_pool(label_idx, main_idx, main_emotion, "same")
    alt1, alt2 = vd.greedy_diverse_pair(main_idx, pool, feats, label_idx, main_emotion, "same")
    d = vd.pair_distances(main_idx, alt1, alt2, feats)
"""
import pathlib
import sys

import numpy as np
import torch
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEPS_DIR = ROOT / "_deps"
MODEL_DIR = ROOT / "模型"
if str(DEPS_DIR) not in sys.path:
    sys.path.insert(0, str(DEPS_DIR))

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DEFAULT_WEIGHTS = MODEL_DIR / "best_model.pth"

# GIF 多帧视觉特征（B/C 方案）的离线资产（17_meme_gif_frames.py 生成）
GIF_FRAMES_DIR = ROOT / "数据" / "meme_gif_frames"
GIF_PEAK_DIR = ROOT / "数据" / "meme_anim_peak"
GIF_FEATURE_MODE_LABELS = {
    "first": "首帧（默认，原行为）",
    "mean": "B：多帧时长加权平均",
    "peak": "C：峰值帧（视觉变化最大帧）",
}

# 与 07_demo_app.py / 09 / 10 保持一致的情绪相似度矩阵
EMOTION_SIM = np.array([
    [1.00, 0.80, 0.50, 0.05, 0.50, 0.20, 0.25],
    [0.80, 1.00, 0.30, 0.02, 0.45, 0.10, 0.20],
    [0.50, 0.30, 1.00, 0.05, 0.40, 0.80, 0.20],
    [0.05, 0.02, 0.05, 1.00, 0.05, 0.60, 0.20],
    [0.50, 0.45, 0.40, 0.05, 1.00, 0.10, 0.60],
    [0.20, 0.10, 0.80, 0.60, 0.10, 1.00, 0.30],
    [0.25, 0.20, 0.20, 0.20, 0.60, 0.30, 1.00],
], dtype=np.float64)

# 备选范围/视觉特征的中文名（UI 与日志共用）
SCOPE_LABELS = {
    "same": "仅同情绪类",
    "majority": "同类为主（至少 1 张同类）",
    "free": "自由补充（可来自相似情绪）",
}
FEATURE_LABELS = {
    "imgnet": "ImageNet 特征（看外观差异）",
    "fer": "FER 特征（偏表情特征）",
}

# 进程内缓存：{(kind, 图片文件列表的哈希, gif_mode, device): 特征矩阵}
_FEATURE_CACHE: dict = {}
_IMGNET_MODELS: dict = {}          # 按设备缓存 ImageNet ResNet18（CPU 兜底时不能复用 GPU 模型）


def release_feature_cache() -> None:
    """清空 meme 视觉特征缓存与 ImageNet 模型缓存（页面"释放显存缓存"按钮用）。"""
    _FEATURE_CACHE.clear()
    _IMGNET_MODELS.clear()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# ---------------- 1. meme 图像 → 张量 / 特征 ----------------

def _pil_to_tensor(im: Image.Image, size: int = 128) -> torch.Tensor:
    """任意 PIL 图 → 居中裁方 + 缩放到 size + ImageNet 归一化的张量（与 build_meme_tensor 同口径）。"""
    im = im.convert("RGB")
    w, h = im.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    im = im.crop((left, top, left + side, top + side)).resize((size, size), Image.Resampling.LANCZOS)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.array(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1)


def build_meme_tensor(memes) -> torch.Tensor:
    """把 58 张处理后的 meme 图（256×256 png）变成 (58,3,128,128) 的 ImageNet 归一化张量。"""
    return torch.stack([_pil_to_tensor(Image.open(p)) for p in memes["path"]], dim=0)


@torch.no_grad()
def _extract(imgnet_or_fer: str, meme_tensors: torch.Tensor, device, fer_model=None) -> np.ndarray:
    """提取 512 维全局特征并做 L2 归一化；返回 (N,512) numpy。"""
    if imgnet_or_fer == "fer":
        if fer_model is None:
            raise RuntimeError("FER 特征需要传入已加载的主模型 fer_model")
        model = fer_model
    elif imgnet_or_fer == "imgnet":
        global _IMGNET_MODELS
        dev_key = str(torch.device(device))
        if dev_key not in _IMGNET_MODELS:
            import timm
            from safetensors.torch import load_file

            m = timm.create_model("resnet18", pretrained=False, num_classes=0)
            sd = load_file(str(DEPS_DIR / "weights" / "resnet18.a1_in1k.safetensors"))
            state = {k: v for k, v in sd.items() if not k.startswith("fc.")}
            m.load_state_dict(state, strict=False)
            _IMGNET_MODELS[dev_key] = m.to(device).eval()
        model = _IMGNET_MODELS[dev_key]
    else:
        raise ValueError(f"未知特征类型：{imgnet_or_fer}")

    feats = model.forward_features(meme_tensors.to(device))
    feats = model.global_pool(feats)
    feats = torch.flatten(feats, 1).float().cpu().numpy()
    return feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)


def _gif_feature_rows(kind: str, gif_memes, device, fer_model, mode: str) -> dict:
    """为 GIF 表情包计算 B(mean)/C(peak) 的视觉特征行，返回 {file: (D,) L2 归一化特征}。

    资产来自 17_meme_gif_frames.py：数据\\meme_gif_frames\\<stem>.npz（帧+时长权重）
    与 数据\\meme_anim_peak\\<stem>.png（峰值帧）。资产缺失的 GIF 保持首帧特征（静默回退）。
    """
    out: dict = {}
    if mode == "peak":
        files, tensors = [], []
        for _, row in gif_memes.iterrows():
            stem = pathlib.Path(str(row["file"])).stem
            p = GIF_PEAK_DIR / f"{stem}.png"
            if p.exists():
                files.append(str(row["file"]))
                tensors.append(_pil_to_tensor(Image.open(p)))
        if tensors:
            feats = _extract(kind, torch.stack(tensors, dim=0), device, fer_model=fer_model)
            out.update({f: feats[i] for i, f in enumerate(files)})
        return out

    # mode == "mean"：多帧特征按时长权重加权平均
    for _, row in gif_memes.iterrows():
        stem = pathlib.Path(str(row["file"])).stem
        npz_path = GIF_FRAMES_DIR / f"{stem}.npz"
        if not npz_path.exists():
            continue
        with np.load(npz_path) as z:
            frames = z["frames"]
            weights = z["weights"].astype(np.float32)
        if len(frames) == 0:
            continue
        tensors = [_pil_to_tensor(Image.fromarray(fr, mode="RGB")) for fr in frames]
        feats = _extract(kind, torch.stack(tensors, dim=0), device, fer_model=fer_model)
        w = weights / (weights.sum() + 1e-9)
        avg = (feats * w[:, None]).sum(axis=0)
        avg = avg / (np.linalg.norm(avg) + 1e-9)
        out[str(row["file"])] = avg.astype(np.float32)
    return out


def get_meme_features(kind: str, memes, device, fer_model=None, gif_mode: str = "first") -> np.ndarray:
    """带缓存的 meme 特征矩阵（58×D，L2 归一化）。

    kind = "imgnet"（默认，衡量外观差异）或 "fer"（衡量表情特征差异）。
    gif_mode 只影响 GIF 表情包（18 张）的视觉特征来源：
      "first"（默认，原行为）：用预处理后的静态首帧；
      "mean"（B）：多帧 + 时长加权平均（视频级表征）；
      "peak"（C）：峰值帧（与首帧视觉差异最大的采样帧）。
    非 GIF 的 40 张任何模式下都不变；主匹配不依赖这些特征。
    """
    kind = "imgnet" if kind not in ("imgnet", "fer") else kind
    gif_mode = gif_mode if gif_mode in GIF_FEATURE_MODE_LABELS else "first"
    dev_key = str(torch.device(device))
    key = (kind, tuple(memes["file"].tolist()), gif_mode, dev_key)
    if key in _FEATURE_CACHE:
        return _FEATURE_CACHE[key]
    tensors = build_meme_tensor(memes)
    feats = _extract(kind, tensors, device, fer_model=fer_model)
    if gif_mode != "first" and "is_gif" in memes.columns:
        gif_memes = memes[memes["is_gif"] == 1]
        if len(gif_memes):
            repl = _gif_feature_rows(kind, gif_memes, device, fer_model, gif_mode)
            if repl:
                idx = {str(f): i for i, f in enumerate(memes["file"])}
                n_repl = 0
                for f, v in repl.items():
                    i = idx.get(str(f))
                    if i is not None and len(v) == feats.shape[1]:
                        feats[i] = v
                        n_repl += 1
                print(f"  [GIF 视觉特征] kind={kind} mode={gif_mode}：替换 {n_repl}/{len(gif_memes)} 张 GIF 特征")
    _FEATURE_CACHE[key] = feats
    return feats


# ---------------- 2. 候选池 / 相似情绪 ----------------

def label_indices(memes) -> np.ndarray:
    """meme 表 → 每张的情绪下标数组。"""
    return memes["emotion"].map({e: i for i, e in enumerate(CLASS_NAMES)}).to_numpy()


def similar_emotions(main_emotion: int, k: int = 2) -> list[int]:
    """按相似度矩阵找出与主情绪最相似的 k 个其他情绪下标（默认 2 个）。"""
    row = EMOTION_SIM[main_emotion].copy()
    row[main_emotion] = -1
    return [int(i) for i in np.argsort(-row)[:k]]


def candidate_pool(label_idx: np.ndarray, main_idx: int, main_emotion: int, scope: str) -> np.ndarray:
    """返回候选 meme 下标（不含主推）。

    scope = "same"     只取同情绪类；
            "majority" / "free"  同类 + 相似度最高的 2 个相邻情绪
            （两者的区别在后续约束：majority 要求至少 1 张同类，free 不要求）。
    """
    n = len(label_idx)
    all_idx = np.arange(n)
    same = np.flatnonzero((label_idx == main_emotion) & (all_idx != main_idx))
    if scope == "same":
        return same
    adj = similar_emotions(main_emotion, k=2)
    extra = np.flatnonzero(np.isin(label_idx, adj) & (all_idx != main_idx))
    return np.concatenate([same, extra])


# ---------------- 3. 距离与备选选择 ----------------

def _band_pick(values: np.ndarray, band: tuple[float, float] = (60, 95)) -> int:
    """在 values 的 [band] 分位带里，取最靠近上界的那个（避免挑到最极端值）。"""
    if len(values) == 0:
        return -1
    if len(values) == 1:
        return 0
    lo, hi = np.percentile(values, band[0]), np.percentile(values, band[1])
    inband = np.flatnonzero((values >= lo) & (values <= hi))
    if len(inband) == 0:
        inband = np.arange(len(values))
    return int(inband[np.argmin(np.abs(values[inband] - hi))])


def _majority_fix(main_idx: int, alt1: int, alt2: int, feat: np.ndarray,
                  label_idx: np.ndarray, main_emotion: int):
    """majority 约束：两张备选都不同类时，把第二张换成"最优同类候选"（保持至少 1 张同类）。"""
    if label_idx[alt1] == main_emotion or label_idx[alt2] == main_emotion:
        return alt1, alt2
    same = np.flatnonzero((label_idx == main_emotion) & (np.arange(len(label_idx)) != main_idx))
    same = same[same != alt1]
    if len(same) == 0:
        return alt1, alt2
    d_main = 1.0 - feat[same] @ feat[main_idx]
    d_alt1 = 1.0 - feat[same] @ feat[alt1]
    alt2 = int(same[int(np.argmax(np.minimum(d_main, d_alt1)))])
    return alt1, alt2


def greedy_diverse_pair(main_idx: int, pool: np.ndarray, feat: np.ndarray,
                        label_idx: np.ndarray, main_emotion: int, scope: str,
                        band: tuple[float, float] = (60, 95)) -> tuple[int, int]:
    """贪心 max-min + 距离分位带，挑 2 张备选（返回下标，可能第二张为 -1）。

    ① 备选1：主推到候选池中距离落在 [60%,95%] 分位带、且最靠近上界的候选；
    ② 备选2：到（主推、备选1）的最小距离同样落在分位带里，取最靠近上界的；
    ③ majority 模式再保证至少 1 张同类。
    """
    if len(pool) == 0:
        return -1, -1
    d_main = 1.0 - feat[pool] @ feat[main_idx]
    alt1 = int(pool[_band_pick(d_main, band)])
    rest = pool[pool != alt1]
    if len(rest) == 0:
        return alt1, -1
    pos = np.flatnonzero(pool != alt1)
    d_alt1 = 1.0 - feat[rest] @ feat[alt1]
    score2 = np.minimum(d_main[pos], d_alt1)
    alt2 = int(rest[_band_pick(score2, band)])
    if scope == "majority":
        alt1, alt2 = _majority_fix(main_idx, alt1, alt2, feat, label_idx, main_emotion)
    return alt1, alt2


def pair_distances(main_idx: int, alt1: int, alt2: int, feat: np.ndarray) -> dict:
    """三张图的余弦距离（1 - cos，0=一样，2=相反），供页面显示。"""
    def d(i, j):
        return float(1.0 - feat[i] @ feat[j])

    return {"main_alt1": d(main_idx, alt1),
            "main_alt2": d(main_idx, alt2),
            "alt1_alt2": d(alt1, alt2)}


def pair_score(main_idx: int, a: int, b: int, feat: np.ndarray) -> float:
    """一对备选的多样性分：三张两两距离的最小值（越大越不重样）。"""
    dd = pair_distances(main_idx, a, b, feat)
    return min(dd.values())


def legal_pairs(main_idx: int, pool: np.ndarray, feat: np.ndarray,
                label_idx: np.ndarray, main_emotion: int, scope: str) -> list[tuple[int, int, float]]:
    """枚举主推固定下的合法备选对（scope 约束下），返回 [(a,b,多样性分), ...]。"""
    import itertools

    out = []
    for a, b in itertools.combinations(pool, 2):
        if scope == "majority" and label_idx[a] != main_emotion and label_idx[b] != main_emotion:
            continue
        out.append((int(a), int(b), pair_score(main_idx, int(a), int(b), feat)))
    return out


def reroll_pair(base: tuple[int, int], main_idx: int, pool: np.ndarray, feat: np.ndarray,
                label_idx: np.ndarray, main_emotion: int, scope: str,
                recent: list[frozenset], rng, keep_ratio: float = 0.9) -> tuple[int, int]:
    """换一批：在"多样性分 >= 最优 90%"的合法备选对里随机抽一个没出现过的。

    base = 当前显示的备选对；recent = 最近显示过的备选对集合（避免连续重复）。
    """
    pairs = legal_pairs(main_idx, pool, feat, label_idx, main_emotion, scope)
    if not pairs:
        return base
    best = max(p[2] for p in pairs)
    pool_top = [(a, b) for a, b, s in pairs if s >= keep_ratio * best]
    used = set(recent) | {frozenset(base)}
    fresh = [(a, b) for a, b in pool_top if frozenset((a, b)) not in used]
    cand = fresh if fresh else [p for p in pool_top if frozenset(p) != frozenset(base)] or pool_top
    return cand[rng.randrange(len(cand))]
