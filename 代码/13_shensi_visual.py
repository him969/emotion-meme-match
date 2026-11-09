# -*- coding: utf-8 -*-
r"""
13_shensi_visual.py —— "神似模式"视觉模块（建议2 落地版）

定位
----
在**情感主判据（现有软匹配/迁移模型）不变**的前提下，给"同类候选里挑哪一张"补一层
视觉"神似"打分。默认关闭，只有 07_demo_app.py 打开"神似模式"开关时才调用。

打分由三路视觉信号加权（都在情绪类内比较，不改变选中情绪 → 情绪命中率不变）：
  ① 神态属性桥（CLIP 文本属性画像）：把"眼睛/嘴/眉毛/气质"拆成 16 个属性
     （张嘴、露齿、吐舌、瞪眼、眯眼、闭眼、皱眉、微笑、歪头、高冷、沙雕、可爱、
     委屈、震惊、得意、特写），人脸与 meme 各自算属性画像，比"画像相似度"。
     —— B1 探针实测：纯属性检索 top1=0.349（随机 0.163，p=0.0015），是唯一有真信号
        的跨域视觉通道；CLIP 图像整体相似度只有随机水平（P2 未过门）。
  ② 关键点几何（mediapipe 478 点 / ONNX 备份）：从人脸量出嘴部开合度、眼睛睁开度、
     嘴角上扬、眉眼距、头部倾斜，再把 meme 的"人工标注神态"（闭嘴/张嘴/张大嘴/吐舌、
     睁眼/瞪眼/眯眼/闭眼…）映射成同一尺度，比"神态动作相似度"。
  ③ CLIP 图像相似度：整体外观/构图相似（权重最低，仅作补充）。

依赖（都在项目内 _deps\，不污染全局环境）
----------------------------------------
  - open_clip + laion CLIP ViT-B/32 权重：_deps\weights\open_clip_vitb32_laion2b.bin
  - 人脸关键点：mediapipe（_deps\mediapipe + _deps\cv2 + _deps\absl + _deps\sounddevice）
    模型 _deps\models\face_landmarker.task；mediapipe 不可用时自动退回
    onnxruntime + _deps\models\face_landmarks_478.onnx（只对"人脸占满画面"的图可靠）
  - meme 气质/部位人工标签：数据\shensi_meme_tags.json（58 张）

缓存
----
  模型\shensi_meme_cache.npz：58 张 meme 的 CLIP 特征 + 属性画像（首次约 2~4 秒，之后 <0.1 秒）
  模型\shensi_geometry_stats.json：验证集几何分位数（由 14_shensi_probe.py 生成；缺省用内置经验值）
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import json
import math
import pathlib
import sys

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEPS_DIR = ROOT / "_deps"
if str(DEPS_DIR) not in sys.path:
    sys.path.insert(0, str(DEPS_DIR))

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
DATA_DIR = ROOT / "数据"
MODEL_DIR = ROOT / "模型"
CLIP_WEIGHTS = DEPS_DIR / "weights" / "open_clip_vitb32_laion2b.bin"
LANDMARK_TASK = DEPS_DIR / "models" / "face_landmarker.task"
LANDMARK_ONNX = DEPS_DIR / "models" / "face_landmarks_478.onnx"
MEME_CSV = DATA_DIR / "meme_index.csv"
MEME_TAGS_JSON = DATA_DIR / "shensi_meme_tags.json"
MEME_CACHE_NPZ = MODEL_DIR / "shensi_meme_cache.npz"
GEOM_STATS_JSON = MODEL_DIR / "shensi_geometry_stats.json"
# GIF 多帧视觉特征（B/C 方案）资产（17_meme_gif_frames.py 生成）
GIF_FRAMES_DIR = DATA_DIR / "meme_gif_frames"
GIF_PEAK_DIR = DATA_DIR / "meme_anim_peak"

# ---------------- 神态属性（与 12_clip_probe.py / p6_diag.py 保持一致） ----------------
# (属性名, 中文, 是否"表情属性"(权重×2), 人脸正/负提示词, meme 正/负提示词)
ATTRS = [
    ("mouth_open", "张嘴", True, "a photo of a person whose mouth is open", "a photo of a person whose mouth is closed",
     "an image of a cat whose mouth is open", "an image of a cat whose mouth is closed"),
    ("teeth", "露齿", True, "a photo of a person showing their teeth", "a photo of a person with lips closed",
     "an image of a cat showing teeth", "an image of a cat with mouth closed"),
    ("tongue", "吐舌", True, "a photo of a person with their tongue out", "a photo of a person with tongue inside",
     "an image of a cat with its tongue out", "an image of a cat with tongue inside"),
    ("eyes_wide", "瞪眼", True, "a photo of a person with wide open eyes", "a photo of a person with relaxed eyes",
     "an image of a cat with wide open eyes", "an image of a cat with relaxed eyes"),
    ("eyes_narrow", "眯眼", True, "a photo of a person with narrowed squinting eyes", "a photo of a person with wide eyes",
     "an image of a cat with narrowed squinting eyes", "an image of a cat with wide eyes"),
    ("eyes_closed", "闭眼", True, "a photo of a person with closed eyes", "a photo of a person with open eyes",
     "an image of a cat with closed eyes", "an image of a cat with open eyes"),
    ("brows_furrowed", "皱眉", True, "a photo of a person with furrowed angry brows", "a photo of a person with relaxed brows",
     "an image of a cat with angry furrowed brows", "an image of a cat with relaxed brows"),
    ("smile", "微笑", True, "a photo of a smiling person", "a photo of a non-smiling person",
     "an image of a smiling cat", "an image of a non-smiling cat"),
    ("head_tilt", "歪头", False, "a photo of a person tilting their head", "a photo of a person looking straight",
     "an image of a cat tilting its head", "an image of a cat looking straight"),
    ("grumpy", "高冷嫌弃", False, "a photo of a grumpy disdainful person", "a photo of a cheerful friendly person",
     "an image of a grumpy disdainful cat", "an image of a cheerful friendly cat"),
    ("silly", "沙雕搞怪", False, "a photo of a silly goofy person", "a photo of a serious calm person",
     "an image of a silly goofy cat", "an image of a serious calm cat"),
    ("cute", "可爱卖萌", False, "a photo of a cute endearing person", "a photo of a plain ordinary person",
     "an image of a cute endearing cat", "an image of a plain ordinary cat"),
    ("pitiful", "委屈可怜", False, "a photo of a sad pitiful teary person", "a photo of a happy confident person",
     "an image of a sad pitiful teary cat", "an image of a happy confident cat"),
    ("shocked", "震惊", False, "a photo of a shocked astonished person", "a photo of an unsurprised person",
     "an image of a shocked astonished cat", "an image of an unsurprised cat"),
    ("smug", "得意", False, "a photo of a smug pleased person", "a photo of a humble modest person",
     "an image of a smug pleased cat", "an image of a humble modest cat"),
    ("closeup", "特写", False, "a close-up photo of a face", "a wide shot showing a whole body",
     "a close-up image of a cat's face", "a wide shot showing a whole cat body"),
]
ATTR_NAMES = [a[0] for a in ATTRS]
ATTR_ZH = {a[0]: a[1] for a in ATTRS}
ATTR_W = np.array([2.0 if a[2] else 1.0 for a in ATTRS], dtype=np.float64)

# ---------------- meme 人工标签 → 几何目标值 ----------------
# 眼睛睁开程度（0=闭眼 1=瞪圆）
EYE_LEVEL = {"闭眼": 0.05, "眯眼": 0.20, "半睁": 0.35, "斜视": 0.32, "墨镜": 0.50,
             "睁眼": 0.60, "含泪": 0.68, "瞪眼": 0.90, "发光眼": 0.95, "点缀": 0.60}
# 嘴部张开程度（0=闭嘴 1=张大嘴）
MOUTH_LEVEL = {"闭嘴": 0.10, "微张": 0.30, "半张": 0.40, "张嘴": 0.55, "吐舌": 0.80, "张大嘴": 0.90}


# =============================================================================
# mediapipe / ONNX 关键点
# =============================================================================

class _Landmarker:
    """人脸 478 关键点：优先 mediapipe（自带检测，任意构图），失败退回 ONNX（需人脸占满画面）。"""

    def __init__(self):
        self.kind = None
        self._mp = None
        self._lm = None
        self._sess = None
        self._in_name = None
        self.last_error = ""
        self._init_mediapipe() or self._init_onnx()

    # ---------- mediapipe ----------
    def _init_mediapipe(self) -> bool:
        try:
            if not LANDMARK_TASK.exists():
                raise FileNotFoundError(str(LANDMARK_TASK))
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
            content = LANDMARK_TASK.read_bytes()          # 用 bytes 绕开中文路径
            opts = vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_buffer=content),
                num_faces=1, running_mode=vision.RunningMode.IMAGE,
                min_face_detection_confidence=0.3,
            )
            self._mp = mp
            self._lm = vision.FaceLandmarker.create_from_options(opts)
            self.kind = "mediapipe"
            return True
        except Exception as e:  # noqa: BLE001
            self.last_error = f"mediapipe 不可用：{repr(e)[:120]}"
            return False

    # ---------- onnxruntime ----------
    def _init_onnx(self) -> bool:
        try:
            import onnxruntime as ort
            self._sess = ort.InferenceSession(str(LANDMARK_ONNX), providers=["CPUExecutionProvider"])
            self._in_name = self._sess.get_inputs()[0].name
            self.kind = "onnx"
            return True
        except Exception as e:  # noqa: BLE001
            self.last_error += f"；onnx 也不可用：{repr(e)[:120]}"
            self.kind = None
            return False

    def detect(self, img: Image.Image) -> np.ndarray | None:
        """返回 (478,2) 像素坐标（原图坐标）；无人脸返回 None。"""
        rgb = img.convert("RGB")
        if self.kind == "mediapipe":
            arr = np.asarray(rgb, dtype=np.uint8).copy()
            res = self._lm.detect(self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=arr))
            if not res.face_landmarks:
                return None
            h, w = arr.shape[:2]
            return np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], dtype=np.float64)
        if self.kind == "onnx":
            side = min(rgb.size)
            left, top = (rgb.width - side) // 2, (rgb.height - side) // 2
            crop = rgb.crop((left, top, left + side, top + side)).resize((256, 256), Image.LANCZOS)
            arr = np.asarray(crop, dtype=np.float32) / 255.0
            outs = self._sess.run(None, {self._in_name: arr[None]})
            pts, presence = None, None
            for o in outs:
                a = np.asarray(o)
                if a.size == 478 * 3:
                    pts = a.reshape(478, 3)
                elif a.size == 1:
                    presence = 1.0 / (1.0 + math.exp(-float(a.ravel()[0])))
            if pts is None or (presence is not None and presence < 0.5):
                return None
            return pts[:, :2].copy()
        return None


_LM_SINGLETON = None


def get_landmarker() -> _Landmarker | None:
    global _LM_SINGLETON
    if _LM_SINGLETON is None:
        _LM_SINGLETON = _Landmarker()
    return _LM_SINGLETON if _LM_SINGLETON.kind else None


# =============================================================================
# 人脸几何特征（478 点 → 可解释的"神态"数值）
# =============================================================================

# MediaPipe FaceMesh 478 点语义索引
I_LEFT_EYE = (33, 133, 159, 145)      # 左眼外角/内角/上眼睑/下眼睑
I_RIGHT_EYE = (263, 362, 386, 374)    # 右眼
I_MOUTH = (61, 291, 13, 14)           # 左嘴角/右嘴角/上唇中/下唇中
I_BROW = (70, 300)                    # 左/右眉中点附近


def face_geometry(img: Image.Image) -> dict | None:
    """量出人脸的神态几何量；检测不到脸返回 None。"""
    lm = get_landmarker()
    if lm is None:
        return None
    # 小图（如 FER 48×48）先放大，否则人脸检测器可能收不到脸
    if min(img.size) < 160:
        scale = 256.0 / min(img.size)
        img = img.resize((max(64, int(img.width * scale)), max(64, int(img.height * scale))),
                         Image.Resampling.LANCZOS)
    pts = lm.detect(img)
    if pts is None:
        return None
    face_w = float(np.linalg.norm(pts[I_LEFT_EYE[0]] - pts[I_RIGHT_EYE[0]])) + 1e-6
    mouth_open = float(np.linalg.norm(pts[I_MOUTH[2]] - pts[I_MOUTH[3]])) / face_w
    mouth_width = float(np.linalg.norm(pts[I_MOUTH[0]] - pts[I_MOUTH[1]])) / face_w
    mouth_aspect = mouth_open / (mouth_width + 1e-6)
    eye_l = float(np.linalg.norm(pts[I_LEFT_EYE[2]] - pts[I_LEFT_EYE[3]])) / (
        float(np.linalg.norm(pts[I_LEFT_EYE[0]] - pts[I_LEFT_EYE[1]])) + 1e-6)
    eye_r = float(np.linalg.norm(pts[I_RIGHT_EYE[2]] - pts[I_RIGHT_EYE[3]])) / (
        float(np.linalg.norm(pts[I_RIGHT_EYE[0]] - pts[I_RIGHT_EYE[1]])) + 1e-6)
    eye_open = (eye_l + eye_r) / 2.0
    corner_lift = float(pts[I_MOUTH[2]][1] - (pts[I_MOUTH[0]][1] + pts[I_MOUTH[1]][1]) / 2.0) / face_w
    brow_eye = float((pts[I_BROW[0]][1] + pts[I_BROW[1]][1]) / 2.0
                     - (pts[I_LEFT_EYE[2]][1] + pts[I_RIGHT_EYE[2]][1]) / 2.0) / face_w
    tilt = math.degrees(math.atan2(float(pts[I_RIGHT_EYE[0]][1] - pts[I_LEFT_EYE[0]][1]),
                                   float(pts[I_RIGHT_EYE[0]][0] - pts[I_LEFT_EYE[0]][0])))
    return {"face_w_px": face_w, "mouth_open": mouth_open, "mouth_width": mouth_width,
            "mouth_aspect": mouth_aspect, "eye_open": eye_open, "corner_lift": corner_lift,
            "brow_eye": brow_eye, "head_tilt": tilt, "n_points": int(len(pts))}


# 经验回退分位（未跑 14_shensi_probe.py 生成 stats 时使用；粗略对应 FER 验证集）
GEOM_FEATURES = ["mouth_aspect", "eye_open", "corner_lift", "brow_eye"]
GEOM_FALLBACK = {   # (p5, p50, p95)
    "mouth_aspect": (0.04, 0.16, 0.55),
    "eye_open": (0.12, 0.25, 0.40),
    "corner_lift": (-0.02, 0.03, 0.09),
    "brow_eye": (0.10, 0.16, 0.24),
}


def load_geom_stats() -> dict | None:
    if GEOM_STATS_JSON.exists():
        try:
            return json.loads(GEOM_STATS_JSON.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None
    return None


def _percentile(feature: str, value: float, stats: dict | None) -> float:
    """把几何量映射到 0~1 分位（用验证集分位点内插）。"""
    if stats and feature in stats:
        qs = np.array(stats[feature]["percentiles"], dtype=np.float64)   # 101 个点，0..100
        vals = np.array(stats[feature]["values"], dtype=np.float64)
        p = float(np.interp(value, vals, qs))
        return max(0.0, min(1.0, p / 100.0))
    p5, p50, p95 = GEOM_FALLBACK[feature]
    # 用三个点做分段线性近似的分位
    if value <= p5:
        return 0.05 * max(0.0, value / (p5 + 1e-9))
    if value <= p50:
        return 0.05 + 0.45 * (value - p5) / (p50 - p5 + 1e-9)
    if value <= p95:
        return 0.50 + 0.45 * (value - p50) / (p95 - p50 + 1e-9)
    return min(1.0, 0.95 + 0.05 * (value - p95) / (p95 * 0.5 + 1e-9))


# =============================================================================
# CLIP：meme 特征 / 属性画像 / 人脸编码
# =============================================================================

_CLIP_STATE = {"model": None, "preprocess": None, "tokenizer": None, "text_feat": None}


def _l2(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)


def clip_available() -> bool:
    return CLIP_WEIGHTS.exists()


_FORCE_CPU = False


def set_force_cpu(flag: bool) -> None:
    """强制 CLIP 用 CPU（页面在 GPU 显存不足时调用）；切换设备会清空已加载的 CLIP，下次惰性重载。

    CPU 模式下神似模式仍可用：meme 侧的 CLIP/属性特征走 `模型\\shensi_meme_cache.npz` 缓存，
    只需要对当前这一张人脸做一次 CLIP 编码（约 1~2 秒），因此显存被占满时也能演示。
    """
    global _FORCE_CPU, _CLIP_STATE
    flag = bool(flag)
    if flag != _FORCE_CPU:
        _FORCE_CPU = flag
        _CLIP_STATE = {"model": None, "preprocess": None, "tokenizer": None, "text_feat": None}


def is_force_cpu() -> bool:
    return _FORCE_CPU


def release_clip() -> None:
    """释放 CLIP 模型（显存/内存），配合页面"释放显存缓存"按钮；下次使用会自动重载。"""
    global _CLIP_STATE
    _CLIP_STATE = {"model": None, "preprocess": None, "tokenizer": None, "text_feat": None}
    try:
        import gc
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except Exception:  # noqa: BLE001
        pass


def _ensure_clip():
    """首次调用才加载 CLIP（约 2 秒），失败抛 RuntimeError。"""
    if _CLIP_STATE["model"] is not None:
        return
    if not CLIP_WEIGHTS.exists():
        raise RuntimeError(f"缺少 CLIP 权重：{CLIP_WEIGHTS}")
    import open_clip
    import torch
    device = "cpu" if (_FORCE_CPU or not torch.cuda.is_available()) else "cuda"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=str(CLIP_WEIGHTS), device=device)
    model.eval()
    tok = open_clip.get_tokenizer("ViT-B-32")
    prompts = []
    for a in ATTRS:
        prompts += [a[3], a[4], a[5], a[6]]
    with torch.no_grad():
        tf = model.encode_text(tok(prompts).to(device)).float().cpu().numpy()
    _CLIP_STATE.update({"model": model, "preprocess": preprocess, "tokenizer": tok,
                        "text_feat": _l2(tf), "device": device, "torch": torch,
                        "logit_scale": float(model.logit_scale.detach().exp())})


def encode_images_clip(images: list[Image.Image]) -> np.ndarray:
    """PIL 列表 → (N,512) L2 归一化 CLIP 特征。"""
    _ensure_clip()
    torch = _CLIP_STATE["torch"]
    preprocess = _CLIP_STATE["preprocess"]
    tens = torch.stack([preprocess(im.convert("RGB")) for im in images]).to(_CLIP_STATE["device"])
    with torch.no_grad():
        feat = _CLIP_STATE["model"].encode_image(tens).float().cpu().numpy()
    return _l2(feat)


def attr_profiles(img_feats: np.ndarray, face_domain: bool) -> np.ndarray:
    """(N,512) CLIP 特征 → (N,16) 每个属性的"正提示词"概率。"""
    _ensure_clip()
    scale = _CLIP_STATE["logit_scale"]
    tf = _CLIP_STATE["text_feat"]
    cols = []
    for a in range(len(ATTRS)):
        base = 4 * a + (0 if face_domain else 2)
        logits = scale * (img_feats @ tf[[base, base + 1]].T)
        logits = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        cols.append(p[:, 0])
    return np.stack(cols, axis=1)


def attr_similarity(face_prof: np.ndarray, meme_prof: np.ndarray) -> np.ndarray:
    """(A,) 或 (N,A) 人脸画像 vs (M,A) meme 画像 → 加权差异相似度 (N,M) 或 (M,)。"""
    if face_prof.ndim == 1:
        face_prof = face_prof[None, :]
    w = ATTR_W / ATTR_W.sum()
    sim = np.zeros((len(face_prof), len(meme_prof)))
    for a in range(len(ATTRS)):
        sim += w[a] * (1.0 - np.abs(face_prof[:, a][:, None] - meme_prof[None, :, a]))
    return sim


# =============================================================================
# meme 资产：特征 + 属性画像 + 人工标签（带磁盘缓存）
# =============================================================================

def load_memes():
    """读 meme 元数据表（返回 DataFrame，含 path/stem 列）。"""
    import pandas as pd
    memes = pd.read_csv(MEME_CSV)
    memes["path"] = memes["processed"].map(lambda p: str(ROOT / pathlib.Path(p)))
    memes["stem"] = memes["file"].str.replace(r"\.[^.]+$", "", regex=True)
    return memes


def load_meme_tags() -> dict:
    """读人工神态/气质标签（58 张）；文件不存在时返回空字典（几何项退化为中性）。"""
    if not MEME_TAGS_JSON.exists():
        return {}
    raw = json.loads(MEME_TAGS_JSON.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


class MemeAssets:
    """58 张 meme 的 CLIP 特征 / 属性画像 / 标签数值化（懒加载 + 缓存）。

    gif_mode（B/C 方案）只影响 18 张 GIF 的视觉特征来源：
      "first"（默认）：静态首帧（原行为）；
      "mean"（B）：多帧时长加权平均 CLIP 特征/属性画像；
      "peak"（C）：峰值帧（与首帧视觉差异最大的采样帧）。
    非 GIF 的 40 张任何模式下都不变。
    """

    def __init__(self, gif_mode: str = "first"):
        self.memes = load_memes()
        self.tags = load_meme_tags()
        self.gif_mode = gif_mode if gif_mode in ("first", "mean", "peak") else "first"
        self.clip_feat = None
        self.attr_prof = None
        self.eye_level = None
        self.mouth_level = None
        self.ready = False
        self.error = ""
        self._load()

    def _load(self):
        try:
            need = [str(p) for p in self.memes["path"]]
            if MEME_CACHE_NPZ.exists():
                with np.load(MEME_CACHE_NPZ) as z:
                    if list(z["files"].astype(str)) == need and z["clip"].shape[0] == len(need):
                        self.clip_feat = z["clip"].astype(np.float64)
                        self.attr_prof = z["attr"].astype(np.float64)
                        self.ready = True
        except Exception:  # noqa: BLE001
            self.ready = False
        if not self.ready and clip_available():
            imgs = [Image.open(p).convert("RGB") for p in self.memes["path"]]
            self.clip_feat = encode_images_clip(imgs)
            self.attr_prof = attr_profiles(self.clip_feat, face_domain=False)
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(MEME_CACHE_NPZ, clip=self.clip_feat.astype(np.float32),
                                attr=self.attr_prof.astype(np.float32),
                                files=np.array([str(p) for p in self.memes["path"]]))
            self.ready = True
        # 人工标签 → 数值（没标签的给中性 0.5）
        eye = np.full(len(self.memes), 0.5)
        mouth = np.full(len(self.memes), 0.5)
        for i, stem in enumerate(self.memes["stem"]):
            tag = self.tags.get(stem) or {}
            eye[i] = EYE_LEVEL.get(tag.get("eyes", ""), 0.5)
            mouth[i] = MOUTH_LEVEL.get(tag.get("mouth", ""), 0.5)
        self.eye_level = eye
        self.mouth_level = mouth
        if self.gif_mode != "first" and self.ready:
            self._apply_gif_mode()

    def _apply_gif_mode(self):
        """把 GIF 行的 CLIP 特征/属性画像替换成 B(mean) 或 C(peak) 版本。"""
        if "is_gif" not in self.memes.columns:
            return
        gif_rows = self.memes[self.memes["is_gif"] == 1]
        if len(gif_rows) == 0:
            return
        idx = {str(f): i for i, f in enumerate(self.memes["file"])}
        replaced = 0
        if self.gif_mode == "peak":
            files, imgs = [], []
            for _, row in gif_rows.iterrows():
                stem = pathlib.Path(str(row["file"])).stem
                p = GIF_PEAK_DIR / f"{stem}.png"
                if p.exists():
                    files.append(str(row["file"]))
                    imgs.append(Image.open(p).convert("RGB"))
            if imgs:
                feats = encode_images_clip(imgs)
                attrs = attr_profiles(feats, face_domain=False)
                for k, f in enumerate(files):
                    i = idx.get(f)
                    if i is not None:
                        self.clip_feat[i] = feats[k]
                        self.attr_prof[i] = attrs[k]
                        replaced += 1
        else:  # mean：多帧时长加权平均
            for _, row in gif_rows.iterrows():
                stem = pathlib.Path(str(row["file"])).stem
                npz_path = GIF_FRAMES_DIR / f"{stem}.npz"
                if not npz_path.exists():
                    continue
                with np.load(npz_path) as z:
                    frames = z["frames"]
                    weights = z["weights"].astype(np.float64)
                if len(frames) == 0:
                    continue
                imgs = [Image.fromarray(fr, mode="RGB") for fr in frames]
                feats = encode_images_clip(imgs)
                attrs = attr_profiles(feats, face_domain=False)
                w = weights / (weights.sum() + 1e-9)
                avg = (feats * w[:, None]).sum(axis=0)
                avg = avg / (np.linalg.norm(avg) + 1e-9)
                attr_avg = (attrs * w[:, None]).sum(axis=0)
                i = idx.get(str(row["file"]))
                if i is not None:
                    self.clip_feat[i] = avg
                    self.attr_prof[i] = attr_avg
                    replaced += 1
        self.gif_mode_replaced = replaced


_ASSETS_CACHE: dict = {}
_DEFAULT_GIF_MODE = "first"


def set_default_gif_mode(mode: str) -> None:
    """页面切换"GIF 视觉特征"时调用：设置后续 get_assets() 使用的模式（first/mean/peak）。"""
    global _DEFAULT_GIF_MODE
    _DEFAULT_GIF_MODE = mode if mode in ("first", "mean", "peak") else "first"


def get_assets(gif_mode: str | None = None) -> MemeAssets:
    mode = gif_mode or _DEFAULT_GIF_MODE
    if mode not in _ASSETS_CACHE:
        _ASSETS_CACHE[mode] = MemeAssets(gif_mode=mode)
    return _ASSETS_CACHE[mode]


# =============================================================================
# 神似打分（类内）
# =============================================================================

# 三路视觉信号的默认权重（attr/geo/clip，和为 1）
DEFAULT_WEIGHTS = (0.50, 0.35, 0.15)


def face_visual_evidence(img: Image.Image) -> dict:
    """抽一次人脸侧的视觉证据（几何 + 属性画像 + CLIP 特征），供类内比较。"""
    assets = get_assets()
    out = {"geometry": None, "attr_prof": None, "clip_feat": None,
           "timing_ms": 0.0, "notes": []}
    import time
    t0 = time.time()
    try:
        geo = face_geometry(img)
        out["geometry"] = geo
    except Exception as e:  # noqa: BLE001
        out["notes"].append(f"几何提取失败：{repr(e)[:80]}")
    if clip_available() and assets.ready:
        try:
            feat = encode_images_clip([img])
            out["clip_feat"] = feat[0]
            out["attr_prof"] = attr_profiles(feat, face_domain=True)[0]
        except Exception as e:  # noqa: BLE001
            out["notes"].append(f"CLIP 提取失败：{repr(e)[:80]}")
    out["timing_ms"] = (time.time() - t0) * 1000
    return out


def visual_components(evidence: dict, memes) -> dict[str, np.ndarray]:
    """对**全部 58 张** meme 算三路视觉信号（0~1 绝对尺度，便于跨类加权求和）。

    返回 {"attr": (58,), "geo": (58,), "clip": (58,), "visual": (58,)}
    visual = w_attr*attr + w_geo*geo + w_clip*clip（权重 DEFAULT_WEIGHTS）。
    """
    assets = get_assets()
    n = len(memes)
    attr = np.full(n, 0.5)
    if evidence.get("attr_prof") is not None and assets.ready:
        attr = attr_similarity(evidence["attr_prof"], assets.attr_prof)[0]
    clip01 = np.full(n, 0.5)
    if evidence.get("clip_feat") is not None and assets.ready:
        cos = assets.clip_feat @ evidence["clip_feat"]
        clip01 = np.clip((cos + 1.0) / 2.0, 0.0, 1.0)
    geo = np.full(n, 0.5)
    geom = evidence.get("geometry")
    if geom is not None and assets.eye_level is not None:
        stats = load_geom_stats()
        p_mouth = _percentile("mouth_aspect", geom["mouth_aspect"], stats)
        p_eye = _percentile("eye_open", geom["eye_open"], stats)
        p_lift = _percentile("corner_lift", geom["corner_lift"], stats)
        m_target = assets.mouth_level
        e_target = assets.eye_level
        geo = 0.55 * (1.0 - np.abs(p_mouth - m_target)) + 0.35 * (1.0 - np.abs(p_eye - e_target))
        geo += 0.10 * (1.0 - np.abs(p_lift - 0.7)) * np.clip(1.0 - m_target, 0.0, 1.0)
        geo = np.clip(geo, 0.0, 1.0)
    w_attr, w_geo, w_clip = DEFAULT_WEIGHTS
    visual = w_attr * attr + w_geo * geo + w_clip * clip01
    return {"attr": attr, "geo": geo, "clip": clip01, "visual": visual}


def rank_candidates(evidence: dict, memes, candidate_positions: list[int],
                    weights: tuple[float, float, float] = DEFAULT_WEIGHTS) -> list[dict]:
    """在候选 meme（DataFrame 位置下标）里按"神似"综合分排序，返回明细列表。

    三路分量都先做类内 min-max 归一化再加权（避免量纲不同互相淹没）。
    """
    assets = get_assets()
    w_attr, w_geo, w_clip = weights
    pos = np.array(list(candidate_positions), dtype=int)
    n = len(pos)
    if n == 0:
        return []
    # ① 属性画像相似度
    attr = np.full(n, 0.5)
    if evidence.get("attr_prof") is not None and assets.ready:
        attr = attr_similarity(evidence["attr_prof"], assets.attr_prof[pos])[0]
    # ② 关键点几何 vs 人工神态标签
    geo = np.full(n, 0.5)
    geom = evidence.get("geometry")
    if geom is not None and assets.eye_level is not None:
        stats = load_geom_stats()
        p_mouth = _percentile("mouth_aspect", geom["mouth_aspect"], stats)
        p_eye = _percentile("eye_open", geom["eye_open"], stats)
        p_lift = _percentile("corner_lift", geom["corner_lift"], stats)
        m_target = assets.mouth_level[pos]                 # 0~1
        e_target = assets.eye_level[pos]
        # 嘴角上扬：只有"闭口微笑"类标签会被加分（张嘴类不看嘴角）
        geo = 0.55 * (1.0 - np.abs(p_mouth - m_target)) + 0.35 * (1.0 - np.abs(p_eye - e_target))
        geo += 0.10 * (1.0 - np.abs(p_lift - 0.7)) * np.clip(1.0 - m_target, 0.0, 1.0)
        geo = np.clip(geo, 0.0, 1.0)
    # ③ CLIP 图像相似度
    clip = np.full(n, 0.5)
    if evidence.get("clip_feat") is not None and assets.ready:
        clip = assets.clip_feat[pos] @ evidence["clip_feat"]

    def minmax(x: np.ndarray) -> np.ndarray:
        lo, hi = float(x.min()), float(x.max())
        if hi - lo < 1e-9:
            return np.full_like(x, 0.5)
        return (x - lo) / (hi - lo)

    attr_n, geo_n, clip_n = minmax(attr), minmax(geo), minmax(clip)
    total = w_attr * attr_n + w_geo * geo_n + w_clip * clip_n
    order = np.argsort(-total)
    out = []
    for k in order:
        i = int(pos[k])
        out.append({
            "position": i,
            "file": str(memes.iloc[i]["file"]),
            "emotion": str(memes.iloc[i]["emotion"]),
            "score": float(total[k]),
            "attr_sim": float(attr[k]),
            "geo_sim": float(geo[k]),
            "clip_sim": float(clip[k]),
        })
    return out


def explain_geometry(geom: dict | None) -> list[tuple[str, str]]:
    """把几何量翻译成页面可读的"神态描述"。"""
    if not geom:
        return []
    stats = load_geom_stats()
    p_mouth = _percentile("mouth_aspect", geom["mouth_aspect"], stats)
    p_eye = _percentile("eye_open", geom["eye_open"], stats)
    p_lift = _percentile("corner_lift", geom["corner_lift"], stats)

    def level(p: float, names: tuple[str, str, str]) -> str:
        return names[0] if p < 0.33 else (names[1] if p < 0.66 else names[2])

    return [
        ("嘴巴开合度", f"{'闭着' if p_mouth < 0.35 else ('微张' if p_mouth < 0.7 else '张大')}"
                    f"（分位 {p_mouth * 100:.0f}%）"),
        ("眼睛睁开度", f"{'偏小/眯' if p_eye < 0.33 else ('正常' if p_eye < 0.66 else '睁大')}"
                    f"（分位 {p_eye * 100:.0f}%）"),
        ("嘴角上扬", f"{'不明显' if p_lift < 0.5 else '明显'}（分位 {p_lift * 100:.0f}%）"),
        ("头部倾斜", f"{geom['head_tilt']:+.1f}°"),
    ]
