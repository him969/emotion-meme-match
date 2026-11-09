# -*- coding: utf-8 -*-
r"""
07_demo_app.py —— 阶段 4 + 5：推理 + meme 匹配 + Streamlit 演示页面

页面流程：
    上传图片 / 随机来一张测试图 → 预处理 → 模型（自写 CNN 或预训练 ResNet）→ 7 类概率
    → 匹配策略挑一张猫 meme → 展示（概率条 + 选中 meme + 匹配依据）

支持的模型（自动识别权重包里的 profile）：
  · 自写 CNN（04_model.py / 05_train.py，profile = baseline / wide / flat）：48×48 灰度
  · ImageNet 预训练 ResNet 微调（08_transfer_train.py，profile = timm:resnet18）：128×128 RGB

TTA（方案 B）：
  推理时把同一张图做 6 个视角（原图 / 水平翻转 / 上下左右小平移），softmax 概率取平均，
  页面侧边栏可以开关（默认开）。实测能小幅提升准确率，且推理成本很低。

两种匹配策略（对应 PROJECT.md 第 5 节阶段 4）：
  ① 硬匹配（baseline）：取概率最大的情绪 → 在该类候选里随机挑一张
     缺点：argmax 错了就整体错；某类只有 1 张时会反复同一张
  ② 软匹配（推荐）：给每张 meme 一个情绪，计算
        score(meme) = Σ_i p_i × sim(emotion_i, meme.emotion)
     其中 sim 是 7×7 情绪相似度矩阵（对角=1，越像越大）。
     好处：模型不确定时（比如 disgust 0.2 / angry 0.5）也会选出合理的表情，
           并且天然带"回退"效果——某类没图时相似类会拿到分。
     注意：同一情绪的所有 meme 分数完全相同；2026-10-31 已修复为"选中情绪不变、
           在同类同分候选里挑一张 + 避免连续重复"，否则同一种情绪永远只显示同一张图。
           同分候选的挑法现在在侧边栏可选（"同分候选选择"）：
             · 随机 + 防连续重复（默认，演示观感最好）
             · 纯随机（不防重复，统计上等价 i.i.d.）
             · 确定性（按图片内容哈希固定，同一张图每次结果相同，可复现/可解释）
             · 固定第一张（修复前行为，仅用于对照演示"旧 bug"）
           这些都只影响"同情绪类内挑哪张"，不影响情绪类别判断与任何准确率指标。

建议 3：视觉多样性备选（侧边栏可开关，默认关）
  打开后展示 1 主推 + 2 备选；备选范围（仅同类/同类为主/自由补充）和视觉特征
  （ImageNet/FER）用户可选，"换一批"只换备选。算法与验证见 代码\11_visual_diversity.py
  与 结果\proto_visual_diversity_report.md（主推逻辑完全不变）。

神似模式（建议 2 落地，侧边栏可开关，默认关；依赖 代码\13_shensi_visual.py）
  情感主判据不变，在候选 meme 上再做一层"视觉加权求和"：
      F(meme) = (1-β) × 归一化情感分 + β × 视觉神似分
      视觉神似分 = 0.50 × 神态属性桥（CLIP 文本属性画像：眼/嘴/眉/气质 16 维）
                 + 0.35 × 关键点几何（人脸 478 点 vs meme 人工神态标签）
                 + 0.15 × CLIP 外观相似度
  验证集实测（β=0.1~0.6）：情绪命中率 0.6773 → 0.6762（-0.11 个百分点，基本无损），
  类内平均视觉神似分 0.691 → 0.734；β 由页面滑条控制（默认 0.35）。
  依赖 _deps 里的 CLIP 权重/mediapipe；缺依赖时该开关自动隐藏，不影响原有功能。

模型优化 v2（2026-10-31，侧边栏可开关，默认关）
  最终模型为 efficientnet_b0@160 微调 + 3 个随机种子（42/43/44）概率平均 + 6 视角 TTA。
  侧边栏"3-seed 集成（概率平均）"勾选后：分别用 v2_seed42/43/44 三个模型推理，
  把 softmax 概率取平均再选情绪（测试集 75.93% / Macro-F1 0.7443）；不勾选时用单模型
  （默认加载 `模型\best_model.pth` = v2 单模型，val 72.64%）。旧 ResNet18 模型保留为
  `模型\best_model_v1.pth` 可对比回退。

GIF 动图播放（侧边栏可开关，默认开）
  数据管线为了训练/匹配/特征一致，把 GIF 也处理成静态首帧 PNG（数据\meme_ready\），
  所以页面原本不会动。代码\16_meme_anim.py 额外生成 256×256 压缩动图（数据\meme_anim\），
  侧边栏"GIF 动图播放"开启时，选中的 GIF 表情包会播放动图（Streamlit 对 ≤1460px 的 GIF
  原样透传，浏览器自动播放）；关闭则显示静态首帧。**动画只影响展示**——模型输入、软匹配、
  视觉特征、神似模式仍然使用静态首帧，准确率与匹配分数不受任何影响。

GIF 视觉特征（B/C 方案，2026-11-01，侧边栏单选，默认"首帧"）
  GIF 表情包（20 张）的视觉特征来源可选：
    · 首帧（默认）：原行为，用数据\meme_ready\ 的静态首帧 PNG；
    · B 多帧平均：按帧时长加权平均多帧视觉特征（视频级表征）；
    · C 峰值帧：与首帧视觉差异最大的采样帧（资产由 代码\17_meme_gif_frames.py 生成）。
  只影响"视觉多样性备选"和"神似模式"的视觉分；非 GIF 的 40 张不变；主匹配、
  情绪标签与准确率完全不受影响（模型输入仍是静态首帧）。

显存不足兜底（2026-11-02 加固）
  · 模型缓存 `st.cache_resource(max_entries=2)`：切换权重/集成不会无限堆积旧模型；
  · 侧边栏"🧹 释放显存/内存缓存"按钮：清模型缓存 + `sv.release_clip()`（CLIP）
    + `vd.release_feature_cache()`（FER/ImageNet 特征与模型）+ empty_cache + gc；
  · 低显存自动降级：空闲显存 < 900MB 时神似模式 CLIP 自动走 CPU（meme 特征走缓存，
    只编码当前一张人脸，约 1~2 秒）；OOM 时再强制 CPU 重试一次；ImageNet 视觉特征
    低显存时也直接用 CPU；均失败才退回普通匹配并提示关闭占 GPU 的程序或 3-seed/神似。

侧边栏可读性 + 匹配层校准（2026-11-02）：
  · 权重选择改为"模型系列 → 具体权重"两级下拉；每项带中文说明与 val/test 指标
    （⭐v2 最终 / 🚀v2 3-seed / v1 旧模型 / 自写 CNN 对照 / v2 消融 / 实验版），
    `build_weight_groups()` 会自动覆盖全部 `.pth`，未登记的归入"其他"组；
  · 新增"概率温度校准"开关（默认关）：T=0.65 只作用于软匹配打分（val 拟合，CV label_hit +0.5pp），
    不改变分类 argmax/准确率，也不改变页面展示的原始概率条。

没有 opencv 的处理：
    本项目环境装不上 opencv（Python 3.14 无 wheel），所以不做人脸检测；
    上传时请上传"已经裁好的脸"或直接用下方"随机来一张 FER2013 测试图"按钮。
    （注：2026-10-31 复核发现 opencv 5.0.0.93 有 abi3 轮子、已随 mediapipe 装进 _deps，
     神似模式的人脸关键点就是用它；但主流程仍不依赖 opencv，原有结论保持。）

运行演示（PowerShell）：
    $env:PYTHONIOENCODING='utf-8'
    python -m streamlit run <项目根>/代码/07_demo_app.py
    # 浏览器打开 http://localhost:8501

不打开网页的自检（验证推理+匹配逻辑能跑通）：
    python <项目根>/代码/07_demo_app.py --selftest
"""
import os
import pathlib
import sys

# 本机 conda 与 torch 各带一份 Intel OpenMP；torch + matplotlib 一起用会报
# "OMP Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized" 并直接中止进程。
# 这个环境变量只是让 OpenMP 允许重复加载（把"崩溃"变"继续跑"），**它不清理/不限制内存或显存**，
# 也不能预防 CUDA out of memory（显存问题要靠 empty_cache / 减少常驻模型 / 降级 CPU 等，见第 7.3 节）。
# 必须在 import torch 之前设。
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# 显存碎片化缓解（可选）：PyTorch 的 expandable_segments 在 Windows 上不支持，故只对非 Windows 生效。
if sys.platform != "win32":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Windows 控制台默认 GBK，打印 ✅/❌ 会抛 UnicodeEncodeError；统一改成 UTF-8 输出
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import importlib.util
import gc
import hashlib
import json
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "代码"
MODEL_DIR = ROOT / "模型"
DATA_DIR = ROOT / "数据"
MEME_INDEX_CSV = DATA_DIR / "meme_index.csv"
MEME_ANIM_DIR = DATA_DIR / "meme_anim"        # GIF 表情包的演示用动画版（16_meme_anim.py 生成，仅用于页面展示）
MULTI_LABEL_JSON = DATA_DIR / "meme_emotion_multi.json"   # 多情绪 meme 人工标注（实验用；默认不启用）
FER_NPZ = DATA_DIR / "fer2013" / "fer2013.npz"
DEFAULT_WEIGHTS = MODEL_DIR / "best_model.pth"

# 迁移学习用到的 timm / huggingface_hub / safetensors 装在项目内 _deps（沙箱不允许改全局环境），
# 先把 _deps 加进 sys.path，这样加载 timm 权重时才 import 得到。
DEPS_DIR = ROOT / "_deps"
if str(DEPS_DIR) not in sys.path:
    sys.path.insert(0, str(DEPS_DIR))


def _load_visual_diversity():
    """按路径加载建议 3 的视觉多样性模块（11_visual_diversity.py），避免模块名冲突。"""
    spec = importlib.util.spec_from_file_location("visual_diversity", CODE_DIR / "11_visual_diversity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 视觉多样性备选（建议 3）的共享逻辑：特征、候选池、贪心选择、换一批
vd = _load_visual_diversity()


def _load_shensi_visual():
    """按路径加载建议 2 的神似视觉模块（13_shensi_visual.py）。

    模块缺依赖/权重时返回 None → 页面自动隐藏"神似模式"开关，原有功能不受影响。
    """
    try:
        spec = importlib.util.spec_from_file_location("shensi_visual", CODE_DIR / "13_shensi_visual.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as e:  # noqa: BLE001
        print(f"[神似模式] 13_shensi_visual.py 加载失败，功能不可用：{type(e).__name__}: {e}")
        return None


sv = _load_shensi_visual()
SHENSI_BETA_DEFAULT = 0.35          # 默认视觉权重：验证集上情绪命中率仅 -0.11pp
SHENSI_TOP_K = 5                    # 页面展示 / "换一张"用的候选个数

# 同分候选选择策略：软/硬匹配在"同一情绪类内"所有 meme 的分数完全相同，
# 这里决定从同分候选里挑哪一张（页面侧边栏可选，默认保持原来的"随机+防重复"）。
TIE_STRATEGIES = {
    "random_avoid": "随机 + 防连续重复（默认，演示观感好）",
    "random": "纯随机（不防重复，统计上等价 i.i.d.）",
    "hash": "确定性（按图片内容固定，同一张图每次结果相同）",
    "first": "固定第一张（修复前行为，仅用于对照）",
}
TIE_DEFAULT = "random_avoid"

# F 方案：概率温度校准（val 拟合 T=0.65，软匹配 label_hit 0.7242→0.7292，5 折 CV +0.50pp）。
# 只影响"匹配打分"，不改变分类 argmax/准确率，也不改变页面展示的原始概率；侧边栏可选，默认关。
TEMP_FOR_MATCH = 0.65


def apply_temperature(probs: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """对类概率做温度缩放：softmax(log(p)/T)。T<1 更尖锐、T>1 更平；argmax 不变。

    支持 1-D（单张图，页面里 predict_probs 的返回值）和 2-D（N×7，批量评估）。
    """
    if temperature <= 0 or abs(temperature - 1.0) < 1e-9:
        return probs
    p_in = np.asarray(probs, dtype=np.float64)
    single = p_in.ndim == 1
    p2 = p_in[None, :] if single else p_in
    logits = np.log(np.clip(p2, 1e-12, None)) / float(temperature)
    logits = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    out = p / p.sum(axis=1, keepdims=True)
    return out[0] if single else out


# 权重分组（提高侧边栏可读性）：组名 → [(文件名, 友好说明), ...]
# 说明里带 val/test 指标，方便演示时直接判断"这是哪个模型"。
WEIGHT_GROUPS = [
    ("⭐ v2 最终模型（推荐）", [
        ("best_model.pth", "v2 最终 · EfficientNet-B0@160（val 72.64%；单模型 TTA 73.00%）"),
    ]),
    ("🚀 v2 3-seed 集成（勾选下方开关）", [
        ("v2_seed42.pth", "seed 42 · val 72.64%"),
        ("v2_seed43.pth", "seed 43 · val 72.14%"),
        ("v2_seed44.pth", "seed 44 · val 71.89%"),
    ]),
    ("v1 旧模型（对比/回退）", [
        ("best_model_v1.pth", "v1 · ResNet18@128 + TTA（val 67.51%；test 69.71%）"),
        ("best_transfer.pth", "v1 副本（与 best_model_v1 相同）"),
    ]),
    ("自写 CNN 对照（早期 baseline）", [
        ("best_model_flat2.pth", "E8 flat2 · 自写 CNN 最佳（test 64.73%）"),
        ("best_model_flat.pth", "E7 flat · val 59.29%"),
        ("best_model_wide.pth", "E5 wide · val 59.63%"),
        ("best_model_wide_weighted.pth", "E6 wide+类别权重 · val 54.61%"),
        ("best_model_plain_p8.pth", "E4 baseline 加强 · val 56.53%"),
        ("best_model_weighted_p8.pth", "E3 baseline+类别权重 · val 52.66%"),
        ("best_model_E1_baseline_w_p3.pth", "E1 baseline+类别权重 · val 50.18%"),
        ("best_model_E2_baseline_plain_p3.pth", "E2 baseline · val 47.37%"),
    ]),
    ("v2 消融 / backbone 实验", [
        ("v2_effb0_160.pth", "v2 单模型副本（= best_model）"),
        ("v2_resnet34_128.pth", "backbone 对比 · ResNet34@128（val 71.16%）"),
        ("v2_r1_amp.pth", "配方消融 R1 · AMP（val 68.35%）"),
        ("v2_r2_ema.pth", "配方消融 R2 · EMA（val 67.90%）"),
        ("v2_r3_mix.pth", "配方消融 R3 · mixup/强增强（val 69.10%）"),
        ("v2_r4_lion.pth", "配方消融 R4 · 再加 Lion（val 69.96%）"),
    ]),
    ("实验版（负结论，仅供研究）", [
        ("ferplus_hard_s42.pth", "P0 · FER+ 硬标签（原标签口径 No-Go）"),
        ("v2_focal_s42.pth", "A · focal loss（整体 No-Go；4 模型集成仅 +0.1pp）"),
        ("v2_effv2b0_s42.pth", "E · EffV2-B0（单模型未胜；+3seed 集成 val +0.28pp）"),
        ("v2_droppath_s42.pth", "B · drop_path+40ep（未胜出；单模型/集成均下降）"),
    ]),
]


def build_weight_groups(existing_files: list[str]) -> list[tuple[str, list[tuple[str, str]]]]:
    """按 WEIGHT_GROUPS 过滤实际存在的权重；未登记的文件归入"其他"组，保证不漏。"""
    exist = set(existing_files)
    groups: list[tuple[str, list[tuple[str, str]]]] = []
    used: set[str] = set()
    for group, items in WEIGHT_GROUPS:
        kept = [(f, desc) for f, desc in items if f in exist]
        if kept:
            groups.append((group, kept))
            used.update(f for f, _ in kept)
    rest = sorted(exist - used)
    if rest:
        groups.append(("🗂 其他/未分类权重", [(f, "未在分组表中登记（文件名即说明）") for f in rest]))
    return groups

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
CLASS_EMOJI = {"angry": "😠", "disgust": "🤢", "fear": "😨", "happy": "😄",
               "sad": "😢", "surprise": "😲", "neutral": "😐"}

# 7×7 情绪相似度矩阵（行/列顺序与 CLASS_NAMES 一致）。
# 数值是"两种情绪有多像"的主观先验：对角=1；越接近 1 越像。
# 用途：软匹配打分；以及某类没有 meme 时按相似度回退（例如 disgust 缺图 → 用 angry）。
EMOTION_SIM = np.array([
    # angry disgust fear happy sad surprise neutral
    [1.00, 0.80, 0.50, 0.05, 0.50, 0.20, 0.25],   # angry
    [0.80, 1.00, 0.30, 0.02, 0.45, 0.10, 0.20],   # disgust
    [0.50, 0.30, 1.00, 0.05, 0.40, 0.80, 0.20],   # fear
    [0.05, 0.02, 0.05, 1.00, 0.05, 0.60, 0.20],   # happy
    [0.50, 0.45, 0.40, 0.05, 1.00, 0.10, 0.60],   # sad
    [0.20, 0.10, 0.80, 0.60, 0.10, 1.00, 0.30],   # surprise
    [0.25, 0.20, 0.20, 0.20, 0.60, 0.30, 1.00],   # neutral
], dtype=np.float64)


def load_multi_labels(memes: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """读取多情绪 meme 标注，返回 (N,7) 分布 q 与信息字典 {file: {...}}。

    未在 JSON 里登记的 meme 保持 one-hot（原情绪=1），因此默认匹配逻辑完全不受影响；
    每个登记项要求：最多 2 个标签、权重和=1、dominant 与原 meme_index 情绪一致（防止误标）。
    """
    n = len(memes)
    q = np.zeros((n, len(CLASS_NAMES)), dtype=np.float64)
    emo_of = dict(zip(memes["file"].astype(str), memes["emotion"].astype(str)))
    file_idx = {f: i for i, f in enumerate(memes["file"].astype(str))}
    for f, emo in emo_of.items():
        q[file_idx[f], CLASS_NAMES.index(emo)] = 1.0
    info: dict = {}
    if not MULTI_LABEL_JSON.exists():
        return q, info
    try:
        raw = json.loads(MULTI_LABEL_JSON.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return q, info
    for f, rec in raw.items():
        if f.startswith("_") or f not in file_idx or not isinstance(rec, dict):
            continue
        labels = rec.get("labels") or []
        if not (1 <= len(labels) <= 2):
            continue
        row = np.zeros(len(CLASS_NAMES), dtype=np.float64)
        ok = True
        for lab in labels:
            emo = str(lab.get("emotion", ""))
            try:
                w = float(lab.get("weight", 0.0))
            except (TypeError, ValueError):
                ok = False
                break
            if emo not in CLASS_NAMES or w <= 0 or w > 1:
                ok = False
                break
            row[CLASS_NAMES.index(emo)] += w
        if not ok or abs(row.sum() - 1.0) > 1e-6:
            continue
        dom = CLASS_NAMES[int(row.argmax())]
        if rec.get("dominant") and str(rec["dominant"]) != dom:
            continue
        if dom != emo_of[f]:          # 主导情绪必须与原标签一致
            continue
        q[file_idx[f]] = row
        info[f] = {"dominant": dom,
                   "labels": [{"emotion": str(l["emotion"]), "weight": float(l["weight"])}
                              for l in labels]}
    return q, info


def meme_label_text(file_name: str, multi_info: dict) -> str:
    """用于页面展示的情绪标签：多标签 meme 显示 '主 0.7 / 副 0.3'，否则返回 ''。"""
    rec = multi_info.get(str(file_name))
    if not rec:
        return ""
    return " / ".join(f"{lab['emotion']} {lab['weight']:.1f}" for lab in rec["labels"])


# ---------------- 共享逻辑（selftest 与 Streamlit 页面都调用） ----------------

def load_checkpoint(weights_path: pathlib.Path, device: torch.device) -> dict:
    """读取训练保存的权重包。"""
    try:
        return torch.load(weights_path, map_location=device, weights_only=True)
    except Exception:
        return torch.load(weights_path, map_location=device, weights_only=False)


def build_model_from_ckpt(ckpt: dict, device: torch.device):
    """按权重包里的 profile 重建模型：
    · profile 以 "timm:" 开头 → ImageNet 预训练微调的 ResNet（08_transfer_train.py 产出）；
    · 其他 → 04_model.py 的自写 CNN（baseline / wide / flat）。
    """
    profile = ckpt.get("profile", "baseline")          # 旧权重没有 profile 字段 → 按 baseline 处理
    if profile.startswith("timm:"):
        import timm                                 # 只有迁移学习模型才需要 timm

        model = timm.create_model(ckpt["model_name"], pretrained=False,
                                  num_classes=int(ckpt["num_classes"]))
    else:
        spec = importlib.util.spec_from_file_location("fer_model", CODE_DIR / "04_model.py")
        md = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(md)
        model = md.build_model(num_classes=int(ckpt["num_classes"]), profile=profile)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


def model_input_spec(ckpt: dict) -> tuple[int, bool]:
    """返回模型的输入规格：(输入边长, 是否需要 3 通道)。
    迁移学习模型：128×128 RGB（ImageNet 归一化）；自写 CNN：48×48 灰度（FER mean/std）。
    """
    if str(ckpt.get("profile", "")).startswith("timm:"):
        return int(ckpt.get("img_size", 128)), True
    return 48, False


def preprocess_image(im: Image.Image, mean, std, out_size: int = 48, rgb: bool = False) -> torch.Tensor:
    """把任意图片变成模型要的输入：
    居中裁成正方形（不拉伸变形）→ 缩放 → 灰度（或 3 通道）→ 归一化。
    注意：没有 opencv，这里不做人脸检测，所以输入最好是"已经裁好的人脸"。
    """
    im = im.convert("RGB" if rgb else "L")                # 迁移模型要 3 通道，自写 CNN 要灰度
    w, h = im.size
    side = min(w, h)
    left, top = (w - side) // 2, (h - side) // 2
    im = im.crop((left, top, left + side, top + side))    # 居中裁正方形
    im = im.resize((out_size, out_size), Image.Resampling.LANCZOS)
    arr = np.asarray(im, dtype=np.float32) / 255.0        # 0~255 → 0~1
    if rgb:
        # 3 通道：mean/std 是 ImageNet 的 3 个值，按通道分别归一化，输出 (1,3,H,W)
        mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        arr = (arr - mean) / std
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    # 单通道：mean/std 是 FER 训练集的标量，输出 (1,1,H,W)
    arr = (arr - float(mean)) / float(std)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def _shift_batch(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """TTA 用：把图整体平移 (dx, dy) 像素，空出来的边填 0（标准化后约等于平均灰度）。"""
    if dx == 0 and dy == 0:
        return x
    h, w = x.shape[-2], x.shape[-1]
    out = torch.zeros_like(x)
    src_h = slice(max(0, -dy), h - max(0, dy))
    src_w = slice(max(0, -dx), w - max(0, dx))
    dst_h = slice(max(0, dy), h - max(0, -dy))
    dst_w = slice(max(0, dx), w - max(0, -dx))
    out[..., dst_h, dst_w] = x[..., src_h, src_w]
    return out


def tta_views(xb: torch.Tensor) -> list[torch.Tensor]:
    """TTA 的 6 个视角：原图、水平翻转、左/右/上/下各平移 s 像素（s 随输入尺寸自适应）。"""
    size = xb.shape[-1]
    s = max(2, round(size / 24))
    return [xb, torch.flip(xb, dims=[-1]),
            _shift_batch(xb, s, 0), _shift_batch(xb, -s, 0),
            _shift_batch(xb, 0, s), _shift_batch(xb, 0, -s)]


@torch.no_grad()
def predict_probs(model, x: torch.Tensor, device: torch.device, tta: bool = False) -> np.ndarray:
    """返回 7 类概率（numpy 数组）。tta=True 时对 6 个视角的 softmax 取平均（方案 B）。"""
    x = x.to(device)
    views = tta_views(x) if tta else [x]
    p = torch.zeros(1, len(CLASS_NAMES), device=device)
    for view in views:
        p += torch.softmax(model(view), dim=1)
    return (p / len(views))[0].cpu().numpy()


def load_memes() -> pd.DataFrame:
    """读取表情包元数据表（02_meme_index.py 生成）。"""
    if not MEME_INDEX_CSV.exists():
        raise FileNotFoundError(f"找不到 {MEME_INDEX_CSV}，请先运行 02_meme_index.py")
    df = pd.read_csv(MEME_INDEX_CSV)
    df["path"] = df["processed"].map(lambda p: str(DATA_DIR.parent / p))
    return df


def load_anim_map(memes: pd.DataFrame) -> dict[str, str]:
    """GIF 表情包 → 演示用动画版路径（不存在则不出现在映射里）。

    动画版由 代码\\16_meme_anim.py 生成（256×256 白底、压缩）；它**只用于页面展示**，
    模型输入/匹配分数/视觉特征/神似模式仍然用 meme_ready 里的静态首帧 PNG。
    """
    if "is_gif" not in memes.columns:
        return {}
    out: dict[str, str] = {}
    for _, row in memes[memes["is_gif"] == 1].iterrows():
        anim = MEME_ANIM_DIR / f"{pathlib.Path(str(row['file'])).stem}.gif"
        if anim.exists():
            out[str(row["file"])] = str(anim)
    return out


def show_meme_image(path, caption: str, anim_path: str | None = None,
                    gif_on: bool = True, **kwargs) -> None:
    """显示 meme 图：有动画版且开关打开时播放 GIF，否则显示静态 PNG。

    Streamlit 1.58 的 st.image 对 ≤1460px 的 GIF 会原样透传为 media/gif，
    浏览器自动播放；这里传 256×256 的压缩动图，体积可控。
    """
    import streamlit as st

    if gif_on and anim_path:
        st.image(anim_path, caption=f"{caption}（动图）", **kwargs)
    else:
        st.image(path, caption=caption, **kwargs)


def soft_scores(probs: np.ndarray, memes: pd.DataFrame,
                multi_q: np.ndarray | None = None) -> pd.DataFrame:
    """给每张 meme 按软匹配公式打分：score = Σ_i p_i × sim(情绪_i, meme 的情绪)。

    multi_q=None（默认）：meme 侧 one-hot（与原行为逐位一致）；
    multi_q=(N,7)：多情绪分布，score = Σ_i Σ_j p_i·SIM[i,j]·q_j = p·SIM·q。
    """
    if multi_q is None:
        # 每张 meme 的情绪在 CLASS_NAMES 里的下标
        idx = memes["emotion"].map({name: i for i, name in enumerate(CLASS_NAMES)}).to_numpy()
        # 对每个 meme：probs 与"它对应那一列"的相似度做点积
        scores = EMOTION_SIM[:, idx].T @ probs            # (n_memes, 7) @ (7,) → (n_memes,)
    else:
        q = np.asarray(multi_q, dtype=np.float64)         # (n_memes, 7)
        scores = (probs @ EMOTION_SIM) @ q.T              # (7,) @ (7,7) @ (7,N) → (N,)
    out = memes.copy()
    out["score"] = scores
    return out


def _pick_from_candidates(cand: pd.DataFrame, rng: random.Random,
                          avoid_file: str | None = None,
                          strategy: str = TIE_DEFAULT,
                          image_key: str | None = None) -> pd.Series:
    """从"同分候选"里挑一张，具体规则由 strategy 决定（页面可选）。

    · random_avoid（默认）：随机挑、并尽量避开上一次出现的那张（演示观感好）；
    · random：纯随机（不防重复，统计上等价 i.i.d.）；
    · hash：按输入图片内容的哈希取模（同一张图每次结果固定，可复现、可解释）；
    · first：固定挑排序第一张（= 修复前的旧行为，仅用于对照说明"修复前只会出同一张"）。
    """
    if strategy == "first":
        return cand.iloc[0]
    if strategy == "hash":
        digest = hashlib.md5((image_key or "").encode("utf-8")).hexdigest()
        return cand.iloc[int(digest[:8], 16) % len(cand)]
    # random / random_avoid：跳过上一张（仅当候选 >1 且上一张在候选里）
    if strategy == "random_avoid" and avoid_file is not None and len(cand) > 1:
        filtered = cand[cand["file"] != avoid_file]
        if len(filtered) > 0:
            cand = filtered
    return cand.iloc[rng.randrange(len(cand))]


def pick_meme(probs: np.ndarray, memes: pd.DataFrame, strategy: str = "soft",
              rng: random.Random | None = None, avoid_file: str | None = None,
              tie_strategy: str = TIE_DEFAULT, image_key: str | None = None,
              multi_q: np.ndarray | None = None, multi_info: dict | None = None) -> dict:
    """按策略挑一张 meme，返回 {'emotion','file','path','score','reason'}。

    ① hard：取 argmax 情绪 → 该类候选随机一张；
             该类没有图 → 找"相似度最高且有图"的情绪回退。
    ② soft：对全部 meme 算软匹配分，取分数最高的一组（天然带回退）。

    关于"同分随机"（软匹配全局修复）：
      软匹配里同一情绪的所有 meme 分数完全相同（S 矩阵那一列只取决于情绪），
      旧实现 sort_values 取第 1 张 → 同一种情绪永远只显示同一张图（实测 3589 次查询只用到 8 张）。
      现在改为：**选中情绪与旧逻辑保持完全一致**，但在这个情绪的所有同分候选里按
      tie_strategy 挑一张，从而让 66 张图都能被用到，且不影响任何准确率指标。
      tie_strategy 的四种取值与适用场景见 _pick_from_candidates 的注释；页面侧边栏可选，
      默认 random_avoid（随机 + 防连续重复）。

    multi_q/multi_info（实验用，默认 None=one-hot）：多情绪 meme 的 (N,7) 分布与信息字典；
    传入后 soft 分支用 p·SIM·q 打分，hard 分支仍按 dominant 单标签。
    """
    rng = rng or random.Random(2026)
    order = np.argsort(probs)[::-1]                  # 概率从大到小
    top_emotion = CLASS_NAMES[int(order[0])]
    available = set(memes["emotion"].unique())
    tie_note = f"同分候选策略：{TIE_STRATEGIES.get(tie_strategy, tie_strategy)}"

    if strategy == "hard":
        if top_emotion in available:
            cand = memes[memes["emotion"] == top_emotion]
            chosen = _pick_from_candidates(cand, rng, avoid_file, tie_strategy, image_key)
            return {"emotion": chosen["emotion"], "file": chosen["file"], "path": chosen["path"],
                    "score": float(probs[int(order[0])]),
                    "reason": f"硬匹配：预测 {top_emotion}（p={probs[int(order[0])]:.3f}），"
                              f"从该类 {len(cand)} 张里选（{tie_note}）"}
        # 回退：按情绪相似度从高到低找有图的情绪
        sims = EMOTION_SIM[int(order[0])]
        for j in np.argsort(sims)[::-1]:
            emo = CLASS_NAMES[int(j)]
            if emo in available:
                cand = memes[memes["emotion"] == emo]
                chosen = _pick_from_candidates(cand, rng, avoid_file, tie_strategy, image_key)
                return {"emotion": chosen["emotion"], "file": chosen["file"], "path": chosen["path"],
                        "score": float(sims[int(j)]),
                        "reason": f"回退：{top_emotion} 没有 meme，用相似情绪 {emo}"
                                  f"（sim={sims[int(j)]:.2f}）里的图（{tie_note}）"}

    # soft（默认）
    scored = soft_scores(probs, memes, multi_q)
    # 先按旧逻辑确定"选中哪一类"（保证和修复前的情绪选择完全一致），再在同类同分候选里选
    old_first = scored.sort_values("score", ascending=False).iloc[0]
    winner_emotion = old_first["emotion"]
    max_score = float(old_first["score"])
    same = scored[(scored["emotion"] == winner_emotion)
                  & np.isclose(scored["score"].to_numpy(), max_score, rtol=0.0, atol=1e-12)]
    chosen = _pick_from_candidates(same, rng, avoid_file, tie_strategy, image_key)
    if len(same) > 1:
        tie_note += f"（同类同分 {len(same)} 张）"
    multi_note = ""
    if multi_info:
        label_text = meme_label_text(str(chosen["file"]), multi_info)
        if label_text:
            multi_note = f"；多情绪标签：{label_text}"
    return {"emotion": chosen["emotion"], "file": chosen["file"], "path": chosen["path"],
            "score": float(chosen["score"]),
            "reason": f"软匹配：score=Σ p_i×sim = {chosen['score']:.3f}"
                      f"（选中 {chosen['emotion']} 类的图；模型最像 {top_emotion}）；{tie_note}{multi_note}"}


def top_meme_table(probs: np.ndarray, memes: pd.DataFrame, k: int = 5,
                   multi_q: np.ndarray | None = None,
                   multi_info: dict | None = None) -> pd.DataFrame:
    """软匹配分数最高的 k 张 meme，用于页面上展示依据。

    multi_q/multi_info 传入时（实验用）按多情绪分布打分，并附加"多情绪标签"列；默认行为不变。
    """
    scored = soft_scores(probs, memes, multi_q).sort_values("score", ascending=False).head(k)
    out = scored[["emotion", "source", "file", "score"]].reset_index(drop=True)
    if multi_info:
        out.insert(1, "多情绪标签", [meme_label_text(f, multi_info) or "-" for f in out["file"]])
    return out


# ---------------- 建议 2：神似模式（情感 + 视觉加权求和） ----------------

def pick_meme_shensi(probs: np.ndarray, memes: pd.DataFrame, strategy: str, image: Image.Image,
                     beta: float = SHENSI_BETA_DEFAULT, top_k: int = SHENSI_TOP_K,
                     rank_offset: int = 0, multi_q: np.ndarray | None = None,
                     multi_info: dict | None = None) -> dict:
    r"""神似模式：F(meme) = (1-β)×归一化情感分 + β×视觉神似分，返回 top-1 与候选明细。

    · 情感分：与软匹配同源的 score = Σ p_i × sim(情绪_i, meme 情绪)，按人脸 min-max 归一化；
    · 视觉神似分（13_shensi_visual.py）：0.50×神态属性画像 + 0.35×关键点几何 + 0.15×CLIP 外观；
    · β=0 时退化为"类内分数相同"（页面仍由旧逻辑随机挑）；β 由页面滑条控制，默认 0.35。
    · rank_offset：页面"换一张"用，在同一张人脸的 Top-K 候选里顺位选择。
    · multi_q/multi_info：实验用多情绪分布；默认 None 时与旧行为一致。
    """
    if sv is None or not sv.clip_available():
        raise RuntimeError("神似模式需要 代码\\13_shensi_visual.py 与 _deps 里的 CLIP 权重")
    base = pick_meme(probs, memes, strategy, multi_q=multi_q, multi_info=multi_info)  # 情感主判据
    evidence = sv.face_visual_evidence(image)
    comps = sv.visual_components(evidence, memes)
    emo = soft_scores(probs, memes, multi_q)["score"].to_numpy()
    span = float(emo.max() - emo.min())
    emo_n = (emo - emo.min()) / span if span > 1e-9 else np.full_like(emo, 0.5)
    final = (1.0 - beta) * emo_n + beta * comps["visual"]
    order = np.argsort(-final)
    k = min(top_k, len(order))
    pos = int(order[int(rank_offset) % k])

    rows = []
    for rank, p in enumerate(order[:k], start=1):
        p = int(p)
        rows.append({"排名": rank, "文件": memes.iloc[p]["file"], "情绪": memes.iloc[p]["emotion"],
                     "属性相似": round(float(comps["attr"][p]), 3),
                     "关键点匹配": round(float(comps["geo"][p]), 3),
                     "外观相似": round(float(comps["clip"][p]), 3),
                     "综合分F": round(float(final[p]), 3)})
    top_emotion = CLASS_NAMES[int(np.argmax(probs))]
    reason = (f"神似模式：F = (1-β)·情感分 + β·视觉分（β={beta:.2f}）\n\n"
              f"· 情感主判据（{strategy}）最像：**{top_emotion}**（p={probs[int(np.argmax(probs))]:.3f}）\n\n"
              f"· 视觉神似分 = 0.50×属性画像 + 0.35×关键点几何 + 0.15×外观相似")
    return {"emotion": str(memes.iloc[pos]["emotion"]),
            "file": str(memes.iloc[pos]["file"]),
            "path": str(memes.iloc[pos]["path"]),
            "score": float(final[pos]),
            "reason": reason,
            "shensi": {"beta": float(beta), "evidence": evidence, "ranking_df": pd.DataFrame(rows),
                       "order": [int(x) for x in order[:k]], "rank_offset": int(rank_offset),
                       "top_emotion": top_emotion, "components": comps}}


# ---------------- 自检模式（不开网页） ----------------

def selftest() -> None:
    """不启动网页，直接验证：权重能读、测试图能预测、两种匹配都能出图。"""
    print("=" * 82)
    print("07_demo_app.py 自检（不启动网页）")
    print("=" * 82)
    if not DEFAULT_WEIGHTS.exists():
        raise SystemExit(f"找不到权重 {DEFAULT_WEIGHTS}，请先运行 05_train.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = load_checkpoint(DEFAULT_WEIGHTS, device)
    model = build_model_from_ckpt(ckpt, device)
    mean, std = ckpt["norm_mean"], ckpt["norm_std"]      # 自写 CNN 是标量；迁移模型是 3 个值
    out_size, rgb = model_input_spec(ckpt)
    memes = load_memes()
    print(f"  设备：{device}   权重：{DEFAULT_WEIGHTS.name}  profile={ckpt.get('profile', 'baseline')}"
          f"  第 {ckpt.get('epoch')} 轮，val_acc={ckpt.get('val_acc'):.4f}")
    print(f"  输入规格：{out_size}×{out_size} {'RGB(ImageNet 归一化)' if rgb else '灰度(FER 归一化)'}")
    print(f"  meme 总数：{len(memes)}，每类：" +
          ", ".join(f"{e}={n}" for e, n in memes['emotion'].value_counts().items()))

    with np.load(FER_NPZ) as z:
        X, y = z["X_test"], z["y_test"]
    rng = random.Random(7)
    print("\n  随机抽 5 张测试图做端到端验证（推理开 TTA）：")
    ok = 0
    for k in range(5):
        i = rng.randrange(len(y))
        im = Image.fromarray(X[i], mode="L")
        x = preprocess_image(im, mean, std, out_size=out_size, rgb=rgb)
        probs = predict_probs(model, x, device, tta=True)
        top3 = np.argsort(probs)[::-1][:3]
        hard = pick_meme(probs, memes, "hard")
        soft = pick_meme(probs, memes, "soft")
        assert pathlib.Path(hard["path"]).exists() and pathlib.Path(soft["path"]).exists(), "选中的 meme 文件不存在！"
        ok += 1
        print(f"\n  样本 #{k}: 真实={CLASS_NAMES[int(y[i])]}")
        print("    top-3 概率：" + "  ".join(f"{CLASS_NAMES[int(j)]}={probs[int(j)]:.3f}" for j in top3))
        print(f"    硬匹配 → {hard['emotion']:9s} {pathlib.Path(hard['path']).name}")
        print(f"    软匹配 → {soft['emotion']:9s} {pathlib.Path(soft['path']).name}  score={soft['score']:.3f}")

    # 额外验证"回退"逻辑：假设 disgust 没有图
    probs_disgust = np.zeros(len(CLASS_NAMES))
    probs_disgust[1] = 0.9                                 # 假设模型强烈预测 disgust
    memes_no_disgust = memes[memes["emotion"] != "disgust"]
    fb = pick_meme(probs_disgust, memes_no_disgust, "hard")
    print(f"\n  回退逻辑验证：disgust 无图 + 模型预测 disgust → 选中 {fb['emotion']} 类（应为 angry）")
    assert fb["emotion"] == "angry", f"回退逻辑不对，选成了 {fb['emotion']}"

    # ---- 软匹配"同分随机"全局修复的验收（不依赖模型，直接构造概率，跑得快） ----
    print("\n  软匹配同分随机验证")
    # ① 选中情绪必须和修复前完全一致（逐条比较 100 组随机概率）
    mismatch = 0
    for k in range(100):
        p = np.random.default_rng(k).dirichlet(np.ones(len(CLASS_NAMES)))
        old_emotion = soft_scores(p, memes).sort_values("score", ascending=False).iloc[0]["emotion"]
        new_emotion = pick_meme(p, memes, "soft", rng=random.Random(k))["emotion"]
        mismatch += int(old_emotion != new_emotion)
    print(f"    100 组随机概率：选中情绪与旧逻辑不一致的条数 = {mismatch}（应为 0）")
    assert mismatch == 0, "软匹配随机化改变了选中的情绪类别！"

    # ② 同一情绪的多张图应该都能被选中（以 sad 为例：sad 类有 6 张）
    onehot_sad = np.zeros(len(CLASS_NAMES))
    onehot_sad[CLASS_NAMES.index("sad")] = 1.0
    sad_picks = [pick_meme(onehot_sad, memes, "soft", rng=random.Random(k)) for k in range(50)]
    sad_unique = len({p["file"] for p in sad_picks})
    print(f"    sad 类 50 次软匹配：用到 {sad_unique} 张不同 meme（应 > 1，理想接近 6）")
    assert all(p["emotion"] == "sad" for p in sad_picks), "同分随机不应改变情绪类别"
    assert sad_unique >= 5, "同分随机没有生效，仍然反复同一张图"

    # ③ 连续两次不重复：把第一张传给 avoid_file，第二次应换一张
    first = sad_picks[0]
    second = pick_meme(onehot_sad, memes, "soft", rng=random.Random(0), avoid_file=first["file"])
    print(f"    avoid_file 验证：第一次 {first['file']} → 第二次 {second['file']}")
    assert second["file"] != first["file"], "avoid_file 没有生效"

    # ④ 同分候选选择策略（页面可选）：random / hash / first 均不应改变选中情绪
    print("\n  同分候选选择策略验证（random / hash / first）")
    for ts in TIE_STRATEGIES:
        for k in range(10):
            pk = pick_meme(onehot_sad, memes, "soft", rng=random.Random(k),
                           tie_strategy=ts, image_key=f"demo-{k}")
            assert pk["emotion"] == "sad", f"{ts} 策略改变了选中情绪（{pk['emotion']}）"
    print("    4 种策略选中情绪均为 sad（情感主判据不受影响）")
    picks_random = [pick_meme(onehot_sad, memes, "soft", rng=random.Random(k), tie_strategy="random")
                    for k in range(50)]
    unique_r = len({p["file"] for p in picks_random})
    print(f"    纯随机：50 次用到 {unique_r} 张（应 >1，理想 6）")
    assert unique_r > 1, "纯随机没有生效"
    h1 = pick_meme(onehot_sad, memes, "soft", tie_strategy="hash", image_key="same-image")
    h2 = pick_meme(onehot_sad, memes, "soft", tie_strategy="hash", image_key="same-image")
    files_h = {pick_meme(onehot_sad, memes, "soft", tie_strategy="hash",
                         image_key=f"img-{i}")["file"] for i in range(8)}
    print(f"    确定性：同一图片两次都选 {h1['file']}（一致 = {h1['file'] == h2['file']}）；"
          f"8 个不同图片用到 {len(files_h)} 张")
    assert h1["file"] == h2["file"], "hash 策略不是确定性的（同一图片两次结果不同）"
    assert len(files_h) > 1, "hash 策略对不同图片没有区分度"
    picks_first = [pick_meme(onehot_sad, memes, "soft", tie_strategy="first") for _ in range(10)]
    unique_f = len({p["file"] for p in picks_first})
    print(f"    修复前行为（固定第一张）：10 次只用到 {unique_f} 张（{picks_first[0]['file']}）")
    assert unique_f == 1, "first 策略应该固定同一张"

    # ---- 建议 3：视觉多样性备选验收（1 主推 + 2 备选） ----
    print("\n  视觉多样性备选验证")
    feats = {kind: vd.get_meme_features(kind, memes, device, fer_model=model)
             for kind in ("imgnet", "fer")}
    label_idx = vd.label_indices(memes)
    n_checks = 0
    rng_vd = random.Random(11)
    last = None
    for _ in range(20):
        i = rng_vd.randrange(len(y))
        im = Image.fromarray(X[i], mode="L")
        x = preprocess_image(im, mean, std, out_size=out_size, rgb=rgb)
        probs = predict_probs(model, x, device, tta=True)
        picked = pick_meme(probs, memes, "soft", rng=random.Random(i))
        main_idx = int(np.flatnonzero(memes["file"].to_numpy() == picked["file"])[0])
        main_emotion = CLASS_NAMES.index(picked["emotion"])
        for scope in vd.SCOPE_LABELS:
            for kind, fm in feats.items():
                pool = vd.candidate_pool(label_idx, main_idx, main_emotion, scope)
                a, b = vd.greedy_diverse_pair(main_idx, pool, fm, label_idx, main_emotion, scope)
                assert a >= 0 and b >= 0, "备选数量不足（每类应至少 4 张）"
                assert len({main_idx, a, b}) == 3, "主推/备选出现重复图"
                if scope == "same":
                    assert label_idx[a] == main_emotion and label_idx[b] == main_emotion, "仅同类模式备选情绪不符"
                if scope == "majority":
                    assert label_idx[a] == main_emotion or label_idx[b] == main_emotion, "同类为主模式没有同类备选"
                npair = vd.reroll_pair((a, b), main_idx, pool, fm, label_idx, main_emotion, scope,
                                       recent=[], rng=random.Random(99))
                assert npair[0] != main_idx and npair[1] != main_idx, "换一批把主推换掉了"
                assert npair[0] >= 0 and npair[1] >= 0, "换一批返回了非法下标"
                n_checks += 1
        last = (picked, main_idx, a, b)
    picked, main_idx, a, b = last
    print(f"    20 张测试图 × 3 种范围 × 2 种特征 = {n_checks} 次选择/换一批全部通过")
    print(f"    样例：主推 {picked['file']}；备选 {memes['file'].iloc[a]} / {memes['file'].iloc[b]}")

    # ---- GIF 表情包动画版验收（页面"GIF 动图播放"开关用） ----
    print("\n  GIF 表情包动画版验证")
    anim_map = load_anim_map(memes)
    gif_files = memes[memes["is_gif"] == 1]["file"].tolist() if "is_gif" in memes.columns else []
    missing = [f for f in gif_files if f not in anim_map]
    print(f"    GIF {len(gif_files)} 张，找到动画版 {len(anim_map)} 张，缺失 {len(missing)} 张")
    if missing:
        print(f"    ⚠ 缺少动画版（请运行 代码\\16_meme_anim.py 生成）：{missing}")
    else:
        total_kb = 0.0
        for f, ap in anim_map.items():
            with Image.open(ap) as im_anim:
                assert im_anim.format == "GIF", f"{ap} 不是 GIF"
                assert getattr(im_anim, "n_frames", 1) > 1, f"{ap} 不是动图"
                assert im_anim.width == 256 and im_anim.height == 256, f"{ap} 尺寸不是 256×256"
            total_kb += pathlib.Path(ap).stat().st_size / 1024
        non_gif_anim = [f for f in anim_map if f not in gif_files]
        assert not non_gif_anim, f"非 GIF 出现动画版：{non_gif_anim}"
        print(f"    全部为 256×256 真动图，合计 {total_kb/1024:.1f} MB；演示开关默认开，关闭回退静态首帧")

    # ---- GIF 视觉特征模式（B 多帧平均 / C 峰值帧）验收 ----
    print("\n  GIF 视觉特征模式验证（first / mean / peak）")
    gif_files = memes[memes["is_gif"] == 1]["file"].tolist() if "is_gif" in memes.columns else []
    if gif_files and (data_dir_frames := (DATA_DIR / "meme_gif_frames")).exists():
        feat_by_mode = {}
        for _mode in vd.GIF_FEATURE_MODE_LABELS:
            fm = vd.get_meme_features("imgnet", memes, device, fer_model=model, gif_mode=_mode)
            assert fm.shape[0] == len(memes), f"{_mode} 特征行数不对"
            feat_by_mode[_mode] = fm
        gif_idx = [i for i, f in enumerate(memes["file"]) if f in gif_files]
        non_gif_idx = [i for i, f in enumerate(memes["file"]) if f not in gif_files]
        for _mode in ("mean", "peak"):
            diff_non = float(np.abs(feat_by_mode["first"][non_gif_idx]
                                    - feat_by_mode[_mode][non_gif_idx]).max())
            assert diff_non < 1e-6, f"{_mode} 模式改动了非 GIF 的特征（{diff_non}）"
            changed = int((np.abs(feat_by_mode["first"][gif_idx]
                                  - feat_by_mode[_mode][gif_idx]).sum(axis=1) > 1e-4).sum())
            print(f"    {_mode:5s}：GIF 特征被替换 {changed}/{len(gif_idx)} 张；"
                  f"非 GIF 不变（max diff={diff_non:.2e}）")
            assert changed >= max(1, len(gif_idx) // 2), f"{_mode} 模式几乎没有替换 GIF 特征"
        if sv is not None and sv.clip_available():
            base = sv.get_assets("first")
            assert base.ready, "神似模式基础资产未就绪"
            for _mode in ("mean", "peak"):
                a = sv.get_assets(_mode)
                assert a.ready and a.clip_feat.shape == base.clip_feat.shape
                d_clip = float(np.abs(a.clip_feat[[i for i, f in enumerate(memes["file"])
                                                    if f in gif_files]]
                                     - base.clip_feat[[i for i, f in enumerate(memes["file"])
                                                       if f in gif_files]]).sum())
                print(f"    神似资产 {_mode:5s}：CLIP/属性画像已切换（GIF 行差异和={d_clip:.3f}）")
        # CPU 兜底路径（页面在 GPU 显存不足时会自动走这条）
        cpu_feat = vd._extract("imgnet", vd.build_meme_tensor(memes.head(4)), torch.device("cpu"))
        print(f"    CPU 兜底：ImageNet 特征 {cpu_feat.shape} 可正常计算（显存不足时页面自动降级）")
        print("    说明：只影响'视觉多样性备选'与'神似模式'的视觉分；主匹配/准确率不变")
    else:
        print("    跳过（没有 GIF 或缺少 数据\\meme_gif_frames 资产，请先运行 17_meme_gif_frames.py）")

    # ---- 建议 2：神似模式（情感 + 视觉加权）验收 ----
    print("\n  神似模式（情感 + 视觉加权）验证")
    if sv is None or not sv.clip_available():
        print("    跳过：神似视觉模块不可用（缺 CLIP 权重或依赖）")
    else:
        assets = sv.get_assets()
        assert assets.ready, "神似模式：meme 视觉资产未就绪"
        rng_s = random.Random(13)
        n_s = 0
        for k in range(3):
            i = rng_s.randrange(len(y))
            im = Image.fromarray(X[i], mode="L")
            x = preprocess_image(im, mean, std, out_size=out_size, rgb=rgb)
            probs = predict_probs(model, x, device, tta=True)
            pk = pick_meme_shensi(probs, memes, "soft", im, beta=0.35)
            info = pk["shensi"]
            assert pathlib.Path(pk["path"]).exists(), "神似模式选中的文件不存在"
            assert len(info["ranking_df"]) == SHENSI_TOP_K, "神似候选数量不对"
            f_col = info["ranking_df"]["综合分F"].to_numpy()
            assert f_col[0] >= f_col[-1], "神似候选没有按综合分降序"
            if k == 0:
                geo = info["evidence"]["geometry"]
                top3 = "；".join(f"{r['文件']}（F={r['综合分F']:.3f}）"
                                for _, r in info["ranking_df"].head(3).iterrows())
                print(f"    样例 #{i}：情感主判据 {info['top_emotion']}；神似 top-3 = {top3}")
                if geo:
                    print(f"    关键点：检出 {geo['n_points']} 点，嘴开合 {geo['mouth_aspect']:.2f}，"
                          f"眼睁开 {geo['eye_open']:.2f}；视觉证据提取 {info['evidence']['timing_ms']:.0f}ms")
                else:
                    print("    关键点：未检出人脸（本次视觉分只用属性 + 外观）")
            n_s += 1
        print(f"    {n_s}/3 张测试图神似模式全部通过（Top-5 排序 / 文件存在性 / 几何提取）")

    # ---- 显存不足降级 / 缓存释放（对应页面加固：低显存 CPU 版神似 + 一键释放）----
    print("\n  显存不足降级 / 缓存释放验证")
    if sv is not None and sv.clip_available():
        sv.set_force_cpu(True)
        try:
            pk_cpu = pick_meme_shensi(probs, memes, "soft", im, beta=0.35)
            assert pathlib.Path(pk_cpu["path"]).exists(), "CPU 版神似选中的文件不存在"
            print(f"    CPU 版神似模式 OK：{pk_cpu['file']}（is_force_cpu={sv.is_force_cpu()}）")
        finally:
            sv.set_force_cpu(False)
        sv.release_clip()
        assert not sv.is_force_cpu(), "release_clip 后强制 CPU 标志应为 False"
        assert sv.clip_available(), "release_clip 后 CLIP 权重仍应可重新加载"
        print("    release_clip() 释放 CLIP 后仍可惰性重载（clip_available=True）")
    vd.release_feature_cache()
    f_after = vd.get_meme_features("imgnet", memes, torch.device("cpu"))
    assert f_after.shape[0] == len(memes), "释放特征缓存后重新计算失败"
    print(f"    release_feature_cache() 后重新计算特征 OK：{f_after.shape}")

    # ---- 模型优化 v2：3-seed 集成（权重齐全时验证概率平均链路） ----
    print("\n  模型优化 v2：3-seed 集成验证")
    seed_files = [MODEL_DIR / f"v2_seed{s}.pth" for s in (42, 43, 44)]
    ensemble_note = ""
    if all(p.exists() for p in seed_files):
        seed_models = []
        for sf in seed_files:
            ck_s = load_checkpoint(sf, device)
            seed_models.append((build_model_from_ckpt(ck_s, device), ck_s))
        i = rng.randrange(len(y))
        im = Image.fromarray(X[i], mode="L")
        probs_sum = None
        for m2, ck2 in seed_models:
            sz2, rgb2 = model_input_spec(ck2)
            xx = preprocess_image(im, ck2["norm_mean"], ck2["norm_std"], out_size=sz2, rgb=rgb2)
            pp = predict_probs(m2, xx, device, tta=True)
            probs_sum = pp if probs_sum is None else probs_sum + pp
        pe = probs_sum / len(seed_models)
        assert pe.shape == (len(CLASS_NAMES),), "集成概率形状不对"
        assert abs(float(pe.sum()) - 1.0) < 1e-4, "集成概率不归一"
        print(f"    3 个 seed 加载 + 概率平均 OK（样例 #{i}：top1={CLASS_NAMES[int(pe.argmax())]} "
              f"p={pe.max():.3f}；各 seed val_acc="
              + "、".join(f"{c.get('val_acc', 0):.2%}" for _, c in seed_models) + "）")
        ensemble_note = "；v2 3-seed 集成已验证"
    else:
        print("    跳过（v2_seed42/43/44 不全）")

    # ---- F 温度校准 / 权重分组可读性 ----
    print("\n  温度校准 / 权重分组验证")
    p_demo = np.array([0.05, 0.02, 0.10, 0.55, 0.08, 0.15, 0.05])
    p_demo = p_demo / p_demo.sum()
    p_cal = apply_temperature(p_demo, TEMP_FOR_MATCH)
    assert np.isclose(p_cal.sum(), 1.0), "温度校准后概率不归一"
    assert int(p_cal.argmax()) == int(p_demo.argmax()), "温度校准改变了 argmax（不应改变）"
    assert float(p_cal.max()) >= float(p_demo.max()), "T<1 应使概率更尖锐"
    print(f"    温度校准 T={TEMP_FOR_MATCH}: 概率和={p_cal.sum():.4f}，argmax 不变，"
          f"最高概率 {p_demo.max():.3f}→{p_cal.max():.3f}")
    weight_files_all = sorted(p.name for p in MODEL_DIR.glob("*.pth"))
    groups_demo = build_weight_groups(weight_files_all)
    covered = {f for _, items in groups_demo for f, _ in items}
    missing = set(weight_files_all) - covered
    assert not missing, f"权重分组遗漏：{missing}"
    assert any(g.startswith("⭐") and items[0][0] == DEFAULT_WEIGHTS.name for g, items in groups_demo), \
        "默认组应包含 best_model.pth"
    print(f"    权重分组：{len(groups_demo)} 组，覆盖 {len(covered)}/{len(weight_files_all)} 个 .pth（无遗漏），"
          f"默认组含 {DEFAULT_WEIGHTS.name}")

    print("\n" + "=" * 82)
    shensi_note = "；神似模式（情感+视觉加权）已验证" if (sv is not None and sv.clip_available()) else ""
    print(f"自检通过 ✅（{ok}/5 张样本；硬匹配/软匹配/回退正常；软匹配同分随机 + 防重复（含 4 种同分策略）；"
          f"视觉多样性 1 主推 + 2 备选已验证{shensi_note}{ensemble_note}；温度校准与权重分组已验证）")
    print("  下一步：用 streamlit 启动网页演示")
    print("=" * 82)


# ---------------- Streamlit 页面 ----------------

def is_oom_error(e: BaseException) -> bool:
    """判断是否为显存/内存不足（torch 版本不同，字符串兜底；也覆盖 CUDA driver 报错）。"""
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def cuda_free_mb() -> float | None:
    """当前 GPU 空闲显存（MB）；无 CUDA 时返回 None。"""
    if not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.mem_get_info()[0] / 2**20
    except Exception:  # noqa: BLE001
        return None


def run_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="人脸表情 → 猫 meme", page_icon="🐱", layout="wide")

    @st.cache_resource(show_spinner="加载模型中…", max_entries=2)   # 上限防切换权重时旧模型无限堆积
    def cached_models(names: tuple[str, ...]):
        """加载 1 个或多个权重；多模型时返回 [(model, ckpt), ...]，后续做概率平均集成。"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaded = []
        for name in names:
            ckpt = load_checkpoint(MODEL_DIR / name, device)
            loaded.append((build_model_from_ckpt(ckpt, device), ckpt))
        return loaded, device

    @st.cache_data(show_spinner="加载表情包…")
    def cached_memes():
        return load_memes()

    @st.cache_resource(show_spinner="加载测试集…")
    def cached_testset():
        with np.load(FER_NPZ) as z:
            return z["X_test"], z["y_test"]

    st.title("🐱 人脸表情识别 → 猫 meme 匹配")
    st.caption("课程项目演示：CNN 识别 7 类表情，再从已标注情绪的猫 meme 里挑最匹配的一张。"
               "可选模型：自写 CNN 或 ImageNet 预训练 ResNet 微调；推理默认开 TTA（多视角平均）。")

    # ---- 侧边栏 ----
    with st.sidebar:
        st.header("设置")
        weight_files = sorted(MODEL_DIR.glob("*.pth")) if MODEL_DIR.exists() else []
        if not weight_files:
            st.error("模型目录里没有 .pth 权重，请先运行 05_train.py")
            st.stop()
        # 按"模型系列 → 具体权重"两级选择，带中文说明与 val/test 指标（比裸文件名清晰）
        weight_groups = build_weight_groups([p.name for p in weight_files])
        group_names = [g for g, _ in weight_groups]
        default_group = next((i for i, g in enumerate(group_names) if g.startswith("⭐")), 0)
        group = st.selectbox("模型系列", group_names, index=default_group,
                             help="按用途分组：v2 最终 / 3-seed / v1 / 自写 CNN / 消融实验 / 实验版。"
                                  "每组里的说明都标了 val/test 指标。")
        group_items = dict(weight_groups)[group]
        item_labels = {f: desc for f, desc in group_items}
        if len(group_items) == 1:
            weights_name = group_items[0][0]
            st.caption(f"已选：{group_items[0][1]}")
        else:
            weights_name = st.selectbox("具体权重", [f for f, _ in group_items],
                                        format_func=lambda f: item_labels[f],
                                        help="同系列内可切换；说明中标注了验证集/测试集指标")
            st.caption(f"已选：{item_labels[weights_name]}")
        if group.startswith("🚀"):
            st.caption("提示：勾选下方「3-seed 集成（概率平均）」即可使用三模型集成。")
        # 模型优化 v2：如果存在 3 个随机种子的权重，允许多模型概率平均集成（默认关）
        ensemble_names = tuple(f"v2_seed{s}.pth" for s in (42, 43, 44))
        ensemble_ok = all((MODEL_DIR / n).exists() for n in ensemble_names)
        use_ensemble = False
        if ensemble_ok:
            use_ensemble = st.checkbox("3-seed 集成（概率平均）", value=False,
                                       help="用 v2_seed42/43/44 三个模型各自的 softmax 概率取平均再预测，"
                                            "通常比单模型更稳（约 +1~2 个点）")
        if use_ensemble:
            selected_names = ensemble_names
            weights_name = "集成：" + "+".join(n.replace(".pth", "") for n in ensemble_names)
        else:
            selected_names = (weights_name,)
        strategy = st.radio("匹配策略", ["soft", "hard"],
                            format_func=lambda s: "软匹配（推荐）" if s == "soft" else "硬匹配（baseline）")
        # 同分候选选择：软/硬匹配在"同情绪类内"分数相同，这里决定挑哪一张（默认保持原行为）
        tie_strategy = st.radio("同分候选选择", list(TIE_STRATEGIES.keys()), index=0,
                                format_func=lambda k: TIE_STRATEGIES[k],
                                help="同一情绪类内所有 meme 分数相同，这里决定挑哪一张：\n"
                                     "· 随机+防重复（默认）：演示观感最好\n"
                                     "· 纯随机：统计上独立同分布，不防重复\n"
                                     "· 确定性：按图片内容固定，同一张图每次结果相同（可复现/可解释）\n"
                                     "· 固定第一张：修复前的旧行为，仅用于对照（只会出同一张）\n"
                                     "注：神似模式下不生效（它按视觉分排序）。")
        tta_on = st.checkbox("TTA 多视角平均（方案 B，推荐）", value=True,
                             help="推理时对原图/翻转/平移共 6 个视角的概率取平均，通常更稳、准确率略高")
        # F 方案：概率温度校准（默认关，保持原行为；只影响匹配打分，不改变分类准确率/页面概率条）
        temp_on = st.checkbox("概率温度校准（改善软匹配，val +0.5pp）", value=False,
                              help="用 T=0.65 缩放模型概率后再做软匹配（T 在验证集上拟合；"
                                   "不改变分类 argmax/准确率，也不改变页面展示的原始概率条）。"
                                   "相似度矩阵仍用手工版本（矩阵调优经 CV 验证会过拟合，未采用）。")
        # GIF 表情包动图播放（默认开）：只影响展示，不影响模型/匹配/特征
        gif_on = st.checkbox("GIF 动图播放（GIF 表情包动起来）", value=True,
                             help="开启后，若选中的表情包是 GIF 且有动画版（数据\\meme_anim），页面播放动图；"
                                  "关闭则显示静态首帧。动画只影响展示，不影响模型/匹配/特征。")
        # GIF 视觉特征来源（B/C 方案）：只影响"视觉多样性备选"和"神似模式"的视觉分
        gif_feature_mode = st.radio(
            "GIF 视觉特征（仅影响备选/神似）", list(vd.GIF_FEATURE_MODE_LABELS.keys()), index=0,
            format_func=lambda k: vd.GIF_FEATURE_MODE_LABELS[k],
            help="GIF 表情包（20 张）的视觉特征来源：\n"
                 "· 首帧（默认）：原行为（预处理后的静态首帧）\n"
                 "· B 多帧平均：按时长加权平均多帧特征（视频级表征，更代表整段动画）\n"
                 "· C 峰值帧：与首帧视觉差异最大的采样帧（对'变脸/首帧淡入'类 GIF 更合适）\n"
                 "非 GIF 的 40 张不变；主匹配/情绪判断/准确率完全不受影响。")
        # 建议 3：视觉多样性展示（1 主推 + 2 备选）；默认关闭，保持原有单图行为
        diversity_on = st.checkbox("视觉多样性展示（1 主推 + 2 备选）", value=False,
                                   help="打开后额外展示 2 张视觉上尽量不同的备选图；主推逻辑不变")
        alt_scope, alt_feature = "same", "imgnet"
        if diversity_on:
            alt_scope = st.radio("备选范围", list(vd.SCOPE_LABELS.keys()), index=0,
                                 format_func=lambda k: vd.SCOPE_LABELS[k])
            alt_feature = st.radio("视觉特征", list(vd.FEATURE_LABELS.keys()), index=0,
                                   format_func=lambda k: vd.FEATURE_LABELS[k])
        st.divider()
        # 建议 2 落地：神似模式（情感为主 + 视觉加权），默认关；依赖缺失时自动隐藏
        shensi_on = False
        shensi_beta = SHENSI_BETA_DEFAULT
        if sv is not None and sv.clip_available():
            shensi_on = st.checkbox("神似模式（情感 + 视觉加权）", value=False,
                                    help="在情感主判据基础上，按「神态属性画像 + 人脸关键点几何 + 外观相似」"
                                         "给 meme 做视觉加权，挑更神似的一张。首次打开需加载 CLIP（约几秒）")
            if shensi_on:
                shensi_beta = st.slider("视觉权重 β", 0.10, 0.60, SHENSI_BETA_DEFAULT, 0.05,
                                        help="F = (1-β)·情感分 + β·视觉神似分。验证集实测 β=0.1~0.6 时"
                                             "情绪命中率只变化约 0.1 个百分点")
        elif sv is not None:
            st.caption("神似模式不可用：缺少 CLIP 权重或依赖（见 代码\\13_shensi_visual.py）")
        st.divider()
        # 显存/内存急救按钮：清空 Streamlit 模型缓存 + 释放 CUDA 缓存分配器 + 回收 Python 内存。
        # 不删除任何文件，也不会改匹配逻辑；点击后下次交互自动重载模型（约几秒）。
        if st.button("🧹 释放显存/内存缓存（卡住时用）", use_container_width=True,
                     help="清空已加载模型/CLIP/视觉特征的缓存，并调用 torch.cuda.empty_cache()/gc.collect()；"
                          "不会删除权重/数据，也不影响匹配结果，点击后下次交互自动重载（约几秒）。"):
            st.cache_resource.clear()
            if sv is not None:
                sv.release_clip()              # 释放 CLIP（神似模式占显存最多的一块）
            if vd is not None:
                vd.release_feature_cache()     # 释放 FER/ImageNet 特征与 ImageNet 模型缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            st.toast("已释放模型/CLIP/特征缓存与显存，正在重新加载…")
            st.rerun()
        st.markdown("**上传建议**\n\n"
                    "· 打开神似模式时会用 mediapipe 检测人脸关键点（可自动裁脸），普通照片也可以\n\n"
                    "· 不打开神似模式时仍建议上传**已经裁好的脸**（主流程不做人脸检测）")
        if st.button("🎲 随机来一张 FER2013 测试图", use_container_width=True):
            X_test, y_test = cached_testset()
            i = random.randrange(len(y_test))
            st.session_state["fer_index"] = int(i)

    loaded, device = cached_models(selected_names)
    model0, ckpt0 = loaded[0]
    memes = cached_memes()
    anim_map = load_anim_map(memes)     # GIF 表情包 → 演示用动画版（没有则为空字典）
    out_size, rgb = model_input_spec(ckpt0)
    profile = ckpt0.get("profile", "baseline")
    if len(loaded) > 1:
        profile = f"{profile} × {len(loaded)}（集成）"

    # ---- 输入区 ----
    st.subheader("① 选择一张人脸")
    col_in1, col_in2 = st.columns([1, 1])
    uploaded = None
    with col_in1:
        uploaded = st.file_uploader("上传图片（jpg/png，越接近正面清晰人脸越好）", type=["jpg", "jpeg", "png", "gif"])
    with col_in2:
        use_random = "fer_index" in st.session_state
        st.write("")
        st.write("")
        st.info("点击侧边栏的「🎲 随机来一张 FER2013 测试图」也可以直接体验。" if not use_random
                else f"当前使用 FER2013 测试集第 {st.session_state['fer_index']} 张图。")

    image = None
    true_label = None
    if uploaded is not None:
        image = Image.open(uploaded)
    elif use_random:
        X_test, y_test = cached_testset()
        i = st.session_state["fer_index"]
        image = Image.fromarray(X_test[i], mode="L")
        true_label = CLASS_NAMES[int(y_test[i])]

    if image is None:
        st.stop()

    # ---- 推理 + 匹配 ----
    # 逐模型预处理 + 推理（集成时概率平均）：不同模型可能输入尺寸不同，各自按自己的规格预处理；
    # 侧边栏的 tta_on 决定是否做 6 视角平均（方案 B）。
    probs_sum = None
    for m_k, ck_k in loaded:
        size_k, rgb_k = model_input_spec(ck_k)
        mean_k, std_k = ck_k["norm_mean"], ck_k["norm_std"]
        x_k = preprocess_image(image, mean_k, std_k, out_size=size_k, rgb=rgb_k)
        p_k = predict_probs(m_k, x_k, device, tta=tta_on)
        probs_sum = p_k if probs_sum is None else probs_sum + p_k
    probs = probs_sum / len(loaded)
    order = np.argsort(probs)[::-1]
    # F 方案：温度校准只用于匹配打分（不改变展示的概率条与分类 argmax）
    match_probs = apply_temperature(probs, TEMP_FOR_MATCH) if temp_on else probs
    # 输入图片的内容哈希：供"确定性（hash）"同分策略与神似模式"换一张"的顺位记忆使用
    image_key = hashlib.md5(image.convert("L").tobytes()).hexdigest()
    if shensi_on:
        # 神似模式：情感分与视觉分加权求和；按 图片+策略+β 记住"换一张"的顺位
        shensi_key = image_key + f"|{strategy}|{shensi_beta:.2f}"
        if st.session_state.get("shensi_key") != shensi_key:
            st.session_state["shensi_key"] = shensi_key
            st.session_state["shensi_offset"] = 0
        offset = int(st.session_state.get("shensi_offset", 0))
        if sv is not None:
            # GIF 视觉特征来源（B/C）：设置神似模式里 meme 侧 CLIP/属性画像使用的模式
            sv.set_default_gif_mode(gif_feature_mode)
        # 低显存自动降级：空闲 < 900MB 时，神似模式的 CLIP 直接走 CPU
        # （meme 特征走 模型\shensi_meme_cache.npz 缓存，只需要编码当前这一张人脸，约 1~2 秒）
        free_mb = cuda_free_mb()
        use_cpu_shensi = bool(free_mb is not None and free_mb < 900)
        if sv is not None:
            sv.set_force_cpu(use_cpu_shensi)
        if use_cpu_shensi:
            st.caption(f"GPU 仅剩 {free_mb:.0f} MB 空闲 → 神似模式自动改用 CPU 计算（约慢 1~2 秒）")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()              # 为 CLIP/关键点腾出显存，降低 OOM 概率
        with st.spinner("神似模式：提取人脸关键点 + 神态属性中…（首次需加载 CLIP，约几秒）"):
            try:
                picked = pick_meme_shensi(match_probs, memes, strategy, image,
                                          beta=shensi_beta, rank_offset=offset)
            except Exception as e:  # noqa: BLE001
                if is_oom_error(e):
                    # 仍 OOM：强制 CPU 重试一次，成功则继续用神似结果；失败才退回普通匹配
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if sv is not None:
                        sv.set_force_cpu(True)
                    try:
                        picked = pick_meme_shensi(match_probs, memes, strategy, image,
                                                  beta=shensi_beta, rank_offset=offset)
                        st.info("显存不足，神似模式已自动改用 CPU 重新计算成功（结果一致，只是稍慢）。")
                    except Exception as e2:  # noqa: BLE001
                        st.warning(f"神似模式失败（显存不足，CPU 重试仍失败：{type(e2).__name__}: {e2}）；"
                                   f"已自动退回普通匹配。建议关闭 3-seed 集成/神似模式，"
                                   f"或关闭微信/QQ/Edge 等占 GPU 的程序后再试。")
                        picked = pick_meme(match_probs, memes, strategy, rng=random.Random(),
                                           avoid_file=st.session_state.get("last_picked_file"),
                                           tie_strategy=tie_strategy, image_key=image_key)
                else:
                    st.warning(f"神似模式失败（{type(e).__name__}: {e}），已自动退回普通匹配。")
                    picked = pick_meme(match_probs, memes, strategy, rng=random.Random(),
                                       avoid_file=st.session_state.get("last_picked_file"),
                                       tie_strategy=tie_strategy, image_key=image_key)
    else:
        # 软/硬匹配：同分候选按侧边栏 tie_strategy 挑（默认"随机+防连续重复"）
        last_picked_file = st.session_state.get("last_picked_file")
        picked = pick_meme(match_probs, memes, strategy, rng=random.Random(),
                           avoid_file=last_picked_file,
                           tie_strategy=tie_strategy, image_key=image_key)
    st.session_state["last_picked_file"] = picked["file"]
    if temp_on:
        st.caption(f"匹配打分使用温度校准 T={TEMP_FOR_MATCH:.2f}（验证集拟合；"
                   f"不改变分类结果与页面概率条，仅让软匹配更稳）")

    st.subheader("② 模型看到的图 + 7 类概率")
    col_img, col_prob = st.columns([1, 2])
    with col_img:
        input_desc = f"{out_size}×{out_size} {'RGB' if rgb else '灰度'}"
        st.image(image, caption=f"输入图片（居中裁方 + 缩放到 {input_desc} 后送进模型）",
                 use_container_width=True)
        if true_label:
            st.success(f"这张图的真实标签：{CLASS_EMOJI[true_label]} {true_label}")
    with col_prob:
        st.markdown(f"**预测结果：{CLASS_EMOJI[CLASS_NAMES[int(order[0])]]} "
                    f"{CLASS_NAMES[int(order[0])]}**（概率 {probs[int(order[0])]:.1%}）")
        prob_df = pd.DataFrame({"情绪": [f"{CLASS_EMOJI[c]} {c}" for c in CLASS_NAMES],
                                "概率": probs})
        # 用带进度条的表格展示 7 类概率（Streamlit 的 ProgressColumn 自带横向条）
        st.dataframe(prob_df, hide_index=True, use_container_width=True,
                     column_config={"概率": st.column_config.ProgressColumn(
                         "概率", min_value=0.0, max_value=1.0, format="%.3f")})
        top3_text = "、".join(f"{CLASS_NAMES[int(j)]} {probs[int(j)]:.1%}" for j in order[:3])
        st.caption(f"Top-3：{top3_text}")

    st.subheader("③ 最匹配的猫 meme")
    col_meme, col_reason = st.columns([1, 2])
    with col_meme:
        show_meme_image(picked["path"], f"选中：{picked['file']}",
                        anim_map.get(picked["file"]), gif_on, use_container_width=True)
    with col_reason:
        st.markdown(f"**匹配到的情绪：{CLASS_EMOJI[picked['emotion']]} {picked['emotion']}**")
        st.write(picked["reason"])
        st.divider()
        st.markdown("**软匹配分数 Top-5（每张 meme 的打分依据）**")
        st.dataframe(top_meme_table(match_probs, memes, k=5), hide_index=True, use_container_width=True)
        with st.expander("看看软匹配是怎么算的"):
            st.markdown(
                "对每张 meme，它的情绪是已知的（例如 angry 类的图就是 angry）。\n\n"
                "score(meme) = Σ over 7 类 i：p_i × sim(i, meme 的情绪)\n\n"
                "· p_i 是模型给出的第 i 类概率\n\n"
                "· sim 是 7×7 情绪相似度矩阵（对角=1，例如 sad 和 neutral 有 0.6 的相似度）\n\n"
                "所以模型就算对 `disgust` 只有 0.2 的信心，只要 angry 的 0.5 不低，"
                "angry 的图也会得到不错的分数 —— 这就是「软匹配」比「硬匹配」稳的原因。")

    # ---- 建议 2：神似模式明细（侧边栏可开关；情感主判据不变） ----
    if shensi_on and isinstance(picked.get("shensi"), dict):
        info = picked["shensi"]
        st.divider()
        st.markdown(f"**神似模式明细**（β={info['beta']:.2f}：F = (1-β)·情感分 + β·视觉神似分）")
        col_geo, col_attr = st.columns([1, 1])
        with col_geo:
            st.markdown("**① 从人脸量出来的关键部位（478 点）**")
            geo = info["evidence"].get("geometry")
            if geo:
                for name, desc in sv.explain_geometry(geo):
                    st.write(f"· {name}：{desc}")
                st.caption(f"关键点提取 + 视觉打分耗时约 {info['evidence']['timing_ms']:.0f} ms（首次含加载模型）")
            else:
                st.caption("未检测到人脸关键点（人脸太小/侧脸/非人脸）：本次视觉分只用属性 + 外观相似")
        with col_attr:
            st.markdown("**② 神态属性画像（CLIP 零样本，16 维）**")
            prof = info["evidence"].get("attr_prof")
            if prof is not None:
                top_attr = np.argsort(-prof)[:6]
                st.write(" · ".join(f"{sv.ATTR_ZH[sv.ATTR_NAMES[int(j)]]} {prof[int(j)]:.2f}"
                                    for j in top_attr))
                st.caption("属性靠 CLIP 文本提示词（'张嘴/眯眼/高冷/沙雕…'）与图像算相似度，无需额外标注")
            else:
                st.caption("属性画像不可用（缺 CLIP 权重）")
        st.markdown("**③ 神似候选 Top-5（按综合分 F 排序）**")
        st.dataframe(info["ranking_df"], hide_index=True, use_container_width=True)
        btn_col, tip_col = st.columns([1, 3])
        with btn_col:
            if st.button("🎲 换一张神似候选", key="shensi_reroll"):
                st.session_state["shensi_offset"] = (
                    int(st.session_state.get("shensi_offset", 0)) + 1) % SHENSI_TOP_K
                st.rerun()
        with tip_col:
            st.caption("「换一张」只在这张人脸的 Top-5 神似候选里顺位切换；"
                       "情绪主判据与页面其他功能都不受影响。")

    # ---- ④ 视觉多样性备选（建议 3，侧边栏可开关） ----
    if diversity_on:
        st.subheader("④ 视觉备选（1 主推 + 2 备选）")
        st.caption(f"备选范围：{vd.SCOPE_LABELS[alt_scope]}　视觉特征：{vd.FEATURE_LABELS[alt_feature]}　"
                   f"差异分 = 1 - 余弦相似度（0=一样，越大越不同）")
        feats, feats_note = None, ""
        free_mb = cuda_free_mb()
        feature_device = device
        if free_mb is not None and free_mb < 900 and alt_feature == "imgnet":
            # 低显存时 ImageNet 特征（模型小）直接走 CPU，避免在 GPU 上反复 OOM
            feature_device = torch.device("cpu")
            st.caption(f"GPU 仅剩 {free_mb:.0f} MB 空闲 → 视觉特征自动改用 CPU 计算（约慢 1~2 秒）")
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()          # 先释放推理残留，降低 OOM 概率
            with st.spinner("加载 meme 视觉特征中…（首次需要几秒）"):
                feats = vd.get_meme_features(alt_feature, memes, feature_device, fer_model=model0,
                                             gif_mode=gif_feature_mode)
        except Exception as e:  # noqa: BLE001
            # 显存不足兜底：清缓存重试 → FER 自动降级 ImageNet → 再不行用 CPU 算 ImageNet
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if alt_feature == "fer":
                try:
                    with st.spinner("FER 特征加载失败，改用更省显存的 ImageNet 特征重试…"):
                        feats = vd.get_meme_features("imgnet", memes, device, fer_model=model0,
                                                     gif_mode=gif_feature_mode)
                    feats_note = (f"FER 特征加载失败（{type(e).__name__}），"
                                  f"已自动改用 ImageNet 特征继续（更省显存）。")
                except Exception:  # noqa: BLE001
                    feats = None
            if feats is None:
                try:
                    with st.spinner("GPU 显存不足，改用 CPU 计算 ImageNet 特征…"):
                        feats = vd.get_meme_features("imgnet", memes, torch.device("cpu"),
                                                     gif_mode=gif_feature_mode)
                    feats_note = (f"视觉特征加载失败（{type(e).__name__}），"
                                  f"已自动改用 CPU 上的 ImageNet 特征继续。")
                except Exception as e2:  # noqa: BLE001
                    msg = (f"视觉特征加载失败（{type(e2).__name__}: {e2}）；"
                           f"建议关闭微信/QQ/Edge 等占 GPU 的程序，或关闭 3-seed 集成/神似模式后重试。")
                    if is_oom_error(e2):
                        msg = "视觉特征加载失败：**显存不足**。" + msg
                    st.warning(msg)
        if feats_note:
            st.info(feats_note)
        if feats is not None:
            label_idx = vd.label_indices(memes)
            main_matches = np.flatnonzero(memes["file"].to_numpy() == picked["file"])
            main_idx = int(main_matches[0]) if len(main_matches) else 0
            main_emotion = CLASS_NAMES.index(picked["emotion"])
            pool = vd.candidate_pool(label_idx, main_idx, main_emotion, alt_scope)
            # 同一张主推 + 同一组选项 → 复用上次的备选；换图/换选项/点换一批时更新
            state_key = f"{picked['file']}|{main_emotion}|{alt_scope}|{alt_feature}"
            if st.session_state.get("diversity_key") != state_key or "diversity_pair" not in st.session_state:
                st.session_state["diversity_key"] = state_key
                st.session_state["diversity_pair"] = vd.greedy_diverse_pair(
                    main_idx, pool, feats, label_idx, main_emotion, alt_scope)
                st.session_state["diversity_recent"] = []
                st.session_state["diversity_reroll"] = 0
            pair = st.session_state["diversity_pair"]
            cols = st.columns(2)
            for k in range(2):
                alt = int(pair[k]) if k < len(pair) else -1
                if alt < 0:
                    continue
                dist_main = float(1.0 - feats[main_idx] @ feats[alt])
                emo = memes["emotion"].iloc[alt]
                with cols[k]:
                    show_meme_image(
                        memes["path"].iloc[alt],
                        f"备选{k+1}：{CLASS_EMOJI.get(emo, '')} {emo} | 与主推差异 {dist_main:.2f}",
                        anim_map.get(str(memes["file"].iloc[alt])), gif_on, use_container_width=True)
                    st.caption(f"来源：{memes['source'].iloc[alt]}　文件：{memes['file'].iloc[alt]}")
            if len(pair) == 2 and pair[0] >= 0 and pair[1] >= 0:
                dd = vd.pair_distances(main_idx, pair[0], pair[1], feats)
                st.caption(f"视觉差异分：主推↔备选1 {dd['main_alt1']:.2f}，主推↔备选2 {dd['main_alt2']:.2f}，"
                           f"备选1↔备选2 {dd['alt1_alt2']:.2f}（数值越大越不重样）")
            if st.button("🎲 换一批备选", key="diversity_reroll_btn"):
                rng = random.Random(st.session_state.get("diversity_reroll", 0) + 1)
                new_pair = vd.reroll_pair(pair, main_idx, pool, feats, label_idx, main_emotion,
                                          alt_scope, recent=st.session_state.get("diversity_recent", []),
                                          rng=rng)
                st.session_state["diversity_recent"] = (
                    st.session_state.get("diversity_recent", []) + [frozenset(pair)])[-7:]
                st.session_state["diversity_pair"] = new_pair
                st.session_state["diversity_reroll"] = st.session_state.get("diversity_reroll", 0) + 1
                st.rerun()

    st.divider()
    with st.expander("模型与数据信息"):
        ck0 = ckpt0
        st.write(f"权重：`{weights_name}`　模型：`{profile}`（第 {ck0.get('epoch')} 轮，"
                 f"验证集准确率 {ck0.get('val_acc', 0):.2%}，Macro-F1 {ck0.get('val_macro_f1', 0):.4f}）")
        if len(loaded) > 1:
            st.write("集成成员：" + "、".join(
                f"{n}（val {c.get('val_acc', 0):.2%}）" for n, (_, c) in zip(selected_names, loaded)))
        st.write(f"设备：`{device}`　输入：{out_size}×{out_size} {'RGB（ImageNet 归一化）' if rgb else '灰度（FER 归一化）'}"
                 f"　TTA：{'开（6 视角平均）' if tta_on else '关'}　meme 总数：{len(memes)} 张")
        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            st.caption(f"GPU 显存：已用 {(total_b-free_b)/2**20:.0f} MB / 共 {total_b/2**20:.0f} MB"
                       f"（显存不足时页面会自动降级视觉特征；也可关闭 3-seed 集成/神似模式）")
        else:
            st.caption("GPU：不可用（CPU 模式）")
        st.write("情绪相似度矩阵（行/列顺序：" + "、".join(CLASS_NAMES) + "）")
        st.dataframe(pd.DataFrame(EMOTION_SIM, index=CLASS_NAMES, columns=CLASS_NAMES).round(2),
                     use_container_width=True)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        run_app()
