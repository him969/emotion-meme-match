# -*- coding: utf-8 -*-
r"""
17_meme_gif_frames.py —— GIF 多帧视觉特征（B/C 方案）的基础设施

产出（只读原始 GIF，不改数据管线/模型）：
  数据\meme_gif_frames\<stem>.npz   采样帧 (N,256,256,3) uint8 + 时长权重 weights(N,) + peak_index
  数据\meme_gif_frames\_index.json 每个 GIF 的元信息（文件、情绪、帧数、峰值帧等）
  数据\meme_anim_peak\<stem>.png    "峰值帧"图像（供 C 方案/目检用）

两种模式（页面侧边栏可选）：
  B "mean"：多帧时长加权平均视觉特征（视频级表征；帧权重按 duration 加权，更符合"时间占比"）
  C "peak"：峰值帧视觉特征——定义为"与首帧视觉差异最大的采样帧"（ImageNet ResNet18 余弦距离），
            对"表情变化大/首帧是淡入黑屏"的 GIF 更贴切；峰值帧由本脚本离线确定，保证 B/C 两处引用同一帧。

运行：
  $env:PYTHONIOENCODING='utf-8'
  python 代码\17_meme_gif_frames.py
  # 可选 --max-frames 16 --size 256 --force
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
from PIL import Image, ImageSequence

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "数据"
DEPS_DIR = ROOT / "_deps"
if str(DEPS_DIR) not in sys.path:            # timm / safetensors 装在项目内 _deps
    sys.path.insert(0, str(DEPS_DIR))
MEME_CSV = DATA_DIR / "meme_index.csv"
SRC_ROOT = DATA_DIR / "cat_meme_extracted" / "cat_meme"
FRAMES_DIR = DATA_DIR / "meme_gif_frames"
PEAK_DIR = DATA_DIR / "meme_anim_peak"
INDEX_JSON = FRAMES_DIR / "_index.json"
IMGNET_WEIGHTS = DEPS_DIR / "weights" / "resnet18.a1_in1k.safetensors"
IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


def find_source(file_name: str, emotion: str) -> pathlib.Path | None:
    p = SRC_ROOT / emotion / file_name
    if p.exists():
        return p
    matches = list(SRC_ROOT.rglob(file_name))
    return matches[0] if matches else None


def load_frames(src: pathlib.Path, size: int, max_frames: int):
    """返回 (frames_uint8, weights, durations, n_total)。均匀采样 + 时长加权。"""
    im = Image.open(src)
    n_total = getattr(im, "n_frames", 1)
    if n_total <= 1:
        return None
    step = max(1, n_total // max_frames)
    frames, weights, durations = [], [], []
    for idx, fr in enumerate(ImageSequence.Iterator(im)):
        if idx % step != 0 and idx != n_total - 1:
            continue
        duration = fr.info.get("duration", im.info.get("duration", 80)) or 80
        rgb = fr.convert("RGBA")
        w, h = rgb.size
        scale = size / max(w, h)
        rgb = rgb.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), "white")
        canvas.paste(rgb, ((size - rgb.width) // 2, (size - rgb.height) // 2), rgb)
        frames.append(np.asarray(canvas, dtype=np.uint8))
        # 采样跳帧时，该帧代表 step 个原始帧 → 权重按时长×step
        weights.append(float(min(500, duration * step)))
        durations.append(float(duration))
        if len(frames) >= max_frames:
            break
    w = np.asarray(weights, dtype=np.float32)
    w = w / w.sum()
    return np.stack(frames, axis=0), w, np.asarray(durations, dtype=np.float32), n_total


class _ImgNetFeature:
    """峰值帧检测用：timm ResNet18 全局特征（与 11_visual_diversity.py 的 imgnet 同款）。"""

    def __init__(self):
        import torch
        import timm
        from safetensors.torch import load_file

        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = timm.create_model("resnet18", pretrained=False, num_classes=0)
        sd = load_file(str(IMGNET_WEIGHTS))
        m.load_state_dict({k: v for k, v in sd.items() if not k.startswith("fc.")}, strict=False)
        self.model = m.to(self.device).eval()

    def features(self, frames: np.ndarray, size: int = 128) -> np.ndarray:
        """(N,256,256,3) uint8 → (N,512) L2 归一化特征。"""
        torch = self.torch
        tensors = []
        for arr in frames:
            im = Image.fromarray(arr, mode="RGB").resize((size, size), Image.Resampling.LANCZOS)
            a = np.asarray(im, dtype=np.float32) / 255.0
            a = (a - IMAGENET_MEAN) / IMAGENET_STD
            tensors.append(torch.from_numpy(a).permute(2, 0, 1))
        xb = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            f = self.model.forward_features(xb)
            f = self.model.global_pool(f)
            f = torch.flatten(f, 1).float().cpu().numpy()
        return f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser(description="GIF 采样帧/时长权重/峰值帧 基础设施")
    ap.add_argument("--max-frames", type=int, default=16)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import pandas as pd
    memes = pd.read_csv(MEME_CSV)
    gifs = memes[memes["is_gif"] == 1]
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    PEAK_DIR.mkdir(parents=True, exist_ok=True)

    index = {}
    if INDEX_JSON.exists():
        try:
            index = json.loads(INDEX_JSON.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            index = {}

    feat = _ImgNetFeature()
    print("=" * 92)
    print(f"GIF 采样帧/峰值帧：共 {len(gifs)} 张，输出 {FRAMES_DIR}")
    print("=" * 92)
    t0 = time.time()
    ok, cached, failed = 0, 0, []
    for _, row in gifs.iterrows():
        file_name = row["file"]
        stem = pathlib.Path(file_name).stem
        npz_path = FRAMES_DIR / f"{stem}.npz"
        peak_path = PEAK_DIR / f"{stem}.png"
        if npz_path.exists() and peak_path.exists() and not args.force:
            cached += 1
            print(f"  [跳过] {file_name:40s} 已存在")
            continue
        src = find_source(file_name, row["emotion"])
        if src is None:
            failed.append((file_name, "找不到原图"))
            continue
        try:
            loaded = load_frames(src, args.size, args.max_frames)
            if loaded is None:
                failed.append((file_name, "不是动图"))
                continue
            frames, weights, durations, n_total = loaded
            feats = feat.features(frames)
            # 峰值帧 = 与首帧视觉差异（余弦距离）最大的采样帧
            dist_to_first = 1.0 - (feats @ feats[0])
            peak_idx = int(np.argmax(dist_to_first))
            np.savez_compressed(npz_path, frames=frames, weights=weights,
                                durations=durations, peak_index=np.int32(peak_idx))
            Image.fromarray(frames[peak_idx], mode="RGB").save(peak_path)
            index[stem] = {"file": file_name, "emotion": row["emotion"], "n_total_frames": int(n_total),
                           "n_sampled": int(len(frames)), "peak_index": peak_idx,
                           "peak_distance": round(float(dist_to_first[peak_idx]), 4),
                           "peak_png": str(peak_path.relative_to(ROOT)),
                           "frames_npz": str(npz_path.relative_to(ROOT)),
                           "size": args.size, "weight_sum": round(float(weights.sum()), 4)}
            ok += 1
            print(f"  [OK]   {file_name:40s} 原{n_total:3d}帧 采样{len(frames):3d} "
                  f"峰值帧#{peak_idx:2d} 差异={dist_to_first[peak_idx]:.3f} "
                  f"时长{weights.sum():.1f}ms")
        except Exception as e:  # noqa: BLE001
            failed.append((file_name, repr(e)[:100]))
            print(f"  [失败] {file_name:40s} {type(e).__name__}: {e}")

    INDEX_JSON.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print("-" * 92)
    print(f"完成：新生成 {ok}，复用 {cached}，失败 {len(failed)}，用时 {time.time() - t0:.1f}s")
    if failed:
        for name, reason in failed:
            print(f"  - {name}: {reason}")
    total_frames = sum(v["n_sampled"] for v in index.values())
    print(f"共 {total_frames} 帧被采样；峰值不是首帧的 GIF："
          f"{sum(1 for v in index.values() if v['peak_index'] != 0)}/{len(index)}")
    print(f"索引：{INDEX_JSON.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
