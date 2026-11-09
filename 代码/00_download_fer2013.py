# -*- coding: utf-8 -*-
r"""
00_download_fer2013.py —— 阶段 1.1：获取 FER2013 数据集（HuggingFace 镜像版）

为什么用这个数据源？
  Kaggle 官方 FER2013 需要登录账号才能下载；HuggingFace 上有人把它转成了 parquet。
  本机实测 huggingface.co 直连超时、hf-mirror.com 可通，所以走镜像。
  该仓库的划分与官方完全一致：
      train       28709 张（训练）
      publicTest   3589 张（本项目当验证集）
      privateTest  3589 张（本项目当测试集）
  标签 0~6 与 Kaggle 官方顺序一致（见下方 CLASS_NAMES）。

产出：
  数据\fer2013\_raw\*.parquet      原始下载文件（保留，便于复查/重跑）
  数据\fer2013\fer2013.npz         转换后的数组：X_train/y_train、X_val/y_val、X_test/y_test
  数据\fer2013\meta.json           数据来源、划分、每类数量等元信息

运行（PowerShell）：
    $env:PYTHONIOENCODING='utf-8'
    python <项目根>/代码/00_download_fer2013.py
    # 想强制重新下载+重建：在末尾加 --rebuild
"""
import io
import json
import pathlib
import sys
import time
import urllib.request

import numpy as np

# Windows 控制台默认 GBK，打印 ✅/❌ 会抛 UnicodeEncodeError；统一改成 UTF-8 输出
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------- 路径：全部相对本文件定位，项目整体搬家不用改 ----------
ROOT = pathlib.Path(__file__).resolve().parents[1]      # 表情识别项目\
FER_DIR = ROOT / "数据" / "fer2013"
RAW_DIR = FER_DIR / "_raw"
NPZ_PATH = FER_DIR / "fer2013.npz"
META_PATH = FER_DIR / "meta.json"

# ---------- 数据源 ----------
MIRROR = "https://hf-mirror.com"                        # HuggingFace 国内镜像（直连 huggingface.co 超时）
REPO = "Aaryan333/fer2013_train_publicTest_privateTest"  # 提供官方三分划的仓库
# split 名 -> (仓库内文件路径, 期望字节数, 本项目里的 split 名)
FILES = {
    "train": ("data/train-00000-of-00001-5eab84e1c6a2fc27.parquet", 106477215, "train"),
    "publicTest": ("data/publicTest-00000-of-00001-f41bb7384b8aad6e.parquet", 13347727, "val"),
    "privateTest": ("data/privateTest-00000-of-00001-4b8a0715cf1b7560.parquet", 13360240, "test"),
}

# FER2013 官方七类：下标就是 parquet 里的 label 值（已从仓库 README 的 class_label 核对）
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
IMG_SIZE = 48                                           # FER2013 原图就是 48×48 灰度

FORCE = "--rebuild" in sys.argv                         # 带 --rebuild 时忽略已有文件，全部重来


def download(split: str) -> pathlib.Path:
    """下载一个 split；已存在且大小对得上就跳过（断点式重复运行不会白下）。"""
    rel, expect_size, _ = FILES[split]
    url = f"{MIRROR}/datasets/{REPO}/resolve/main/{rel}"
    out = RAW_DIR / f"{split}.parquet"
    if out.exists() and out.stat().st_size == expect_size and not FORCE:
        print(f"  [skip] {out.name} 已存在且大小正确（{expect_size/1e6:.2f} MB）")
        return out
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".parquet.part")              # 先写临时文件，下完再改名，避免半截文件
    print(f"  下载 {split}: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    t0, done, last_print = time.time(), 0, 0.0
    with urllib.request.urlopen(req, timeout=180) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1 << 20)                     # 每次读 1 MB
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if done - last_print >= 20 * (1 << 20):     # 每 20 MB 报一次进度
                speed = done / 1e6 / max(time.time() - t0, 1e-6)
                print(f"    {done/1e6:6.1f} MB / {expect_size/1e6:6.1f} MB  ({speed:.1f} MB/s)")
                last_print = done
    if done != expect_size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{split} 下载大小不对：得到 {done} 字节，期望 {expect_size} 字节")
    tmp.replace(out)
    print(f"  [ok]   {split}: {done/1e6:.2f} MB，用时 {time.time()-t0:.1f}s")
    return out


def load_split(path: pathlib.Path):
    """读 parquet，把 image 列里的 PNG 字节解码成 (N,48,48) uint8 灰度数组，返回 (X, y)。"""
    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(path, columns=["label", "image"])
    labels = np.asarray(table.column("label").to_pylist(), dtype=np.int64)
    # image 列是 struct<bytes, path>；这里只取 bytes（真正的图片内容）
    img_struct = table.column("image").combine_chunks()
    png_bytes = img_struct.field("bytes").to_pylist()

    n = len(labels)
    X = np.zeros((n, IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    for i, raw in enumerate(png_bytes):
        with Image.open(io.BytesIO(raw)) as im:
            if im.mode != "L":                           # 统一灰度（原图本来就是 L）
                im = im.convert("L")
            if im.size != (IMG_SIZE, IMG_SIZE):          # 保险：尺寸不对就 resize
                im = im.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.LANCZOS)
            X[i] = np.asarray(im, dtype=np.uint8)
    return X, labels


def main() -> None:
    print("=" * 78)
    print("阶段 1.1：获取 FER2013（HuggingFace 镜像，官方三分划）")
    print("=" * 78)
    print(f"  项目根目录：{ROOT}")
    print(f"  数据源：{MIRROR}/datasets/{REPO}")
    print(f"  七类标签：{CLASS_NAMES}")
    print()

    # ① 下载三个 split
    print("① 下载")
    paths = {split: download(split) for split in FILES}

    # ② 若 npz 已存在且没带 --rebuild，直接校验后退出（避免每次重复转换）
    if NPZ_PATH.exists() and not FORCE:
        with np.load(NPZ_PATH) as z:
            print("\n② 转换：fer2013.npz 已存在，跳过（要重建请加 --rebuild）")
            print(f"   已存 splits：train={len(z['y_train'])} val={len(z['y_val'])} test={len(z['y_test'])}")
        return

    # ③ 转成 npz（numpy 压缩数组：后续 Dataset 只依赖 numpy，不依赖 parquet/pyarrow）
    print("\n② 解码 parquet → numpy 数组")
    data = {}
    stats = {}
    for p_split, (_, _, my_split) in FILES.items():
        t0 = time.time()
        X, y = load_split(paths[p_split])
        key = {"train": "train", "val": "val", "test": "test"}[my_split]
        data[f"X_{key}"], data[f"y_{key}"] = X, y
        counts = np.bincount(y, minlength=len(CLASS_NAMES))
        stats[key] = {"n": int(len(y)), "counts": counts.tolist()}
        print(f"  {p_split:11s} → {key:5s}: X={X.shape} y={y.shape} dtype={X.dtype}  "
              f"用时 {time.time()-t0:.1f}s")
        print("     每类数量: " + "  ".join(f"{c}={n}" for c, n in zip(CLASS_NAMES, counts)))

    # ④ 保存 + 自检
    print("\n③ 保存与自检")
    NPZ_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(NPZ_PATH, **data)
    size_mb = NPZ_PATH.stat().st_size / 1e6
    print(f"  [ok] 写出 {NPZ_PATH.name}（{size_mb:.1f} MB）")

    with np.load(NPZ_PATH) as z:
        assert z["X_train"].shape == (28709, 48, 48), "训练集尺寸不对！"
        assert z["X_val"].shape == (3589, 48, 48), "验证集尺寸不对！"
        assert z["X_test"].shape == (3589, 48, 48), "测试集尺寸不对！"
        total = len(z["y_train"]) + len(z["y_val"]) + len(z["y_test"])
        assert total == 35887, f"总数不对：{total}（应为 35887）"
        for key in ("X_train", "X_val", "X_test"):
            assert z[key].min() >= 0 and z[key].max() <= 255, "像素值越界！"
        # 标签必须都落在 0~6
        for key in ("y_train", "y_val", "y_test"):
            assert z[key].min() == 0 and z[key].max() == 6, "标签越界！"
    print(f"  [ok] 自检通过：train 28709 + val 3589 + test 3589 = {total} 张，像素 0~255，标签 0~6")

    # ⑤ 元信息落盘，方便以后写报告引用
    meta = {
        "source": f"{MIRROR}/datasets/{REPO}",
        "mirror": MIRROR,
        "repo": REPO,
        "original_dataset": "FER2013（Kaggle, 7 类 48×48 灰度）",
        "label_map": {str(i): name for i, name in enumerate(CLASS_NAMES)},
        "splits": {"train": "官方 Training", "val": "官方 PublicTest", "test": "官方 PrivateTest"},
        "sizes": {"train": 28709, "val": 3589, "test": 3589},
        "class_counts": stats,
        "npz_keys": ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"],
        "note": "X 为 uint8 灰度 48×48，y 为 int64（0~6）；npz 用 np.load 读取，仅依赖 numpy",
    }
    META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  [ok] 写出 {META_PATH.name}")

    print("\n" + "=" * 78)
    print("结论：FER2013 就位 ✅  下一步 → 03_dataset.py（自写 Dataset）")
    print("=" * 78)


if __name__ == "__main__":
    main()
