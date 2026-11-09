# -*- coding: utf-8 -*-
r"""
03_dataset.py —— 阶段 1.2~1.4：FER2013 数据管线（自写 Dataset，不依赖 torchvision）

为什么自己写 Dataset？
  本机 torchvision 会被 Smart App Control 间歇性拦截（见 PROJECT.md 第 3.2 节），
  所以核心代码只用 torch + numpy + PIL，torchvision 一律不引入。

本文件提供（给 05_train.py / 06_evaluate.py 复用）：
  FERDataset          自写 Dataset：读 numpy 数组 → 增强（可选）→ 归一化 → tensor
  build_dataloaders   一次性构造 train / val / test 三个 DataLoader
  compute_class_weights  按类别频率算损失权重（缓解 disgust 类别样本太少的问题）

直接运行本文件时会做 4 件事（对应阶段 1.2 / 1.3 / 1.4 的验收）：
  ① 读 数据\fer2013\fer2013.npz，报三个 split 的形状；
  ② 统计并打印每类样本分布（打印不均衡有多严重）；
  ③ 用 DataLoader 真实跑 1 个 epoch，测速并检查 batch 形状；
  ④ 生成增强前后对比图 结果\aug_preview.png，人眼确认增强没把图搞坏。

运行（PowerShell）：
    $env:PYTHONIOENCODING='utf-8'
    python <项目根>/代码/03_dataset.py
"""
import json
import os
import pathlib
import sys
import time

# 本机环境里 conda 和 torch 各带一份 Intel OpenMP 运行时（libiomp5md.dll，版本还不同），
# torch 与 matplotlib 同时出现在一个进程时，第一次画图会报 "OMP: Error #15"。
# 这个环境变量让 OpenMP 允许重复加载（Windows 本地跑深度学习时的常见做法）。
# 必须写在 import numpy / torch 之前！
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader, Dataset

# Windows 控制台默认 GBK，打印 ✅/❌ 会抛 UnicodeEncodeError；统一改成 UTF-8 输出
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------- 路径：全部相对本文件定位 ----------
ROOT = pathlib.Path(__file__).resolve().parents[1]
FER_DIR = ROOT / "数据" / "fer2013"
NPZ_PATH = FER_DIR / "fer2013.npz"
NORM_JSON = FER_DIR / "norm_stats.json"       # 归一化用的 mean/std（训练集统计，训练/推理共用）
AUG_PREVIEW = ROOT / "结果" / "aug_preview.png"

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
IMG_SIZE = 48
# ImageNet 的归一化均值/标准差：预训练模型（ResNet 等）训练时用的就是这两个数，
# 迁移学习时输入必须按同样方式归一化，否则预训练特征会"水土不服"。
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def augment_gray_image(img: Image.Image, pad: int = 4) -> Image.Image:
    """灰度正方形图的数据增强（只用 PIL，不依赖 torchvision），训练时用：
    ① 随机水平翻转（p=0.5）—— 表情左右对称，翻转让模型不依赖朝向；
    ② 随机平移 —— 四周 pad 若干像素后随机裁回原尺寸，模拟人脸位置偏移；
    ③ 随机旋转 ±10° —— 模拟脑袋歪一点；
    ④ 亮度随机 ±10% —— 模拟不同光照。

    pad 参数：48×48 用 4；迁移学习放大到 128×128 后按比例用 ~11（由调用方传入）。
    """
    if np.random.rand() < 0.5:
        img = ImageOps.mirror(img)                                   # ① 水平翻转
    w, h = img.size
    big = Image.new("L", (w + 2 * pad, h + 2 * pad), 0)             # ② 随机平移
    big.paste(img, (pad, pad))
    off_x = int(np.random.randint(0, 2 * pad + 1))
    off_y = int(np.random.randint(0, 2 * pad + 1))
    img = big.crop((off_x, off_y, off_x + w, off_y + h))
    angle = np.random.uniform(-10.0, 10.0)
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)  # ③ 旋转（空处填黑）
    factor = np.random.uniform(0.9, 1.1)
    img = ImageEnhance.Brightness(img).enhance(factor)               # ④ 亮度
    return img


def strong_augment_gray_image(img: Image.Image) -> Image.Image:
    """强增强（08_transfer_train.py 的 --aug strong 用；默认路径 augment_gray_image 保持不变）。

    在 basic 增强的基础上加强（表情识别常用的"重增强"组合）：
      ① 随机水平翻转（p=0.5）
      ② 随机旋转 ±15°
      ③ 随机缩放 0.85~1.15
      ④ 随机平移 ±10%
      ⑤ 随机亮度 ±20%
      ⑥ 随机对比度 ±20%
    ②③④ 合并成一次 AFFINE 变换（只做一次重采样，比分开做快）；空出来的边缘填 0（黑）。
    随机擦除不在 PIL 阶段做，而是在 FERTransferDataset 归一化后的张量上做（见 random_erase_tensor）。
    """
    if np.random.rand() < 0.5:
        img = ImageOps.mirror(img)                                     # ① 水平翻转
    w, h = img.size
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    angle = np.deg2rad(np.random.uniform(-15.0, 15.0))                 # ② 旋转 ±15°
    scale = np.random.uniform(0.85, 1.15)                              # ③ 缩放 0.85~1.15
    tx = np.random.uniform(-0.10, 0.10) * w                            # ④ 平移 ±10%
    ty = np.random.uniform(-0.10, 0.10) * h
    ca, sa = np.cos(angle), np.sin(angle)
    # 输出像素 p_out 反查输入像素 p_in = M @ p_out + v，M = (1/s)·R(-angle)
    m00, m01 = ca / scale, sa / scale
    m10, m11 = -sa / scale, ca / scale
    v0 = cx - (m00 * (cx + tx) + m01 * (cy + ty))
    v1 = cy - (m10 * (cx + tx) + m11 * (cy + ty))
    img = img.transform((w, h), Image.AFFINE, (m00, m01, v0, m10, m11, v1),
                        resample=Image.Resampling.BILINEAR)            # ②③④ 一次完成
    img = ImageEnhance.Brightness(img).enhance(float(np.random.uniform(0.8, 1.2)))   # ⑤ 亮度
    img = ImageEnhance.Contrast(img).enhance(float(np.random.uniform(0.8, 1.2)))     # ⑥ 对比度
    return img


def random_erase_tensor(x: torch.Tensor, p: float = 0.3,
                        edge_frac: tuple[float, float] = (0.08, 0.20)) -> torch.Tensor:
    """随机擦除（强增强的最后一步，在**已 ImageNet 归一化**的 (3,H,W) 张量上做）。

    p=0.3 概率触发；擦除块是正方形，边长取图像边长的 8%~20%；擦除区域填 0
    （归一化后 0 正好等于 ImageNet 均值色，即"平均灰"，不是纯黑，避免引入异常暗块）。
    """
    if np.random.rand() >= p:
        return x
    _, h, w = x.shape
    eh = max(1, int(round(h * float(np.random.uniform(*edge_frac)))))
    ew = max(1, int(round(w * float(np.random.uniform(*edge_frac)))))
    top = int(np.random.randint(0, h - eh + 1))
    left = int(np.random.randint(0, w - ew + 1))
    x = x.clone()
    x[:, top:top + eh, left:left + ew] = 0.0
    return x


def load_fer2013(labels: str = "fer2013") -> dict:
    """读取 fer2013.npz，返回 {'train': (X,y), 'val': ..., 'test': ...}，X 是 (N,48,48) uint8。

    labels="fer2013"（默认）：原始 FER2013 标签，行为与改动前完全一致；
    labels="ferplus"：训练/验证/测试标签换成 FER+ 众包硬标签（代码\\18_ferplus_labels.py 生成），
                      并丢弃 FER+ 中所有票都是 unknown/NF 的样本（记录在 meta.json）。
    """
    if not NPZ_PATH.exists():
        raise SystemExit(f"找不到 {NPZ_PATH}，请先运行 00_download_fer2013.py")
    with np.load(NPZ_PATH) as z:
        data = {split: (z[f"X_{split}"], z[f"y_{split}"].astype(np.int64)) for split in ("train", "val", "test")}
    if labels == "fer2013":
        return data
    if labels != "ferplus":
        raise SystemExit(f"未知 labels：{labels}（可选 fer2013 / ferplus）")
    fplus_path = FER_DIR / "ferplus_labels.npz"
    if not fplus_path.exists():
        raise SystemExit(f"找不到 {fplus_path}，请先运行 代码\\18_ferplus_labels.py")
    with np.load(fplus_path) as z:
        out = {}
        for split in ("train", "val", "test"):
            X, _ = data[split]
            valid = z[f"valid_{split}"]
            y = z[f"y_hard_{split}"].astype(np.int64)[valid]
            X = X[valid]
            assert len(X) == len(y)
            # 快捷校验：FER+ 有效标签必须在 0~6
            assert int(y.min()) >= 0 and int(y.max()) < len(CLASS_NAMES), "FER+ 标签越界"
            out[split] = (X, y)
        return out


def calc_norm_stats(X: np.ndarray) -> tuple[float, float]:
    """算训练集的像素均值/标准差（先除 255 变 0~1，再求统计量）。"""
    x = X.astype(np.float32) / 255.0
    return float(x.mean()), float(x.std())


class FERDataset(Dataset):
    """FER2013 数据集。

    __getitem__ 返回 (x, y)：
      x: float32 tensor，形状 (1, 48, 48)，已归一化
      y: int64 标量 tensor，取值 0~6
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, mean: float, std: float, augment: bool = False):
        assert len(X) == len(y), "X 与 y 数量不一致"
        self.X, self.y = X, y
        self.mean, self.std = mean, std
        self.augment = augment

    def __len__(self) -> int:
        return len(self.y)

    def _augment(self, img: Image.Image) -> Image.Image:
        """数据增强：直接复用模块级的 augment_gray_image（保证和迁移学习用的是同一套增强）。"""
        return augment_gray_image(img)

    def __getitem__(self, idx: int):
        img = Image.fromarray(self.X[idx], mode="L")   # uint8 (48,48) → PIL 灰度图
        if self.augment:
            img = self._augment(img)
        arr = np.asarray(img, dtype=np.float32) / 255.0            # 0~255 → 0~1
        arr = (arr - self.mean) / self.std                         # 标准化（和训练时一致）
        x = torch.from_numpy(arr).unsqueeze(0)                     # (48,48) → (1,48,48)，即 1 个灰度通道
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y


class FERTransferDataset(Dataset):
    """给 ImageNet 预训练模型（ResNet 等）用的 FER2013 Dataset。

    和 FERDataset 的区别（迁移学习必须做这三件事）：
      ① 输入尺寸大（默认 128×128）：预训练模型是在大图上学的，48×48 太小、特征对不上；
         （放大在 build_transfer_loaders 里一次性做完，避免每个 epoch 重复 resize 浪费时间）
      ② 灰度复制成 3 通道：预训练第一层卷积是 3 通道，尺寸不匹配就加载不了权重；
      ③ 用 ImageNet 的 mean/std 归一化（而不是 FER 自己的统计量），保持和预训练一致。

    X 必须是已经 resize 到 out_size 的 uint8 数组（由 build_transfer_loaders 预处理）。
    __getitem__ 返回 (x, y)：x 是 (3, out_size, out_size) 的 float32 tensor，y 是 0~6 的标签。

    aug: "basic"（默认，augment_gray_image：翻转/平移/旋转±10°/亮度±10%）
         "strong"（08_transfer_train.py --aug strong：翻转/旋转±15°/缩放/平移±10%/亮度±20%/对比度±20% + 随机擦除）
    """

    def __init__(self, X_resized: np.ndarray, y: np.ndarray, out_size: int = 128,
                 augment: bool = False, aug: str = "basic"):
        assert len(X_resized) == len(y), "X 与 y 数量不一致"
        assert X_resized.shape[1] == out_size, f"X 应已 resize 到 {out_size}，实际 {X_resized.shape[1]}"
        assert aug in ("basic", "strong"), f"未知 aug 类型：{aug}"
        self.X, self.y = X_resized, y
        self.out_size = out_size
        self.augment = augment
        self.aug = aug
        # 增强的平移幅度按尺寸等比例放大：48→4，128→11
        self.aug_pad = max(2, round(out_size * 4 / IMG_SIZE))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        img = Image.fromarray(self.X[idx], mode="L")        # 已 resize 好的 uint8 灰度图
        if self.augment:
            if self.aug == "strong":
                img = strong_augment_gray_image(img)        # 强增强（含随机擦除前的几何/光度变换）
            else:
                img = augment_gray_image(img, pad=self.aug_pad)  # basic 增强（平移幅度按尺寸放大）
        arr = np.asarray(img, dtype=np.float32) / 255.0     # 0~255 → 0~1，形状 (H,W)
        arr = np.repeat(arr[None, :, :], 3, axis=0)         # 灰度复制成 3 通道：(1,H,W) → (3,H,W)
        mean = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
        arr = (arr - mean) / std                            # ImageNet 归一化
        x = torch.from_numpy(arr)
        if self.augment and self.aug == "strong":
            x = random_erase_tensor(x)                      # 随机擦除（张量上，p=0.3）
        y = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return x, y


def resize_split(X: np.ndarray, out_size: int) -> np.ndarray:
    """把 (N,48,48) uint8 一次性放大到 (N,out_size,out_size) uint8（迁移学习用）。

    为什么要一次性做：PIL 的 LANCZOS resize 每张 ~2ms，28709 张如果每个 epoch 都做，
    光 resize 就要 ~60s/epoch；这里只做一次（约 40~60s），之后每个 epoch 只做增强。
    """
    if X.shape[1] == out_size:
        return X
    out = np.empty((len(X), out_size, out_size), dtype=np.uint8)
    for i, x in enumerate(X):
        out[i] = np.asarray(Image.fromarray(x, mode="L").resize((out_size, out_size),
                                                                Image.Resampling.LANCZOS))
    return out


def get_or_compute_norm_stats(train_X: np.ndarray) -> tuple[float, float]:
    """归一化统计量：第一次算出来就存 json，之后直接复用（保证训练/演示/评估完全一致）。"""
    if NORM_JSON.exists():
        info = json.loads(NORM_JSON.read_text(encoding="utf-8"))
        return float(info["mean"]), float(info["std"])
    mean, std = calc_norm_stats(train_X)
    NORM_JSON.write_text(json.dumps({"mean": mean, "std": std, "source": "fer2013 train split"}, indent=2),
                         encoding="utf-8")
    return mean, std


def build_dataloaders(batch_size: int = 128, num_workers: int = 0):
    """给训练/评估用的工厂函数：返回 (train_loader, val_loader, test_loader, class_weights, stats)。

    num_workers=0 表示在主进程里取数据：Windows 下最稳，不会有多进程启动问题。
    """
    data = load_fer2013()
    mean, std = get_or_compute_norm_stats(data["train"][0])

    # 训练集开增强，验证/测试集绝不开增强（评估必须用"原图"）
    ds_train = FERDataset(*data["train"], mean=mean, std=std, augment=True)
    ds_val = FERDataset(*data["val"], mean=mean, std=std, augment=False)
    ds_test = FERDataset(*data["test"], mean=mean, std=std, augment=False)

    loaders = {
        "train": DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            pin_memory=torch.cuda.is_available(), drop_last=False),
        "val": DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                          pin_memory=torch.cuda.is_available()),
        "test": DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=torch.cuda.is_available()),
    }

    # 类别权重：总数 / (类别数 × 该类样本数)。样本越少权重越大，缓解不均衡。
    counts = np.bincount(data["train"][1], minlength=len(CLASS_NAMES)).astype(np.float64)
    weights = counts.sum() / (len(CLASS_NAMES) * np.maximum(counts, 1))
    return loaders, torch.tensor(weights, dtype=torch.float32), {"mean": mean, "std": std}


def build_transfer_loaders(batch_size: int = 64, num_workers: int = 0, out_size: int = 128,
                           aug: str = "basic", persistent_workers: bool | None = None,
                           labels: str = "fer2013", merge_val: bool = False,
                           val_holdout: int = 0, split_seed: int = 42):
    """迁移学习专用的 DataLoader 工厂：返回 (loaders, class_weights, stats)。

    训练集开增强、验证/测试集不开增强；归一化用 ImageNet 统计量（见 FERTransferDataset）。
    图片会在这里**一次性**放大到 out_size（而不是每个 epoch 重新 resize），能省掉大量时间。
    aug: "basic"（默认，与改动前完全一致）/ "strong"（08_transfer_train.py --aug strong）；
    labels: "fer2013"（默认，原标签）/ "ferplus"（FER+ 众包标签，A 快筛用；默认行为不变）。
    merge_val: True 时把 publicTest(val) 并入训练集（A 方案扩训练集）；此时必须给 val_holdout>0，
               从合并后的数据里**按类别分层**留出 val_holdout 张做早停/选轮，其余全部训练。
               默认 False，行为与原版完全一致。
    """
    data = load_fer2013(labels=labels)
    merge_info = {}
    if merge_val:
        if labels != "fer2013":
            raise SystemExit("merge_val=True 目前只支持 labels='fer2013'")
        n_add = len(data["val"][1])
        X_all = np.concatenate([data["train"][0], data["val"][0]], axis=0)
        y_all = np.concatenate([data["train"][1], data["val"][1]])
        if val_holdout <= 0:
            raise SystemExit("merge_val=True 时必须给 val_holdout>0（用于早停/选轮）")
        rng = np.random.RandomState(split_seed)
        hold_parts = []
        for c in range(len(CLASS_NAMES)):
            idx = np.flatnonzero(y_all == c)
            rng.shuffle(idx)
            k = max(1, int(round(len(idx) * val_holdout / len(y_all))))
            hold_parts.append(idx[:k])
        hold_idx = np.concatenate(hold_parts)
        hold_mask = np.zeros(len(y_all), dtype=bool)
        hold_mask[hold_idx] = True
        data["train"] = (X_all[~hold_mask], y_all[~hold_mask])
        data["val"] = (X_all[hold_mask], y_all[hold_mask])
        merge_info = {"merge_val": True, "added_from_public_test": int(n_add),
                      "train_size": int(len(data["train"][1])),
                      "val_holdout_size": int(len(data["val"][1])),
                      "split_seed": int(split_seed)}
        print(f"  A 方案扩训练集：并入 publicTest {n_add} 张 → train {merge_info['train_size']} 张；"
              f"分层留出 {merge_info['val_holdout_size']} 张做早停（seed={split_seed}）")
        print(f"    留出集类别分布：{np.bincount(data['val'][1], minlength=len(CLASS_NAMES)).tolist()}")
    print(f"  预处理：把图片一次性放大到 {out_size}×{out_size}（只做一次；labels={labels}）…")
    t0 = time.time()
    Xr = {split: resize_split(data[split][0], out_size) for split in ("train", "val", "test")}
    print(f"  放大完成，用时 {time.time()-t0:.1f}s；train={Xr['train'].shape}")

    ds_train = FERTransferDataset(Xr["train"], data["train"][1], out_size=out_size,
                                   augment=True, aug=aug)
    ds_val = FERTransferDataset(Xr["val"], data["val"][1], out_size=out_size, augment=False)
    ds_test = FERTransferDataset(Xr["test"], data["test"][1], out_size=out_size, augment=False)

    # num_workers>0 时默认用 persistent_workers，避免每个 epoch 重新启动子进程/重新灌入整份数据
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    loaders = {
        "train": DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                            pin_memory=torch.cuda.is_available(), drop_last=False,
                            persistent_workers=bool(persistent_workers) if num_workers > 0 else False),
        "val": DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                          pin_memory=torch.cuda.is_available(),
                          persistent_workers=bool(persistent_workers) if num_workers > 0 else False),
        "test": DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                           pin_memory=torch.cuda.is_available(),
                           persistent_workers=bool(persistent_workers) if num_workers > 0 else False),
    }
    return loaders, compute_class_weights(data["train"][1]), {
        "mean": IMAGENET_MEAN, "std": IMAGENET_STD, "out_size": out_size, **merge_info,
    }


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """单独暴露一个函数：只根据标签数组算类别权重（方便测试/复用）。"""
    counts = np.bincount(y, minlength=len(CLASS_NAMES)).astype(np.float64)
    weights = counts.sum() / (len(CLASS_NAMES) * np.maximum(counts, 1))
    return torch.tensor(weights, dtype=torch.float32)


def main() -> None:
    print("=" * 82)
    print("阶段 1.2~1.4：FER2013 数据管线（自写 Dataset，不依赖 torchvision）")
    print("=" * 82)
    print(f"  数据文件：{NPZ_PATH}")
    print(f"  torch：{torch.__version__}  CUDA 可用：{torch.cuda.is_available()}")

    # ① 读数据
    print("\n① 读取 npz")
    data = load_fer2013()
    for split, (X, y) in data.items():
        print(f"  {split:5s}: X={X.shape} dtype={X.dtype} | y={y.shape} dtype={y.dtype} "
              f"| 像素范围 [{X.min()}, {X.max()}]")

    # ② 类别分布（阶段 1.4 的验收：要能看到不均衡）
    print("\n② 类别分布（训练集不均衡程度）")
    train_counts = np.bincount(data["train"][1], minlength=len(CLASS_NAMES))
    total = train_counts.sum()
    print(f"  {'类别':<10}{'训练集':>8}{'占比':>8}   柱状图")
    for i, name in enumerate(CLASS_NAMES):
        n = int(train_counts[i])
        bar = "█" * max(1, round(n / 400))
        print(f"  {name:<10}{n:>8}{n/total*100:>7.1f}%   {bar}")
    print(f"  合计：{total} 张")
    print(f"  ⚠ 最少的 disgust 只有 {int(train_counts[1])} 张，是最多的 happy（{int(train_counts[3])} 张）的 "
          f"1/{int(train_counts[3]/max(train_counts[1],1))} → 训练时要处理不均衡")

    # ③ 真跑一个 epoch 测速（阶段 1.2 的验收：能 for batch in loader 跑通）
    print("\n③ 构造 DataLoader 并真实跑 1 个 epoch")
    loaders, class_weights, stats = build_dataloaders(batch_size=128, num_workers=0)
    print(f"  归一化统计量 mean={stats['mean']:.4f} std={stats['std']:.4f}（已写入 {NORM_JSON.name}）")
    print(f"  类别权重 class_weights = {[round(float(w), 3) for w in class_weights]}")

    xb, yb = next(iter(loaders["train"]))
    print(f"  第一个 batch: x={tuple(xb.shape)} {xb.dtype} 范围[{xb.min():.2f}, {xb.max():.2f}] | "
          f"y={tuple(yb.shape)} {yb.dtype}")

    t0 = time.time()
    n_seen = 0
    for xb, yb in loaders["train"]:
        n_seen += len(yb)
        assert xb.shape == (len(yb), 1, IMG_SIZE, IMG_SIZE), f"batch 形状不对：{xb.shape}"
    train_dt = time.time() - t0
    print(f"  ✅ 1 个 epoch 跑通：{n_seen} 张，耗时 {train_dt:.2f}s（纯 CPU 读数据+增强）")

    # ④ 增强效果对比图（阶段 1.3 的验收：增强后可视化正常）
    print("\n④ 生成增强效果对比图")
    ds_aug = FERDataset(*data["train"], mean=stats["mean"], std=stats["std"], augment=True)
    # 找一张"happy"的图当例子（从训练集里查第一个 happy 的下标）
    idx = int(np.where(data["train"][1] == 3)[0][0])
    X, y = data["train"]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 5, figsize=(10, 4.4))
    axes = axes.ravel()
    axes[0].imshow(X[idx], cmap="gray")
    axes[0].set_title(f"original ({CLASS_NAMES[int(y[idx])]})", fontsize=9)
    axes[0].axis("off")
    for k in range(1, 10):
        x_t, _ = ds_aug[idx]                                     # 每次都会随机增强
        img = x_t[0].numpy() * stats["std"] + stats["mean"]      # 反归一化，还原成能看的像素
        axes[k].imshow(np.clip(img, 0, 1), cmap="gray")
        axes[k].set_title(f"aug #{k}", fontsize=9)
        axes[k].axis("off")
    fig.suptitle("FER2013 train image + random augmentations", fontsize=11)
    fig.tight_layout()
    AUG_PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(AUG_PREVIEW, dpi=120)
    plt.close(fig)
    print(f"  [ok] {AUG_PREVIEW}")

    print("\n" + "=" * 82)
    print("结论：数据管线就位 ✅")
    print("  验收：① npz 可读 28709/3589/3589  ② Dataset+DataLoader 跑通  ③ 增强可视化正常  ④ 分布已统计")
    print("  下一步 → 04_model.py（CNN 定义）")
    print("=" * 82)


if __name__ == "__main__":
    main()
