# -*- coding: utf-8 -*-
"""v2 数据管线成本探针：找出 43s/epoch 的瓶颈，评估 pin_memory / 批内归一化 / GPU 增强的收益。

只做测量，不写项目产物。与 3 个训练进程并行跑（CPU 24 核，影响很小）。
"""
import os
import pathlib
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "_deps"))
sys.path.insert(0, str(ROOT / "代码"))
import importlib.util

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

spec = importlib.util.spec_from_file_location("fer_dataset", ROOT / "代码" / "03_dataset.py")
ds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ds)

data = ds.load_fer2013()
X = ds.resize_split(data["train"][0][:6000], 128)
y = data["train"][1][:6000]
print(f"subset: {X.shape}, X.nbytes={X.nbytes/1e6:.0f}MB")

for aug in ("basic", "strong"):
    d = ds.FERTransferDataset(X, y, out_size=128, augment=True, aug=aug)
    for i in range(50):
        d[i]
    t0 = time.time()
    for i in range(3000):
        d[i % len(y)]
    dt = (time.time() - t0) / 3000
    print(f"getitem {aug}: {dt*1000:.3f} ms/img -> epoch(28709) {dt*28709:.1f}s")

# 拆解：PIL 增强 vs numpy 归一化
from PIL import Image

img0 = Image.fromarray(X[0], mode="L")
t0 = time.time()
for _ in range(3000):
    im = Image.fromarray(X[0], mode="L")
print(f"  Image.fromarray: {(time.time()-t0)/3000*1000:.3f} ms")
t0 = time.time()
for _ in range(3000):
    im = ds.strong_augment_gray_image(img0)
print(f"  strong aug (PIL): {(time.time()-t0)/3000*1000:.3f} ms")
t0 = time.time()
for _ in range(3000):
    arr = np.asarray(im, dtype=np.float32) / 255.0
    arr = np.repeat(arr[None, :, :], 3, axis=0)
    arr = (arr - np.float32(0.485)) / np.float32(0.229)
print(f"  numpy 归一化+3通道: {(time.time()-t0)/3000*1000:.3f} ms")
t0 = time.time()
for _ in range(3000):
    t = torch.from_numpy(arr)
print(f"  torch.from_numpy: {(time.time()-t0)/3000*1000:.3f} ms")

# DataLoader 整轮（pin_memory 开/关）
for aug in ("basic", "strong"):
    for pin in (True, False):
        d = ds.FERTransferDataset(X, y, out_size=128, augment=True, aug=aug)
        dl = DataLoader(d, batch_size=64, shuffle=True, num_workers=0,
                        pin_memory=pin and torch.cuda.is_available(), drop_last=False)
        t0 = time.time()
        nb = 0
        for xb, yb in dl:
            nb += 1
        dt = time.time() - t0
        print(f"DataLoader {aug} pin={pin}: {dt/len(y)*1000:.3f} ms/img ({nb} batches) -> epoch {dt/len(y)*28709:.1f}s")


# 若改成"Dataset 只返回 uint8 裸图 + GPU 上做归一化/增强"，数据侧成本是多少
class RawDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return torch.from_numpy(self.X[i][None, :, :].copy()), int(self.y[i])   # (1,128,128) uint8


for pin in (True, False):
    d = RawDataset(X, y)
    dl = DataLoader(d, batch_size=64, shuffle=True, num_workers=0, pin_memory=pin)
    t0 = time.time()
    for xb, yb in dl:
        pass
    dt = time.time() - t0
    print(f"RawDataset pin={pin}: {dt/len(y)*1000:.3f} ms/img -> epoch {dt/len(y)*28709:.1f}s")

# GPU 侧：归一化+3通道复制 / 简化仿射增强 的耗时估计
if torch.cuda.is_available():
    dev = torch.device("cuda")
    xb = torch.randint(0, 255, (64, 1, 128, 128), device=dev, dtype=torch.uint8)
    def gpu_norm(x):
        x = x.float().div_(255.0)
        x = (x - torch.tensor(ds.IMAGENET_MEAN, device=dev).view(1, 3, 1, 1)
             ) / torch.tensor(ds.IMAGENET_STD, device=dev).view(1, 3, 1, 1)
        return x
    t0 = time.time()
    for _ in range(100):
        gpu_norm(xb)
    torch.cuda.synchronize()
    print(f"GPU 批内归一化(64x1x128x128): {(time.time()-t0)/100*1000:.3f} ms/batch")
    # grid_sample 仿射（64 张 128x128）
    x3 = torch.rand(64, 3, 128, 128, device=dev)
    theta = torch.zeros(64, 2, 3, device=dev)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    t0 = time.time()
    for _ in range(100):
        grid = torch.nn.functional.affine_grid(theta, x3.shape, align_corners=False)
        torch.nn.functional.grid_sample(x3, grid, align_corners=False)
    torch.cuda.synchronize()
    print(f"GPU grid_sample 仿射(64x3x128x128): {(time.time()-t0)/100*1000:.3f} ms/batch")
