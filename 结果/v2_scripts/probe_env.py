# -*- coding: utf-8 -*-
"""v2 环境小探针：EMA 模块属性 + 强增强速度基准（不写任何项目产物）。"""
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
import importlib.util

import numpy as np
import timm
import torch
from timm.utils import ModelEmaV3

m = timm.create_model("resnet18", num_classes=7)
e = ModelEmaV3(m, decay=0.999, use_warmup=False, foreach=False)
print("EMA has module:", hasattr(e, "module"), type(getattr(e, "module", None)).__name__)
e.update(m)
print("EMA update OK")

spec = importlib.util.spec_from_file_location("fer_dataset", ROOT / "代码" / "03_dataset.py")
ds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ds)

X = (np.random.rand(200, 128, 128) * 255).astype(np.uint8)
y = np.random.randint(0, 7, 200)
for aug in ("basic", "strong"):
    d = ds.FERTransferDataset(X, y, out_size=128, augment=True, aug=aug)
    t0 = time.time()
    for i in range(200):
        d[i]
    print(f"{aug}: {(time.time()-t0)/200*1000:.2f} ms/image（128x128）")

X2 = (np.random.rand(100, 160, 160) * 255).astype(np.uint8)
y2 = np.random.randint(0, 7, 100)
for aug in ("basic", "strong"):
    d = ds.FERTransferDataset(X2, y2, out_size=160, augment=True, aug=aug)
    t0 = time.time()
    for i in range(100):
        d[i]
    print(f"{aug}: {(time.time()-t0)/100*1000:.2f} ms/image（160x160）")

# 顺带看 GPU 单 batch 前向+反向耗时（resnet18@128, AMP on/off）
sys.path.insert(0, str(ROOT / "代码"))
from torch.amp import autocast, GradScaler

dev = torch.device("cuda")
for name, size, bs in (("resnet18", 128, 64), ("resnet18", 160, 64), ("resnet50", 128, 64),
                       ("resnet34", 128, 64), ("efficientnet_b0", 160, 64)):
    net = timm.create_model(name, pretrained=False, num_classes=7).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    xb = torch.randn(bs, 3, size, size, device=dev)
    yb = torch.randint(0, 7, (bs,), device=dev)
    crit = torch.nn.CrossEntropyLoss()

    def step(amp):
        opt.zero_grad(set_to_none=True)
        t0 = time.time()
        with autocast("cuda", dtype=torch.float16, enabled=amp):
            out = net(xb)
        loss = crit(out.float(), yb)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        return (time.time() - t0) * 1000

    for _ in range(3):
        step(True)
    t_amp = np.median([step(True) for _ in range(5)])
    t_f32 = np.median([step(False) for _ in range(5)])
    print(f"GPU step {name}@{size} bs{bs}: amp={t_amp:.1f}ms  fp32={t_f32:.1f}ms")
    del net, opt
    torch.cuda.empty_cache()

print("GPU 显存 used:", torch.cuda.max_memory_allocated() / 1e9, "GB (峰值)")
