# -*- coding: utf-8 -*-
r"""
08_transfer_train.py (v2) —— 任意 timm ImageNet 预训练模型微调 + 训练配方消融

和 v1 的区别（v1 只支持 resnet18/resnet34、只能 AdamW+余弦）：
  ① 支持任意 timm 模型名（resnet18/34/50、efficientnet_b0、convnext_tiny …）：
     权重仓库从 model.pretrained_cfg["hf_hub_id"] 通用取（如 timm/resnet50.a1_in1k），
     优先下 model.safetensors、404 回退 pytorch_model.bin → _deps\weights\{name}.{tag}.{ext}；
     加载时按"名字 + 形状"通用过滤（1000 类 ImageNet 分类头自动被跳过，不需要知道它叫 fc 还是 classifier）。
  ② 新增训练配方参数（全部默认关，旧的命令行仍然能跑、默认行为不变）：
     --amp               混合精度（训练 fp16 autocast + GradScaler，验证始终 fp32）
     --optimizer          adamw / lion（timm.optim.Lion）
     --warmup-epochs      前 N 轮学习率线性升到 --lr 再余弦退火
     --grad-clip          梯度范数裁剪（0=关）
     --ema-decay          timm.utils.ModelEmaV3（每个 step 更新；每轮同时评 raw 与 EMA，保存较优者）
     --swa-last           训练末尾平均最后 K 轮权重，评估一次，比现有最佳好才采用
     --mixup              mixup alpha（>0 开；软标签交叉熵，兼容 label smoothing）
     --balance            none/weights/sampler/focal（focal γ=1.0，可叠加类别权重）
     --aug                basic（默认，和 v1 完全一致）/ strong（03_dataset.py 的强增强）
     --patience 默认从 5 改成 10
  ③ 每个实验结束把一行记录写入 结果\v2_records\{name}.json，并自动重建
     结果\experiments_v2.jsonl + 结果\v2_experiment_summary.md（带文件锁，允许并跑多个实验）。
     权重包字段与 v1 完全兼容（06_evaluate.py / 07_demo_app.py 无需改动），额外加了 ema/source 字段。

注意：本脚本**只用验证集（val）选模型**，从不调用 test。

运行（PowerShell）：
    $env:PYTHONIOENCODING='utf-8'
    python <项目根>/代码/08_transfer_train.py ^
        --model resnet18 --img-size 128 --epochs 30 --amp --name v2_r1_amp
    # 旧命令仍然可用（默认 adamw / basic aug / 无 AMP）：
    ... 08_transfer_train.py --epochs 20 --img-size 128 --no-weight --name best_transfer
"""
import os
import pathlib
import sys

# 本机 conda 与 torch 各带一份 Intel OpenMP；torch + matplotlib 一起用会报 OMP Error #15，
# 这个环境变量允许重复加载（详见 PROJECT.md 第 3.2 节第 5 条）。必须在 import torch 之前设！
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# timm 的预训练权重走 HuggingFace 镜像下载（直连 huggingface.co 会超时）
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Windows 控制台默认 GBK，打印 ✅/❌ 会抛 UnicodeEncodeError；统一改成 UTF-8 输出
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import collections
import importlib.util
import json
import math
import re
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "代码"
MODEL_DIR = ROOT / "模型"
RESULT_DIR = ROOT / "结果"
RECORD_DIR = RESULT_DIR / "v2_records"          # 每个实验一行记录（json），聚合文件由它重建
EXPERIMENTS_JSONL = RESULT_DIR / "experiments_v2.jsonl"
SUMMARY_MD = RESULT_DIR / "v2_experiment_summary.md"

# timm / huggingface_hub / safetensors 装在本项目的 _deps 里（沙箱不允许改全局环境），
# 所以先把 _deps 加进 sys.path，再 import timm。
DEPS_DIR = ROOT / "_deps"
if str(DEPS_DIR) not in sys.path:
    sys.path.insert(0, str(DEPS_DIR))

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler

# 预训练权重直接按 URL 下载并缓存到这里（不走 huggingface_hub，避免它的缓存目录权限问题）
WEIGHTS_DIR = DEPS_DIR / "weights"


def load_module(name: str, filename: str):
    """按文件路径加载同目录下的脚本模块（文件名以数字开头，不能直接 import）。"""
    spec = importlib.util.spec_from_file_location(name, CODE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ds = load_module("fer_dataset", "03_dataset.py")     # 复用：build_transfer_loaders / strong 增强
md = load_module("fer_model", "04_model.py")         # 复用：get_device
# 注册到 sys.modules：DataLoader 用 num_workers>0 时子进程反序列化 Dataset 需要能按名字导入
sys.modules.setdefault("fer_dataset", ds)

CLASS_NAMES = ds.CLASS_NAMES


# ============================================================================
# 一、预训练权重：通用下载 + 通用加载（支持任意 timm 模型）
# ============================================================================

def pretrained_repo(model_name: str) -> tuple[str, str]:
    """返回 (hf 仓库 id, tag)，例如 ("timm/resnet50.a1_in1k", "a1_in1k")。"""
    probe = timm.create_model(model_name, pretrained=False)
    cfg = getattr(probe, "pretrained_cfg", None) or {}
    repo = cfg.get("hf_hub_id") or f"timm/{model_name}.a1_in1k"
    tag = repo.rsplit(".", 1)[-1] if "." in repo.rsplit("/", 1)[-1] else "a1_in1k"
    return repo, tag


def download_pretrained(model_name: str) -> pathlib.Path:
    r"""下载 timm 的 ImageNet 预训练权重，已存在（>10MB）就跳过。

    说明：正常可以用 timm.create_model(pretrained=True) 自动下载，但它依赖 huggingface_hub 的缓存目录，
    本机沙箱把 `<用户目录>/.cache` 写操作拦掉了；所以这里直接用 URL 下载到项目内 `_deps\weights\`。
    通用做法：仓库地址来自 model.pretrained_cfg["hf_hub_id"]；优先 model.safetensors，
    404 时回退 pytorch_model.bin；文件名 = {model_name}.{tag}.{ext}。
    """
    repo, tag = pretrained_repo(model_name)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    candidates = [WEIGHTS_DIR / f"{model_name}.{tag}.safetensors",
                  WEIGHTS_DIR / f"{model_name}.{tag}.bin"]
    for c in candidates:
        if c.exists() and c.stat().st_size > 10_000_000:
            print(f"  预训练权重已存在：{c.name}（{c.stat().st_size/1e6:.1f} MB）")
            return c

    errors = []
    for fname, suffix in (("model.safetensors", ".safetensors"), ("pytorch_model.bin", ".bin")):
        url = f"https://hf-mirror.com/{repo}/resolve/main/{fname}"
        out = WEIGHTS_DIR / f"{model_name}.{tag}{suffix}"
        print(f"  下载预训练权重：{url}")
        t0 = time.time()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=300) as r, open(out, "wb") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
        except Exception as e:                       # 404 / 超时 → 试下一个文件
            errors.append(f"{fname}: {e}")
            if out.exists() and out.stat().st_size < 10_000_000:
                out.unlink(missing_ok=True)
            print(f"  ⚠ {fname} 下载失败（{e}），尝试下一个")
            continue
        if not out.exists() or out.stat().st_size < 10_000_000:
            errors.append(f"{fname}: 文件过小")
            continue
        print(f"  下载完成：{out.stat().st_size/1e6:.1f} MB，用时 {time.time()-t0:.1f}s")
        return out
    raise SystemExit(f"❌ 预训练权重下载失败（{repo}）：{errors}")


def load_state_any(weights_path: pathlib.Path) -> dict:
    """读 safetensors 或 pytorch_model.bin，统一返回 state_dict（dict[str, Tensor]）。"""
    if weights_path.suffix == ".safetensors":
        return load_file(str(weights_path))
    obj = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    if not isinstance(obj, dict):
        raise SystemExit(f"❌ 无法从 {weights_path} 解析 state_dict")
    return obj


def build_pretrained_model(model_name: str, device: torch.device, drop_path: float = 0.0):
    """建任意 timm 模型（7 类输出），并加载 ImageNet 预训练权重。

    drop_path：随机深度（stochastic depth）比例，0=关（默认，保持旧行为）；建议 0.1~0.2。
    通用过滤规则：预训练权重里"名字存在于当前模型、且形状一致"的键才加载。
    ImageNet 分类头（1000 类）因为形状和 7 类对不上会被自动跳过，
    所以不需要像 v1 那样写死 `fc.`（不同 backbone 分类头可能叫 classifier / head / fc）。
    """
    model = timm.create_model(model_name, pretrained=False, num_classes=len(CLASS_NAMES),
                              drop_path_rate=float(drop_path) if drop_path and drop_path > 0 else 0.0)
    if drop_path and drop_path > 0:
        n_dp = sum(1 for m in model.modules() if hasattr(m, "drop_path_rate") or m.__class__.__name__.endswith("DropPath"))
        print(f"  随机深度 drop_path_rate={drop_path}（模型内 DropPath 模块约 {n_dp} 个）")
    weights_path = download_pretrained(model_name)
    state = load_state_any(weights_path)
    msd = model.state_dict()

    def _filter(st: dict) -> dict:
        return {k: v for k, v in st.items()
                if k in msd and tuple(v.shape) == tuple(msd[k].shape)}

    keep = _filter(state)
    # 少数权重仓库会多一层前缀（"model." / "module."），自动剥一层再试
    if len(keep) < max(1, len(msd) // 2):
        for prefix in ("model.", "module.", "backbone."):
            st2 = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            keep2 = _filter(st2)
            if len(keep2) > len(keep):
                print(f"  检测到权重前缀 '{prefix}'，已自动剥离")
                keep, state = keep2, st2
                break

    missing, unexpected = model.load_state_dict(keep, strict=False)
    print(f"  预训练权重：文件 {weights_path.name}，匹配加载 {len(keep)}/{len(msd)} 个参数张量；"
          f"缺失 {len(missing)} 个、多余 {len(unexpected)} 个")
    if missing:
        print(f"    缺失（预期只有分类头，7 类要重新学）：{missing[:6]}{' …' if len(missing) > 6 else ''}")
    if unexpected:
        print(f"    多余（预训练里有、当前模型没有）：{unexpected[:4]}{' …' if len(unexpected) > 4 else ''}")
    if len(missing) > 4:
        print("    ⚠ 除分类头外还有较多缺失参数（异常，请检查权重与结构是否匹配）")
    elif missing:
        print("    ✅ 除分类头外，所有预训练参数都成功加载（分类头下面用 FER2013 重新学）")
    else:
        print("    ✅ 全部预训练参数（含分类头形状一致）都成功加载")
    return model.to(device)


# ============================================================================
# 二、随机种子 / 评估 / 损失（支持类别权重、label smoothing、focal、mixup 软标签）
# ============================================================================

def set_seed(seed: int) -> None:
    """固定随机种子：数据增强、参数初始化、shuffle 都可复现。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_soft_target(y: torch.Tensor, num_classes: int, weight=None, label_smoothing: float = 0.0):
    """把硬标签 y 变成（可带类别权重、label smoothing 的）软标签分布 t，形状 (N,K)。"""
    t = F.one_hot(y, num_classes).float()
    if label_smoothing > 0:
        t = t * (1.0 - label_smoothing) + label_smoothing / num_classes
    if weight is not None:
        t = t * weight.unsqueeze(0)
    return t


def soft_loss(logits: torch.Tensor, t: torch.Tensor, gamma: float = 0.0) -> torch.Tensor:
    """软标签交叉熵；gamma>0 时是 focal loss（γ=1）。
    按 sum(t) 归一化，等价于 torch 带 weight 的 CE 归一化方式。"""
    logp = F.log_softmax(logits.float(), dim=1)
    per = -(t * logp)
    if gamma > 0:
        per = per * (1.0 - logp.exp()).pow(gamma)
    per = per.sum(dim=1)
    den = t.sum(dim=1).clamp_min(1e-8)
    return (per / den).mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader, device, weight=None, label_smoothing: float = 0.0,
             gamma: float = 0.0) -> dict:
    """在验证集上跑一遍（始终 fp32）：返回 loss、accuracy、macro-F1。"""
    model.eval()
    total_loss, n, correct = 0.0, 0, 0
    all_pred, all_true = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        t = build_soft_target(yb, logits.shape[1], weight, label_smoothing)
        total_loss += soft_loss(logits, t, gamma).item() * len(yb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        n += len(yb)
        all_pred.append(pred.cpu().numpy())
        all_true.append(yb.cpu().numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return {"loss": total_loss / n, "acc": correct / n,
            "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0)}


def train_one_epoch(model, loader, optimizer, scaler, device, cfg: dict, ema=None, epoch: int = 1,
                    n_epochs: int = 1) -> dict:
    """训练一轮：前向（AMP）→ 算 loss → 反向传播 → 梯度裁剪 → 更新参数 → EMA 更新。"""
    model.train()
    total_loss, n, correct = 0.0, 0, 0
    amp_ctx = torch.amp.autocast("cuda", dtype=torch.float16, enabled=cfg["amp"])
    weight = cfg["weight"]
    gamma = cfg["gamma"]
    ls = cfg["label_smoothing"]
    mixup = cfg["mixup"]
    n_batches = len(loader)
    t0 = time.time()
    for i, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        perm, lam = None, 1.0
        if mixup > 0:                                   # mixup：两张图线性混合，标签也线性混合
            perm = torch.randperm(xb.size(0), device=device)
            lam = float(np.random.beta(mixup, mixup))
            xb = lam * xb + (1.0 - lam) * xb[perm]
        optimizer.zero_grad(set_to_none=True)
        with amp_ctx:
            logits = model(xb)
        t = build_soft_target(yb, logits.shape[1], weight, ls)
        if perm is not None:
            t2 = build_soft_target(yb[perm], logits.shape[1], weight, ls)
            t = lam * t + (1.0 - lam) * t2
        loss = soft_loss(logits, t, gamma)              # loss 在 fp32 下算，数值稳定

        if scaler is not None:
            scaler.scale(loss).backward()
            if cfg["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
        if ema is not None:                             # EMA 逐 step 更新（decay 语义是"每次更新的衰减"）
            ema.update(model)

        total_loss += loss.item() * len(yb)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        n += len(yb)
        if (i + 1) % 200 == 0:
            print(f"    batch {i+1}/{n_batches}  loss={total_loss/n:.4f}  "
                  f"acc={correct/n:.4f}  {time.time()-t0:.1f}s", flush=True)
    return {"loss": total_loss / n, "acc": correct / n}


# ============================================================================
# 三、权重保存 / EMA / SWA
# ============================================================================

def save_checkpoint(path: pathlib.Path, state_dict: dict, args, epoch: int, val: dict, source: str) -> None:
    """保存权重包：字段与 v1 完全兼容（06/07 直接可用），额外加 ema/source 字段。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": state_dict,
        "profile": f"timm:{args.model}",           # 标记这是迁移学习模型
        "model_name": args.model,
        "img_size": args.img_size,
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "norm_mean": list(ds.IMAGENET_MEAN),
        "norm_std": list(ds.IMAGENET_STD),
        "epoch": epoch,
        "val_acc": val["acc"],
        "val_macro_f1": val["macro_f1"],
        "ema": source == "ema",
        "source": source,
        "labels": args.labels,                     # fer2013（默认）/ ferplus（P0 实验）
        "args": vars(args),
    }, path)


def make_ema(model: nn.Module, decay: float):
    """timm 的 ModelEmaV3：返回 EMA 对象（未开启返回 None）。"""
    if decay <= 0:
        return None
    from timm.utils import ModelEmaV3
    ema = ModelEmaV3(model, decay=decay, use_warmup=False)   # foreach 默认开，更新开销更小
    print(f"  EMA：ModelEmaV3 decay={decay}（逐 step 更新，每轮同时评估 raw 与 EMA）")
    return ema


def ema_module(ema):
    """取 EMA 里的那份模型（timm ModelEmaV3/V2 都是 .module）。"""
    if hasattr(ema, "module"):
        return ema.module
    return ema


def average_state_dicts(sds: list) -> dict:
    """SWA：对最后 K 轮的 state_dict 逐元素取平均（浮点参数/缓冲平均，整型缓冲取最后一轮）。"""
    avg = {}
    for k in sds[0].keys():
        vals = [sd[k] for sd in sds if k in sd]
        if len(vals) == 0:
            continue
        if vals[0].is_floating_point():
            avg[k] = torch.stack([v.detach().float().cpu() for v in vals], dim=0).mean(dim=0).to(vals[0].dtype)
        else:
            avg[k] = vals[-1].detach().cpu().clone()
    return avg


# ============================================================================
# 四、实验记录（每个实验一行 json + 自动重建 jsonl / summary.md，带文件锁可并发）
# ============================================================================

def _acquire_lock(lock_path: pathlib.Path, timeout: float = 20.0):
    """跨进程文件锁（多实验并跑时保护聚合文件）；拿到返回 True。"""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            try:
                if time.time() - lock_path.stat().st_mtime > 120:   # 陈旧锁自动清理
                    lock_path.unlink(missing_ok=True)
                    continue
            except Exception:
                pass
            time.sleep(0.1)
    return False


def rebuild_aggregates() -> None:
    r"""把 结果\v2_records\*.json 汇总成 experiments_v2.jsonl 与 v2_experiment_summary.md。"""
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    lock = RECORD_DIR / ".aggregate.lock"
    if not _acquire_lock(lock):
        print("  ⚠ 聚合文件锁被占用，本次跳过重建（记录已保存，稍后可重新触发）")
        return
    try:
        records = []
        for p in sorted(RECORD_DIR.glob("*.json")):
            try:
                records.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception as e:
                print(f"  ⚠ 读取记录失败 {p.name}：{e}")
        records.sort(key=lambda r: (r.get("group", ""), r.get("name", "")))
        tmp = EXPERIMENTS_JSONL.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
        os.replace(tmp, EXPERIMENTS_JSONL)
        SUMMARY_MD.write_text(build_summary_md(records), encoding="utf-8")
        print(f"  实验记录已更新：{EXPERIMENTS_JSONL.name}（{len(records)} 条）、{SUMMARY_MD.name}")
    finally:
        lock.unlink(missing_ok=True)


def build_summary_md(records: list) -> str:
    """生成按实验名列出 val 指标的 markdown 汇总表（Lead 直接引用）。"""
    lines = [
        "# 模型优化 v2 实验结果汇总（只统计验证集 val，未评估 test）",
        "",
        f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "- 选模型规则：按 **val_macro_f1 优先、val_acc 次之** 挑最佳 epoch；所有实验均未使用 test 集。",
        "- 权重目录：`模型\\{experiment_name}.pth`；训练历史：`结果\\train_{exp}.json`；曲线："
        "`结果\\training_curves_{exp}.png`；完整训练日志：`结果\\v2_logs\\{exp}.log`。",
        "",
    ]

    def one_table(rows: list) -> list:
        out = ["| 实验名 | 模型 | 输入 | 轮数(跑/设) | 优化器 | AMP | EMA | aug | mixup | balance | LS | "
               "**val_macro_f1** | **val_acc** | best_epoch | 耗时(min) | 权重来源 | 备注 |",
               "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            out.append(
                "| {name} | {model} | {img_size} | {epochs_run}/{epochs} | {optimizer} | {amp} | {ema} | "
                "{aug} | {mixup} | {balance} | {ls} | {f1:.4f} | {acc:.2%} | {ep} | {t:.1f} | {src} | {note} |".format(
                    name=r.get("name", ""), model=r.get("model", ""), img_size=r.get("img_size", ""),
                    epochs_run=r.get("epochs_run", ""), epochs=r.get("epochs", ""),
                    optimizer=r.get("optimizer", ""), amp="开" if r.get("amp") else "关",
                    ema=(f"{r.get('ema_decay')}" if r.get("ema_decay") else "关"),
                    aug=r.get("aug", ""), mixup=(r.get("mixup") or "关"),
                    balance=r.get("balance", ""), ls=r.get("label_smoothing", ""),
                    f1=float(r.get("best_val_macro_f1", 0.0)), acc=float(r.get("best_val_acc", 0.0)),
                    ep=r.get("best_epoch", ""), t=float(r.get("total_time_s") or 0.0) / 60.0,
                    src=r.get("best_source", ""), note=r.get("note", "") or "")
            )
        out.append("")
        return out

    groups = collections.OrderedDict()
    for r in records:
        groups.setdefault(r.get("group", "其他"), []).append(r)
    titles = {"R": "阶段 1：配方消融（resnet18@128, 30 epoch, AMP）",
              "P": "阶段 2：backbone 对比（阶段 1 最优配方）",
              "seed": "阶段 4：最优配置多种子（seed 42/43/44）",
              "其他": "其他实验"}
    for key, rows in groups.items():
        lines.append(f"## {titles.get(key, key)}")
        lines.append("")
        lines.extend(one_table(rows))

    if records:
        best = max(records, key=lambda r: (float(r.get("best_val_macro_f1", 0.0)),
                                           float(r.get("best_val_acc", 0.0))))
        lines.append("## 当前最优")
        lines.append("")
        lines.append(f"- **{best.get('name')}**（{best.get('model')}@{best.get('img_size')}，"
                     f"val_macro_f1={float(best.get('best_val_macro_f1', 0)):.4f}，"
                     f"val_acc={float(best.get('best_val_acc', 0)):.2%}，best_epoch={best.get('best_epoch')}）")
        lines.append("")
    lines.append("> 说明：v1 历史基线 ResNet18@128 20ep（fp32/basic aug，仅 val，参考）："
                 "val_acc 67.51% / macro-F1 0.6557。v2 全部实验均只做 val，test 集留给 Lead 最终决定后再评估。")
    lines.append("")
    return "\n".join(lines)


def write_experiment_record(args, history: list, best: dict, elapsed: float,
                            epochs_run: int, extra: dict) -> None:
    """写单个实验的记录 json，然后重建 jsonl + summary。"""
    name = args.name
    if name.startswith("v2_seed"):
        group = "seed"
    elif name.startswith("v2_resnet") or name.startswith("v2_eff"):
        group = "P"
    elif re.match(r"v2_r\d", name) or name.startswith("v2_r1") or name.startswith("v2_r2") \
            or name.startswith("v2_r3") or name.startswith("v2_r4"):
        group = "R"
    else:
        group = "其他"
    record = {
        "name": args.name, "group": group, "model": args.model, "img_size": args.img_size,
        "epochs": args.epochs, "epochs_run": epochs_run, "batch_size": args.batch_size,
        "seed": args.seed, "optimizer": args.optimizer, "lr": args.lr,
        "weight_decay": args.weight_decay, "amp": bool(args.amp),
        "ema_decay": args.ema_decay, "swa_last": args.swa_last,
        "aug": args.aug, "mixup": args.mixup, "balance": args.balance,
        "label_smoothing": args.label_smoothing, "warmup_epochs": args.warmup_epochs,
        "grad_clip": args.grad_clip, "patience": args.patience,
        "best_epoch": best["epoch"], "best_val_acc": best["val_acc"],
        "best_val_macro_f1": best["macro_f1"], "best_val_loss": best.get("val_loss"),
        "best_source": best.get("source", "raw"),
        "total_time_s": round(elapsed, 1), "note": args.note or "",
    }
    record.update(extra)
    RECORD_DIR.mkdir(parents=True, exist_ok=True)
    (RECORD_DIR / f"{args.name}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    rebuild_aggregates()


# ============================================================================
# 五、训练曲线
# ============================================================================

def plot_curves(history: list, name: str, path: pathlib.Path) -> None:
    """画训练曲线：loss / accuracy / 验证集 macro-F1 三张子图（有 EMA 就把 EMA 曲线一起画）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    has_ema = any("val_acc_ema" in h for h in history)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, [h["train_acc"] for h in history], label="train")
    axes[1].plot(epochs, [h["val_acc"] for h in history], label="val")
    if has_ema:
        axes[1].plot(epochs, [h.get("val_acc_ema", np.nan) for h in history], label="val (EMA)", ls="--")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    axes[2].plot(epochs, [h["val_macro_f1"] for h in history], color="tab:green", label="val")
    if has_ema:
        axes[2].plot(epochs, [h.get("val_macro_f1_ema", np.nan) for h in history],
                     color="tab:orange", ls="--", label="val (EMA)")
        axes[2].legend()
    axes[2].set_title("Validation Macro-F1")
    axes[2].set_xlabel("epoch")
    fig.suptitle(f"Transfer learning curves: {name}", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ============================================================================
# 六、主流程
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="ImageNet 预训练模型微调 FER2013（v2：任意 timm 模型 + 训练配方消融）")
    parser.add_argument("--model", type=str, default="resnet18",
                        help="任意 timm 模型名，如 resnet18/resnet34/resnet50/efficientnet_b0/convnext_tiny")
    parser.add_argument("--img-size", type=int, default=128, help="输入放大到多大（128 或 160/224）")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4, help="微调学习率（比从头训练小一个量级）")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--no-weight", action="store_true", help="关闭类别权重（仍兼容旧命令）")
    parser.add_argument("--patience", type=int, default=10, help="早停：验证集指标连续几轮不升就停")
    parser.add_argument("--name", type=str, default="best_transfer", help="权重文件名（不含 .pth）")
    parser.add_argument("--labels", choices=["fer2013", "ferplus"], default="fer2013",
                        help="训练/验证标签：fer2013（默认，原标签）/ ferplus（FER+ 众包标签；"
                             "需先运行 代码\\18_ferplus_labels.py；测试集不受影响）")
    parser.add_argument("--merge-val", action="store_true",
                        help="A 方案扩训练集：把 publicTest(val) 并入训练；需配合 --val-holdout")
    parser.add_argument("--val-holdout", type=int, default=1500,
                        help="merge-val 时从合并数据中分层留出的早停/选轮样本数（默认 1500）")
    parser.add_argument("--split-seed", type=int, default=42,
                        help="merge-val 分层留出集的随机种子（默认 42）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    # ---- v2 新增配方参数（全部默认关 / 保持旧行为）----
    parser.add_argument("--amp", action="store_true", help="混合精度训练（fp16 autocast + GradScaler，验证 fp32）")
    parser.add_argument("--optimizer", choices=["adamw", "lion"], default="adamw",
                        help="优化器：adamw（默认）/ lion（建议 lr 1e-4）")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="前 N 轮学习率线性升到 --lr，再余弦退火")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="梯度范数裁剪阈值（0=关）")
    parser.add_argument("--ema-decay", type=float, default=0.0, help="EMA 衰减（0=关，建议 0.999）")
    parser.add_argument("--swa-last", type=int, default=0, help="训练末尾平均最后 K 轮权重（0=关）")
    parser.add_argument("--mixup", type=float, default=0.0, help="mixup 的 beta 参数（0=关，建议 0.2）")
    parser.add_argument("--balance", choices=["none", "weights", "sampler", "focal"], default="none",
                        help="类别不均衡处理：none=保持旧行为（默认带类别权重，除非 --no-weight）；"
                             "weights=类别权重；sampler=WeightedRandomSampler；focal=FocalLoss(γ=1)")
    parser.add_argument("--aug", choices=["basic", "strong"], default="basic",
                        help="数据增强：basic（默认，与旧版一致）/ strong（强增强）")
    parser.add_argument("--drop-path", type=float, default=0.0,
                        help="随机深度（stochastic depth）比例，0=关（默认）；建议 0.1~0.2")
    parser.add_argument("--note", type=str, default="", help="实验备注（写进实验记录汇总表）")
    args = parser.parse_args()

    set_seed(args.seed)
    device = md.get_device()
    torch.backends.cudnn.benchmark = True          # 输入尺寸固定，让 cuDNN 自动选最快的卷积算法

    # 类别权重 / 损失配置
    use_weights = (not args.no_weight) and args.balance != "sampler"
    gamma = 1.0 if args.balance == "focal" else 0.0
    num_classes = len(CLASS_NAMES)

    print("=" * 96)
    print("v2 训练：ImageNet 预训练模型微调（任意 timm 模型 + 配方消融）")
    print("=" * 96)
    print(f"  设备：{device}   seed={args.seed}   模型：{args.model}   输入尺寸：{args.img_size}   标签={args.labels}")
    print(f"  epochs={args.epochs}  batch_size={args.batch_size}  lr={args.lr}  "
          f"weight_decay={args.weight_decay}  optimizer={args.optimizer}")
    print(f"  amp={'开' if args.amp else '关'}  类别权重={'开' if use_weights else '关'}  "
          f"balance={args.balance}  label_smoothing={args.label_smoothing}  drop_path={args.drop_path}")
    print(f"  ema_decay={args.ema_decay}  swa_last={args.swa_last}  warmup_epochs={args.warmup_epochs}  "
          f"grad_clip={args.grad_clip}  mixup={args.mixup}  aug={args.aug}  patience={args.patience}")
    print(f"  扩训练集：merge_val={args.merge_val}  val_holdout={args.val_holdout}  split_seed={args.split_seed}")
    print(f"  num_workers={args.num_workers}   项目内依赖目录：{DEPS_DIR}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"  GPU 显存：空闲 {free/1e9:.1f}/{total/1e9:.1f} GB")

    # ① 数据（train 开增强、val/test 不开；这里只用 train/val，绝不评估 test）
    print("\n① 构造 DataLoader（灰度→3 通道、放大、ImageNet 归一化）")
    loaders, class_weights, stats = ds.build_transfer_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers, out_size=args.img_size,
        aug=args.aug, labels=args.labels, merge_val=args.merge_val,
        val_holdout=args.val_holdout, split_seed=args.split_seed)
    print(f"  train {len(loaders['train'].dataset)} 张 / val {len(loaders['val'].dataset)} 张"
          f"（test 集不参与本次训练/验证，未使用）")
    print(f"  归一化：mean={stats['mean']}  std={stats['std']}  out_size={stats['out_size']}  aug={args.aug}")
    print(f"  类别权重（{len(CLASS_NAMES)} 类）= {[round(float(w), 3) for w in class_weights]}")
    if args.balance == "sampler":
        ytr = np.asarray(loaders["train"].dataset.y)
        sample_w = torch.tensor(np.asarray(class_weights)[ytr], dtype=torch.double)
        sampler = WeightedRandomSampler(sample_w, num_samples=len(ytr), replacement=True)
        loaders["train"] = DataLoader(loaders["train"].dataset, batch_size=args.batch_size,
                                      sampler=sampler, num_workers=args.num_workers,
                                      pin_memory=torch.cuda.is_available(), drop_last=False,
                                      persistent_workers=args.num_workers > 0)
        print("  已启用 WeightedRandomSampler（按类别频率重采样，损失里不再叠类别权重）")

    weight_t = class_weights.to(device) if use_weights else None
    loss_cfg = {"amp": bool(args.amp), "weight": weight_t, "gamma": gamma,
                "label_smoothing": args.label_smoothing, "mixup": args.mixup,
                "grad_clip": args.grad_clip}

    # ② 预训练模型
    print("\n② 加载 ImageNet 预训练模型")
    model = build_pretrained_model(args.model, device, drop_path=args.drop_path)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  参数量：总 {total:,} / 可训练 {trainable:,}")

    # ②.5 优化器 / 学习率调度 / EMA
    if args.optimizer == "lion":
        from timm.optim import Lion
        optimizer = Lion(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"  优化器：Lion（timm）lr={args.lr} weight_decay={args.weight_decay}")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print(f"  优化器：AdamW lr={args.lr} weight_decay={args.weight_decay}")

    if args.warmup_epochs > 0:
        warmup = min(args.warmup_epochs, args.epochs)

        def _lr_lambda(ep: int) -> float:            # ep 从 0 开始
            if ep < warmup:
                return (ep + 1) / warmup
            progress = (ep - warmup) / max(1, args.epochs - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
        print(f"  学习率调度：warmup {warmup} 轮线性升温 → cosine 退火到 0")
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        print(f"  学习率调度：CosineAnnealingLR（T_max={args.epochs}）")

    ema = make_ema(model, args.ema_decay)
    scaler = torch.amp.GradScaler("cuda") if args.amp else None
    if scaler is not None:
        print("  AMP：训练 fp16 autocast + GradScaler；验证始终 fp32")

    # ③ 训练循环
    print(f"\n③ 开始微调（最多 {args.epochs} 轮，早停 patience={args.patience}，"
          f"最佳模型按 val_macro_f1 优先、val_acc 次之挑选）")
    header = (f"  {'epoch':>5} {'lr':>9} {'train_loss':>11} {'train_acc':>10} "
              f"{'val_loss':>9} {'val_acc':>8} {'val_macroF1':>12} {'time':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    history = []
    best = {"key": (-1.0, -1.0), "epoch": -1, "val_acc": -1.0, "macro_f1": -1.0,
            "val_loss": None, "source": "raw"}
    max_acc = {"val_acc": -1.0, "macro_f1": -1.0, "epoch": -1}
    patience_left = args.patience
    swa_buffer = collections.deque(maxlen=max(1, args.swa_last)) if args.swa_last > 0 else None
    t_start = time.time()
    ckpt_path = MODEL_DIR / f"{args.name}.pth"

    def _try_update_best(val: dict, epoch: int, state_dict: dict, source: str) -> bool:
        """按 (macro_f1, acc) 字典序判断是否更新最佳并保存权重；返回是否更新。"""
        key = (float(val["macro_f1"]), float(val["acc"]))
        if key > best["key"]:
            best.update({"key": key, "epoch": epoch, "val_acc": float(val["acc"]),
                         "macro_f1": float(val["macro_f1"]), "val_loss": float(val["loss"]),
                         "source": source})
            save_checkpoint(ckpt_path, state_dict, args, epoch, val, source)
            return True
        return False

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, loaders["train"], optimizer, scaler, device, loss_cfg,
                             ema=ema, epoch=epoch, n_epochs=args.epochs)
        val_kw = {"weight": weight_t, "label_smoothing": args.label_smoothing, "gamma": gamma}
        va = evaluate(model, loaders["val"], device, **val_kw)
        va_ema = None
        if ema is not None:
            va_ema = evaluate(ema_module(ema), loaders["val"], device, **val_kw)
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()
        dt = time.time() - t0
        row = {"epoch": epoch, "lr": lr_now, "train_loss": tr["loss"], "train_acc": tr["acc"],
               "val_loss": va["loss"], "val_acc": va["acc"], "val_macro_f1": va["macro_f1"],
               "time_s": dt}
        if va_ema is not None:
            row.update({"val_loss_ema": va_ema["loss"], "val_acc_ema": va_ema["acc"],
                        "val_macro_f1_ema": va_ema["macro_f1"]})
        history.append(row)

        # 记录各 epoch 的最高 acc（只做记录，不用于选模型）
        if va["acc"] > max_acc["val_acc"]:
            max_acc = {"val_acc": va["acc"], "macro_f1": va["macro_f1"], "epoch": epoch}

        raw_improved = _try_update_best(va, epoch, model.state_dict(), "raw")
        ema_improved = False
        if va_ema is not None:
            ema_improved = _try_update_best(va_ema, epoch, ema_module(ema).state_dict(), "ema")

        if swa_buffer is not None:                  # 缓存最后 K 轮权重用于 SWA
            swa_buffer.append({k: v.detach().cpu().clone() for k, v in model.state_dict().items()})

        mark = " ← best" if (raw_improved or ema_improved) else ""
        if va_ema is not None:
            print(f"  {epoch:>5} {lr_now:>9.2e} {tr['loss']:>11.4f} {tr['acc']:>10.4f} "
                  f"{va['loss']:>9.4f} {va['acc']:>8.4f} {va['macro_f1']:>12.4f} {dt:>6.1f}s{mark}"
                  f"  | EMA: acc={va_ema['acc']:.4f} F1={va_ema['macro_f1']:.4f}"
                  f"{' ←best' if ema_improved else ''}", flush=True)
        else:
            print(f"  {epoch:>5} {lr_now:>9.2e} {tr['loss']:>11.4f} {tr['acc']:>10.4f} "
                  f"{va['loss']:>9.4f} {va['acc']:>8.4f} {va['macro_f1']:>12.4f} {dt:>6.1f}s{mark}",
                  flush=True)

        if not (raw_improved or ema_improved):
            patience_left -= 1
        else:
            patience_left = args.patience
        if patience_left <= 0:
            print(f"  早停：验证集指标已连续 {args.patience} 轮没有提升，第 {epoch} 轮停止。")
            break

    total_time = time.time() - t_start

    # ④ SWA：平均最后 K 轮权重，评估一次，比现有最佳好才采用
    swa_info = None
    if swa_buffer is not None and len(swa_buffer) >= 2:
        print(f"\n④ SWA：平均最后 {len(swa_buffer)} 轮权重并评估 val …")
        swa_sd = average_state_dicts(list(swa_buffer))
        model.load_state_dict(swa_sd)
        val_kw = {"weight": weight_t, "label_smoothing": args.label_smoothing, "gamma": gamma}
        va_swa = evaluate(model, loaders["val"], device, **val_kw)
        print(f"  SWA val：acc={va_swa['acc']:.4f}  macro_F1={va_swa['macro_f1']:.4f}")
        adopted = _try_update_best(va_swa, len(history), swa_sd, "swa")
        swa_info = {"swa_val_acc": va_swa["acc"], "swa_val_macro_f1": va_swa["macro_f1"],
                    "swa_adopted": adopted}
        print(f"  SWA 权重{'已采用（优于当前最佳）' if adopted else '未采用（不如当前最佳 raw/EMA）'}")

    # ⑤ 保存历史与曲线
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / f"train_{args.name}.json").write_text(
        json.dumps({"args": vars(args), "history": history, "best": {k: v for k, v in best.items() if k != "key"},
                    "max_acc": max_acc, "swa": swa_info}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    curves_path = RESULT_DIR / f"training_curves_{args.name}.png"
    plot_curves(history, args.name, curves_path)

    # ⑥ 实验记录（jsonl + summary.md）
    extra = {"max_acc": max_acc["val_acc"], "max_acc_epoch": max_acc["epoch"],
             "max_acc_macro_f1": max_acc["macro_f1"]}
    if swa_info:
        extra.update(swa_info)
    write_experiment_record(args, history, best, total_time, len(history), extra)

    # ⑦ 汇总
    print("\n" + "=" * 96)
    print("v2 训练结束")
    print("=" * 96)
    print(f"  实验名：{args.name}   模型：{args.model}@{args.img_size}   seed={args.seed}")
    print(f"  总用时：{total_time/60:.1f} 分钟（{len(history)} 轮，平均 {total_time/max(1,len(history)):.1f}s/轮）")
    print(f"  最佳 epoch：{best['epoch']} [{best['source']}]  val_acc={best['val_acc']:.4f}  "
          f"val_macro_F1={best['macro_f1']:.4f}  val_loss={best['val_loss']:.4f}")
    print(f"  最高 val_acc 出现在第 {max_acc['epoch']} 轮：acc={max_acc['val_acc']:.4f} "
          f"macro_F1={max_acc['macro_f1']:.4f}")
    print(f"  权重文件：{ckpt_path}")
    print(f"  训练历史：{RESULT_DIR / ('train_' + args.name + '.json')}")
    print(f"  训练曲线：{curves_path}")
    print(f"  实验汇总：{EXPERIMENTS_JSONL}  /  {SUMMARY_MD}")
    print("  下一步 → 06_evaluate.py --weights 模型\\" + args.name + ".pth（仅在 Lead 允许时评估 test）")


if __name__ == "__main__":
    main()
