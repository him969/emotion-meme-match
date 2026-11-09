# -*- coding: utf-8 -*-
r"""
05_train.py —— 阶段 3：训练 + 早停 + 保存权重

训练策略（对应 PROJECT.md 第 5 节阶段 3）：
    损失      CrossEntropyLoss（可选类别权重，缓解 disgust 只有 436 张的不均衡）
    优化器    Adam(lr=1e-3) + CosineAnnealingLR（学习率按余弦曲线慢慢降）
    轮数      默认 30 epoch，早停：验证集准确率连续 --patience 轮不升就停
    指标      Accuracy、Macro-F1（每轮都算，存进训练历史）
    权重      验证集准确率最好的那一轮存成 模型\best_model.pth

命令行用法（都能改）：
    python 05_train.py                          # 默认：30 epoch、带类别权重
    python 05_train.py --epochs 20 --no-weight  # 消融实验：不加权
    python 05_train.py --name best_model_plain  # 换一个权重文件名

运行（PowerShell）：
    $env:PYTHONIOENCODING='utf-8'
    python <项目根>/代码/05_train.py
"""
import os
import pathlib
import sys

# 本机 conda 与 torch 各带一份 Intel OpenMP；torch + matplotlib 一起用会报 OMP Error #15，
# 这个环境变量允许重复加载（详见 PROJECT.md 第 3.2 节第 5 条）。必须在 import torch 之前设！
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Windows 控制台默认 GBK，打印 ✅/❌ 会抛 UnicodeEncodeError；统一改成 UTF-8 输出
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import importlib.util
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "代码"
MODEL_DIR = ROOT / "模型"
RESULT_DIR = ROOT / "结果"
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def load_module(name: str, filename: str):
    """按文件路径加载同目录下的脚本模块（文件名以数字开头，不能直接 import）。
    这些脚本都有 __main__ 保护，import 时只会定义函数/类，不会执行 main()。"""
    spec = importlib.util.spec_from_file_location(name, CODE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ds = load_module("fer_dataset", "03_dataset.py")     # 复用：build_dataloaders / FERDataset
md = load_module("fer_model", "04_model.py")         # 复用：build_model / get_device / count_parameters


def set_seed(seed: int) -> None:
    """固定随机种子：数据增强、参数初始化、shuffle 都可复现（深度学习实验的基本要求）。"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader, criterion, device) -> dict:
    """在验证/测试集上跑一遍：返回 loss、accuracy、macro-F1。"""
    model.eval()                                   # 切换到评估模式（BN 用固定统计量、关闭 dropout）
    total_loss, n, correct = 0.0, 0, 0
    all_pred, all_true = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * len(yb)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        n += len(yb)
        all_pred.append(pred.cpu().numpy())
        all_true.append(yb.cpu().numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return {
        "loss": total_loss / n,
        "acc": correct / n,
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }


def train_one_epoch(model, loader, criterion, optimizer, device) -> dict:
    """训练一轮：前向 → 算 loss → 反向传播 → 更新参数。"""
    model.train()                                  # 训练模式（BN 用当前 batch 的统计量）
    total_loss, n, correct = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)      # 清空上一轮的梯度（否则会累加）
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()                            # 反向传播：算每个参数的梯度
        optimizer.step()                           # 用梯度更新参数
        total_loss += loss.item() * len(yb)
        correct += (logits.argmax(dim=1) == yb).sum().item()
        n += len(yb)
    return {"loss": total_loss / n, "acc": correct / n}


def plot_curves(history: list, name: str, path: pathlib.Path) -> None:
    """把训练曲线画出来（loss/acc/macro-F1 三张子图）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(epochs, [h["train_loss"] for h in history], label="train")
    axes[0].plot(epochs, [h["val_loss"] for h in history], label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()

    axes[1].plot(epochs, [h["train_acc"] for h in history], label="train")
    axes[1].plot(epochs, [h["val_acc"] for h in history], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    axes[2].plot(epochs, [h["val_macro_f1"] for h in history], color="tab:green")
    axes[2].set_title("Validation Macro-F1")
    axes[2].set_xlabel("epoch")

    fig.suptitle(f"Training curves: {name}", fontsize=12)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="FER2013 表情分类训练")
    parser.add_argument("--epochs", type=int, default=30, help="最多训练多少轮")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3, help="初始学习率")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--profile", choices=["baseline", "wide", "flat"], default="baseline",
                        help="模型档位：baseline=9.4 万参数（原方案）；wide=44 万；flat=237 万（保留空间信息）")
    parser.add_argument("--no-weight", action="store_true", help="关闭类别权重（消融实验用）")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="标签平滑系数（如 0.05，缓解模型过于自信）")
    parser.add_argument("--patience", type=int, default=3, help="早停：验证集准确率连续几轮不升就停")
    parser.add_argument("--name", type=str, default="best_model", help="权重文件名（不含 .pth）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 子进程数，Windows 建议 0")
    args = parser.parse_args()

    set_seed(args.seed)
    device = md.get_device()
    weighted = not args.no_weight
    print("=" * 88)
    print("阶段 3：训练 + 早停 + 保存权重")
    print("=" * 88)
    print(f"  设备：{device}   seed={args.seed}   模型档位：{args.profile}")
    print(f"  epochs={args.epochs}  batch_size={args.batch_size}  lr={args.lr}  "
          f"weight_decay={args.weight_decay}  类别权重={'开' if weighted else '关'}  "
          f"label_smoothing={args.label_smoothing}  patience={args.patience}")

    # ① 数据
    print("\n① 构造 DataLoader")
    loaders, class_weights, stats = ds.build_dataloaders(batch_size=args.batch_size, num_workers=args.num_workers)
    print(f"  train {len(loaders['train'].dataset)} 张 / val {len(loaders['val'].dataset)} 张 / "
          f"test {len(loaders['test'].dataset)} 张")
    print(f"  归一化 mean={stats['mean']:.4f} std={stats['std']:.4f}")
    print(f"  类别权重（{len(CLASS_NAMES)} 类）= {[round(float(w), 3) for w in class_weights]}")

    # ② 模型与训练组件
    model = md.build_model(profile=args.profile).to(device)
    total_params, _ = md.count_parameters(model)
    criterion = nn.CrossEntropyLoss(weight=(class_weights.to(device) if weighted else None),
                                    label_smoothing=args.label_smoothing)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    print(f"\n② 模型档位：{args.profile}  结构：{model.cfg}  参数量：{total_params:,}")

    # ③ 训练循环
    print(f"\n③ 开始训练（最多 {args.epochs} 轮，早停 patience={args.patience}）")
    header = (f"  {'epoch':>5} {'lr':>9} {'train_loss':>11} {'train_acc':>10} "
              f"{'val_loss':>9} {'val_acc':>8} {'val_macroF1':>12} {'time':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    history, best = [], {"val_acc": -1.0, "epoch": -1}
    patience_left = args.patience
    t_start = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr = train_one_epoch(model, loaders["train"], criterion, optimizer, device)
        va = evaluate(model, loaders["val"], criterion, device)
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()                       # 每个 epoch 结束，学习率按余弦曲线下降
        dt = time.time() - t0

        row = {"epoch": epoch, "lr": lr_now, "train_loss": tr["loss"], "train_acc": tr["acc"],
               "val_loss": va["loss"], "val_acc": va["acc"], "val_macro_f1": va["macro_f1"], "time_s": dt}
        history.append(row)

        is_best = va["acc"] > best["val_acc"]
        if is_best:
            best = {"val_acc": va["acc"], "val_macro_f1": va["macro_f1"], "val_loss": va["loss"],
                    "epoch": epoch, "lr": lr_now}
            # 保存完整权重：连模型结构参数、归一化统计量、类别名一起存，推理时不用再猜
            torch.save({
                "state_dict": model.state_dict(),
                "profile": args.profile,
                "model_cfg": dict(model.cfg),
                "widths": model.cfg["widths"],
                "num_classes": len(CLASS_NAMES),
                "class_names": CLASS_NAMES,
                "norm_mean": stats["mean"],
                "norm_std": stats["std"],
                "epoch": epoch,
                "val_acc": va["acc"],
                "val_macro_f1": va["macro_f1"],
                "args": vars(args),
            }, MODEL_DIR / f"{args.name}.pth")
            patience_left = args.patience
            mark = " ← best"
        else:
            patience_left -= 1
            mark = ""

        print(f"  {epoch:>5} {lr_now:>9.2e} {tr['loss']:>11.4f} {tr['acc']:>10.4f} "
              f"{va['loss']:>9.4f} {va['acc']:>8.4f} {va['macro_f1']:>12.4f} {dt:>6.1f}s{mark}")

        if patience_left <= 0:
            print(f"  早停：验证集准确率已连续 {args.patience} 轮没有提升，第 {epoch} 轮停止。")
            break

    total_time = time.time() - t_start

    # ④ 保存训练历史 + 曲线
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / f"train_{args.name}.json").write_text(
        json.dumps({"args": vars(args), "history": history, "best": best}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    curves_path = RESULT_DIR / f"training_curves_{args.name}.png"
    plot_curves(history, args.name, curves_path)

    # ⑤ 汇总
    print("\n" + "=" * 88)
    print("训练结束")
    print("=" * 88)
    print(f"  总用时：{total_time/60:.1f} 分钟（{len(history)} 轮，平均 {total_time/len(history):.1f}s/轮）")
    print(f"  最佳 epoch：{best['epoch']}  val_acc={best['val_acc']:.4f}  "
          f"val_macro_F1={best['val_macro_f1']:.4f}  val_loss={best['val_loss']:.4f}")
    print(f"  权重文件：{MODEL_DIR / (args.name + '.pth')}")
    print(f"  训练历史：{RESULT_DIR / ('train_' + args.name + '.json')}")
    print(f"  训练曲线：{curves_path}")
    print("  下一步 → 06_evaluate.py（在测试集上算 Accuracy / Macro-F1 / 混淆矩阵）")


if __name__ == "__main__":
    main()
