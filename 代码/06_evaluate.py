# -*- coding: utf-8 -*-
r"""
06_evaluate.py —— 阶段 3：模型评估（Accuracy / Macro-F1 / 7×7 混淆矩阵），支持 TTA

评估什么？
  Accuracy（准确率）  = 预测对的张数 / 总张数
  Macro-F1           = 7 个类各自 F1 的算术平均（不按样本数加权）
                       本数据集类别很不均衡，Macro-F1 比 Accuracy 更能反映"小类有没有被放弃"
  混淆矩阵            = 7×7 表格：行 = 真实类别，列 = 预测类别

两种模型都能评估（自动识别权重包里的 profile）：
  · 自己从头训练的 CNN（04_model.py，profile = baseline / wide / flat）
  · ImageNet 预训练微调的 ResNet（08_transfer_train.py，profile = timm:resnet18 / timm:resnet34）

TTA（Test-Time Augmentation，测试时增强，方案 B）：
  推理时把同一张图做多个"视角"（原图 / 水平翻转 / 上下左右小平移），各自算概率后取平均。
  模型对单个视角的噪声会被平均掉，通常能涨 0.5~1 个点，且完全不用重新训练。
  加 `--tta` 打开（6 视角，默认推荐）；`--tta-strong` 是 10 视角版（6 基础 + 旋转 ±8° + 缩放
  1.08/0.92），是否更好要用验证集实测，不一定会涨。

评估范围：
  默认 `--splits all`（val + test，正式评估用）；调参/对比实验请用 `--splits val`，
  避免反复使用测试集（本项目测试集已用过多次，后续应只在最终确认时跑一次 test）。

产出（`--name` 默认 test；开 TTA 时自动加 `_tta` / `_strong` 后缀，避免覆盖已有评估）：
  结果\evaluation_{tag}.json / .txt      指标 + 每类 precision/recall/F1 + 混淆矩阵
  结果\confusion_matrix_{tag}.png        混淆矩阵热力图（左：张数，右：按行归一化）

运行（PowerShell）：
    $env:PYTHONIOENCODING='utf-8'
    python <项目根>/代码/06_evaluate.py
    # 带 TTA 评估迁移学习模型：
    ... 06_evaluate.py --weights 模型\best_transfer.pth --name transfer --tta
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

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "代码"
MODEL_DIR = ROOT / "模型"
RESULT_DIR = ROOT / "结果"
DEPS_DIR = ROOT / "_deps"                      # timm/huggingface_hub/safetensors 装在这里
if str(DEPS_DIR) not in sys.path:
    sys.path.insert(0, str(DEPS_DIR))

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def load_module(name: str, filename: str):
    """按文件路径加载同目录下的脚本模块（文件名以数字开头，不能直接 import）。"""
    spec = importlib.util.spec_from_file_location(name, CODE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ds = load_module("fer_dataset", "03_dataset.py")    # build_dataloaders / build_transfer_loaders
md = load_module("fer_model", "04_model.py")        # build_model / get_device / count_parameters


def load_checkpoint(path: pathlib.Path, device: torch.device) -> dict:
    """读取训练时保存的权重包（含 state_dict、结构参数、归一化统计量等）。"""
    if not path.exists():
        raise SystemExit(f"找不到权重文件：{path}\n请先运行 05_train.py 或 08_transfer_train.py")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
    except Exception:
        # 老版本 torch 或权重里有非基础类型时退回普通加载（本项目的权重是自己训练的，安全）
        ckpt = torch.load(path, map_location=device, weights_only=False)
    return ckpt


def build_model_from_ckpt(ckpt: dict, device: torch.device):
    """按权重包里的 profile 重建模型：
    · profile 以 "timm:" 开头 → ImageNet 预训练微调模型（08_transfer_train.py 产出）；
    · 其他 → 自己的 CNN（04_model.py 的三档之一）。
    """
    profile = ckpt.get("profile", "baseline")      # 旧权重没有 profile 字段 → 按 baseline 处理
    if profile.startswith("timm:"):
        import timm                                 # 只有迁移学习模型才需要 timm

        model = timm.create_model(ckpt["model_name"], pretrained=False,
                                  num_classes=int(ckpt["num_classes"]))
    else:
        model = md.build_model(num_classes=int(ckpt["num_classes"]), profile=profile)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), profile


def _shift_batch(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """把一批图整体平移 (dx, dy) 像素，空出来的边填 0（0 在标准化后约等于"平均灰度"）。

    用途：TTA 时造出"人脸略微偏移"的视角。dx>0 表示向右，dy>0 表示向下。
    """
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
    """TTA 的 6 个视角：原图、水平翻转、左/右/上/下各平移 s 像素。

    平移量 s 按输入边长自适应：48×48 的模型 s=2，128×128 的迁移模型 s=5。
    （表情左右对称，翻转型安全；小幅平移模拟人脸位置没对准。）
    """
    size = xb.shape[-1]
    s = max(2, round(size / 24))
    return [xb, torch.flip(xb, dims=[-1]),
            _shift_batch(xb, s, 0), _shift_batch(xb, -s, 0),
            _shift_batch(xb, 0, s), _shift_batch(xb, 0, -s)]


def _affine_view(xb: torch.Tensor, angle_deg: float, scale: float = 1.0) -> torch.Tensor:
    """绕中心做旋转/缩放（纯 torch，不依赖 torchvision），用于强 TTA。

    angle_deg：旋转角度（度）；scale >1 放大、<1 缩小（缩小后四周用边缘像素填充）。
    """
    import math

    theta = torch.zeros(len(xb), 2, 3, device=xb.device, dtype=xb.dtype)
    a = math.radians(angle_deg)
    cos, sin = math.cos(a), math.sin(a)
    # affine_grid 用「输出坐标 → 输入坐标」的逆变换矩阵
    theta[:, 0, 0] = cos / scale
    theta[:, 0, 1] = sin / scale
    theta[:, 1, 0] = -sin / scale
    theta[:, 1, 1] = cos / scale
    grid = torch.nn.functional.affine_grid(theta, list(xb.shape), align_corners=False)
    return torch.nn.functional.grid_sample(xb, grid, align_corners=False, padding_mode="border")


def tta_views_strong(xb: torch.Tensor) -> list[torch.Tensor]:
    """强 TTA 的 10 个视角：6 个基础视角 + 旋转 ±8° + 缩放 1.08 / 0.92。"""
    return tta_views(xb) + [
        _affine_view(xb, 8.0), _affine_view(xb, -8.0),
        _affine_view(xb, 0.0, 1.08), _affine_view(xb, 0.0, 0.92),
    ]


@torch.no_grad()
def predict_all(model, loader, device, tta: bool = False,
                strong: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """在整个数据集上预测，返回 (y_true, y_pred, probs)，probs 形状 (N, 7)。

    tta=True 时把 6 个视角的 softmax 概率平均后再取 argmax（方案 B）；
    strong=True 时用 10 个视角（基础 6 + 旋转/缩放），对应 --tta-strong。
    """
    model.eval()
    ys, probs = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        if strong:
            views = tta_views_strong(xb)
        elif tta:
            views = tta_views(xb)
        else:
            views = [xb]
        p = torch.zeros(len(yb), len(CLASS_NAMES), device=device)
        for view in views:
            p += torch.softmax(model(view), dim=1)   # logits → 7 类概率
        p /= len(views)                              # 多视角平均
        probs.append(p.cpu().numpy())
        ys.append(yb.numpy())
    probs = np.concatenate(probs)
    return np.concatenate(ys), probs.argmax(axis=1), probs


def print_confusion(cm: np.ndarray, names: list[str]) -> None:
    """以文本表格形式打印混淆矩阵（行=真实，列=预测）。"""
    short = [n[:3] for n in names]
    print("  行=真实类别，列=预测类别")
    print("  " + " " * 11 + "".join(f"{s:>7}" for s in short) + f"{'合计':>8}")
    for i, name in enumerate(names):
        row = "".join(f"{int(v):>7}" for v in cm[i])
        print(f"  {name:<10}{row}{int(cm[i].sum()):>8}")
    print("  " + "-" * (11 + 7 * len(names) + 8))


def plot_confusion(cm: np.ndarray, names: list[str], path: pathlib.Path) -> None:
    """画混淆矩阵热力图：左图是张数，右图是「每行归一化」（看出每类被分到哪去的比例）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm_norm = cm.astype(np.float64) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, data, title, fmt in [
        (axes[0], cm, "Confusion matrix (counts)", "d"),
        (axes[1], cm_norm, "Confusion matrix (row-normalized)", ".2f"),
    ]:
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_yticklabels(names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        thresh = (data.max() / 2.0) if data.size else 0.5
        for i in range(len(names)):
            for j in range(len(names)):
                text = f"{int(data[i, j])}" if fmt == "d" else f"{data[i, j]:.2f}"
                ax.text(j, i, text, ha="center", va="center", fontsize=8,
                        color="white" if data[i, j] > thresh else "black")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def top_confusions(cm: np.ndarray, names: list[str], k: int = 5) -> list[dict]:
    """找出最容易混淆的 (真实, 预测) 类别对（排除对角线）。"""
    pairs = []
    for i in range(len(names)):
        for j in range(len(names)):
            if i != j and cm[i, j] > 0:
                pairs.append({"true": names[i], "pred": names[j], "count": int(cm[i, j]),
                              "rate_of_true": float(cm[i, j] / max(cm[i].sum(), 1))})
    pairs.sort(key=lambda d: d["count"], reverse=True)
    return pairs[:k]


def metrics_of(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """算一组预测的 Acc / Macro-F1。"""
    return {"acc": float((y_true == y_pred).mean()),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "n": int(len(y_true))}


def main() -> None:
    parser = argparse.ArgumentParser(description="评估表情分类模型（可选 TTA / 强 TTA）")
    parser.add_argument("--weights", type=str, default="模型/best_model.pth", help="权重文件路径")
    parser.add_argument("--name", type=str, default="test", help="输出文件后缀名")
    parser.add_argument("--tta", action="store_true",
                        help="开启 TTA（6 视角平均，通常 +0.5~1 个点）；输出文件自动加 _tta 后缀")
    parser.add_argument("--tta-strong", action="store_true",
                        help="开启强 TTA（10 视角：6 基础 + 旋转 ±8° + 缩放 1.08/0.92）；自动启用 TTA")
    parser.add_argument("--splits", choices=["all", "val", "test"], default="all",
                        help="评估哪些 split：all=val+test（默认，最终评估用）；val=只用验证集（调参阶段用，避免反复碰测试集）")
    args = parser.parse_args()

    strong = bool(args.tta_strong)
    use_tta = bool(args.tta or strong)
    splits = ["val", "test"] if args.splits == "all" else [args.splits]
    weights_path = pathlib.Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = ROOT / weights_path
    tag = args.name + ("_tta" if use_tta else "") + ("_strong" if strong else "")

    device = md.get_device()
    tta_desc = "关"
    if strong:
        tta_desc = "开（强 TTA：10 视角 = 6 基础 + 旋转 ±8° + 缩放 1.08/0.92）"
    elif use_tta:
        tta_desc = "开（6 个视角平均）"
    print("=" * 88)
    print("阶段 3：模型评估（Accuracy / Macro-F1 / 混淆矩阵）" + (" + 强 TTA" if strong else (" + TTA" if use_tta else "")))
    print("=" * 88)
    print(f"  设备：{device}")
    print(f"  权重：{weights_path}")
    print(f"  评估 split：{splits}")
    print(f"  TTA：{tta_desc}")

    # ① 读权重 + 重建模型
    ckpt = load_checkpoint(weights_path, device)
    model, profile = build_model_from_ckpt(ckpt, device)
    total_params, _ = md.count_parameters(model)
    print(f"  模型：{profile}  num_classes={ckpt['num_classes']}  参数量={total_params:,}")
    print(f"  权重来自第 {ckpt.get('epoch')} 轮，训练时 val_acc={ckpt.get('val_acc', float('nan')):.4f}，"
          f"val_macro_F1={ckpt.get('val_macro_f1', float('nan')):.4f}")

    # ② DataLoader：迁移学习模型用 FERTransferDataset（放大 + 3 通道 + ImageNet 归一化），
    #    从头训练的模型用 FERDataset（48×48 灰度 + FER 自己的 mean/std）。
    if profile.startswith("timm:"):
        loaders, _, stats = ds.build_transfer_loaders(batch_size=256, num_workers=0,
                                                      out_size=int(ckpt["img_size"]))
        print(f"  预处理：迁移学习模式  输入 {ckpt['img_size']}×{ckpt['img_size']} RGB  "
              f"ImageNet mean/std")
    else:
        loaders, _, stats = ds.build_dataloaders(batch_size=256, num_workers=0)
        print(f"  预处理：自写 CNN 模式  输入 48×48 灰度  mean={stats['mean']:.4f} std={stats['std']:.4f}")

    # ③ 在指定 split 上评估：普通 + TTA 各跑一遍，方便对比
    print("\n① 各 split 指标（普通推理 vs TTA）")
    results = {"no_tta": {}, "tta": {}}
    preds_store = {}
    for split in splits:
        y_true, y_pred, probs = predict_all(model, loaders[split], device, tta=False)
        results["no_tta"][split] = metrics_of(y_true, y_pred)
        line = (f"  {split:5s}: {len(y_true)} 张  普通 Accuracy={results['no_tta'][split]['acc']:.4f}  "
                f"Macro-F1={results['no_tta'][split]['macro_f1']:.4f}")
        if use_tta:
            y_true_t, y_pred_t, probs_t = predict_all(model, loaders[split], device,
                                                      tta=not strong, strong=strong)
            results["tta"][split] = metrics_of(y_true_t, y_pred_t)
            line += (f"   |  {'强 ' if strong else ''}TTA Accuracy={results['tta'][split]['acc']:.4f}  "
                     f"Macro-F1={results['tta'][split]['macro_f1']:.4f}")
        print(line)
        # 详细报告用哪种预测：开了 TTA 就用 TTA 的
        if use_tta:
            preds_store[split] = (y_true_t, y_pred_t, probs_t)
        else:
            preds_store[split] = (y_true, y_pred, probs)

    # ④ 详细结果：优先测试集；只跑 val（调参阶段）时用 val
    detail_split = "test" if "test" in preds_store else splits[-1]
    y_true, y_pred, probs = preds_store[detail_split]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    report = classification_report(y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
                                   target_names=CLASS_NAMES, zero_division=0, digits=4)
    print(f"\n② {detail_split} 每类指标（precision 查准率 / recall 查全率 / f1）")
    print("\n".join("  " + line for line in report.splitlines()))
    print(f"\n③ {detail_split} 混淆矩阵")
    print_confusion(cm, CLASS_NAMES)

    confusions = top_confusions(cm, CLASS_NAMES)
    print("\n④ 最容易混淆的类别对（真实 → 被预测成）")
    for c in confusions:
        print(f"  {c['true']:>8s} → {c['pred']:<8s} {c['count']:>4d} 张"
              f"（占该类 {c['rate_of_true']*100:.1f}%）")

    # ⑤ 存盘
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULT_DIR / f"evaluation_{tag}.json"
    payload = {
        "weights": str(weights_path),
        "profile": profile,
        "checkpoint_epoch": ckpt.get("epoch"),
        "tta": use_tta,
        "tta_strong": strong,
        "splits": splits,
        "detail_split": detail_split,
        "metrics": results,
        "class_names": CLASS_NAMES,
        f"{detail_split}_confusion_counts": cm.tolist(),
        "top_confusions": confusions,
    }
    rep_dict = classification_report(y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
                                     target_names=CLASS_NAMES, zero_division=0,
                                     output_dict=True, digits=4)
    payload[f"{detail_split}_per_class"] = {k: rep_dict[k] for k in CLASS_NAMES}
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    out_txt = RESULT_DIR / f"evaluation_{tag}.txt"
    tta_label = "强 TTA（10 视角）" if strong else ("TTA（6 视角）" if use_tta else "关")
    lines = [
        "FER2013 表情分类模型评估报告",
        "=" * 60,
        f"权重文件: {weights_path}",
        f"模型: {profile}",
        f"权重轮次: {ckpt.get('epoch')}",
        f"TTA: {tta_label}",
        f"评估 split: {splits}",
        "",
        "各 split 指标:",
    ]
    for kind in ("no_tta", "tta"):
        if not results[kind]:
            continue
        for split, m in results[kind].items():
            lines.append(f"  [{kind:6s}] {split:5s}: n={m['n']}  Accuracy={m['acc']:.4f}  "
                         f"Macro-F1={m['macro_f1']:.4f}")
    lines += ["", f"{detail_split} 每类指标（{'强 TTA' if strong else ('TTA' if use_tta else '普通')}预测）:", report,
              f"{detail_split} 混淆矩阵（行=真实，列=预测）:", str(cm), "", "最容易混淆的类别对:"]
    for c in confusions:
        lines.append(f"  {c['true']} -> {c['pred']}: {c['count']} 张（占该类 {c['rate_of_true']*100:.1f}%）")
    out_txt.write_text("\n".join(lines), encoding="utf-8")

    out_png = RESULT_DIR / f"confusion_matrix_{tag}.png"
    plot_confusion(cm, CLASS_NAMES, out_png)

    final_metrics = results["tta"][detail_split] if use_tta else results["no_tta"][detail_split]
    print("\n" + "=" * 88)
    print("评估完成 ✅" + ("（强 TTA）" if strong else ("（TTA）" if use_tta else "")))
    print(f"  指标 JSON：{out_json}")
    print(f"  文本报告：{out_txt}")
    print(f"  混淆矩阵图：{out_png}")
    print(f"  {detail_split} 集 Accuracy = {final_metrics['acc']*100:.2f}%，"
          f"Macro-F1 = {final_metrics['macro_f1']:.4f}")
    print("  下一步 → 07_demo_app.py（Streamlit 演示：上传图片 → 概率条 → 匹配猫 meme）")
    print("=" * 88)


if __name__ == "__main__":
    main()
