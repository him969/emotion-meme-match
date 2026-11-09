# -*- coding: utf-8 -*-
r"""
15_ensemble_eval.py —— 多模型集成评估（概率平均 + 可选强 TTA）

用途
----
模型优化 v2 阶段 4：把多个 seed / 多个配置训练出来的模型在**同一批验证集**上做概率平均
（soft-voting ensemble），再用 TTA（6 视角或强 TTA 10 视角）评估，通常比单模型 +1~2 个点。
只有最终确认时才对 test 评估一次（--splits all）；平时调参请用 --splits val。

用法
----
  $env:PYTHONIOENCODING='utf-8'
  # 开发阶段：只用验证集，看集成是否比单模型好
  python 代码\15_ensemble_eval.py --weights 模型\v2_seed42.pth 模型\v2_seed43.pth 模型\v2_seed44.pth --name v2_ensemble --tta --splits val
  # 最终评估（test 只跑这一次）
  ... 15_ensemble_eval.py --weights 模型\v2_seed42.pth 模型\v2_seed43.pth 模型\v2_seed44.pth --name v2_final --tta-strong --splits all

产出：结果\evaluation_{name}.json / .txt、结果\confusion_matrix_{name}.png
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pathlib
import sys

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import json
import importlib.util

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix

ROOT = pathlib.Path(__file__).resolve().parents[1]
CODE_DIR = ROOT / "代码"
RESULT_DIR = ROOT / "结果"
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


def load_module(name: str, filename: str):
    """按文件路径加载同目录脚本（文件名以数字开头，不能直接 import）。"""
    spec = importlib.util.spec_from_file_location(name, CODE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 复用 06 的模型重建 / TTA / 指标 / 画图逻辑（06 的 main 只在 __main__ 下执行）
ev = load_module("fer_evaluate", "06_evaluate.py")
md = ev.md
ds = ev.ds


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="多模型集成评估（概率平均 + TTA）")
    parser.add_argument("--weights", nargs="+", required=True, help="一个或多个权重文件路径")
    parser.add_argument("--name", type=str, default="ensemble", help="输出文件后缀名")
    parser.add_argument("--tta", action="store_true", help="6 视角 TTA")
    parser.add_argument("--tta-strong", action="store_true", help="强 TTA（10 视角）")
    parser.add_argument("--splits", choices=["all", "val", "test"], default="val",
                        help="默认只评估 val（调参阶段）；最终评估用 all")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    strong = bool(args.tta_strong)
    use_tta = bool(args.tta or strong)
    splits = ["val", "test"] if args.splits == "all" else [args.splits]
    tag = args.name + ("_tta" if use_tta else "") + ("_strong" if strong else "")
    device = md.get_device()

    weights_paths = []
    for w in args.weights:
        p = pathlib.Path(w)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise SystemExit(f"找不到权重：{p}")
        weights_paths.append(p)

    print("=" * 88)
    print("多模型集成评估（概率平均）")
    print("=" * 88)
    print(f"  设备：{device}")
    print(f"  模型数：{len(weights_paths)}")
    print(f"  TTA：{'强 TTA（10 视角）' if strong else ('6 视角' if use_tta else '关')}")
    print(f"  splits：{splits}")

    # 按 (profile, img_size) 缓存 DataLoader，避免每个模型重复构造
    loader_cache: dict = {}

    def get_loaders(profile: str, img_size: int):
        key = (profile.split(":")[0], int(img_size))
        if key not in loader_cache:
            if profile.startswith("timm:"):
                loaders, _, _ = ds.build_transfer_loaders(batch_size=args.batch_size,
                                                          num_workers=0, out_size=int(img_size))
            else:
                loaders, _, _ = ds.build_dataloaders(batch_size=args.batch_size, num_workers=0)
            loader_cache[key] = loaders
        return loader_cache[key]

    # ① 逐模型预测 → 概率累加
    prob_sums = {split: None for split in splits}
    y_true_map = {}
    per_model_metrics = []
    for wi, wp in enumerate(weights_paths, start=1):
        ckpt = ev.load_checkpoint(wp, device)
        model, profile = ev.build_model_from_ckpt(ckpt, device)
        loaders = get_loaders(profile, ckpt.get("img_size", 128))
        line = f"  [{wi}/{len(weights_paths)}] {wp.name}  profile={profile}  img_size={ckpt.get('img_size')}  " \
               f"epoch={ckpt.get('epoch')}  val_acc={ckpt.get('val_acc', float('nan')):.4f}"
        print(line)
        for split in splits:
            y_true, y_pred, probs = ev.predict_all(model, loaders[split], device,
                                                   tta=not strong, strong=strong)
            y_true_map[split] = y_true
            prob_sums[split] = probs.astype(np.float64) if prob_sums[split] is None \
                else prob_sums[split] + probs.astype(np.float64)
            m = ev.metrics_of(y_true, y_pred)
            per_model_metrics.append({"weights": wp.name, "split": split,
                                      "acc": m["acc"], "macro_f1": m["macro_f1"]})
            print(f"      {split:5s}: acc={m['acc']:.4f}  macro-F1={m['macro_f1']:.4f}")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ② 集成（概率平均）
    print("\n② 集成结果（概率平均）")
    results = {}
    detail_probs = {}
    for split in splits:
        probs = prob_sums[split] / len(weights_paths)
        y_pred = probs.argmax(axis=1)
        m = ev.metrics_of(y_true_map[split], y_pred)
        results[split] = m
        detail_probs[split] = probs
        avg_individual = np.mean([x["acc"] for x in per_model_metrics if x["split"] == split])
        print(f"  {split:5s}: 集成 acc={m['acc']:.4f}  macro-F1={m['macro_f1']:.4f}  "
              f"（单模型平均 acc={avg_individual:.4f}，提升 {m['acc'] - avg_individual:+.4f}）")

    # ③ 详细报告（优先 test；只跑 val 时用 val）
    detail_split = "test" if "test" in splits else splits[-1]
    y_true, y_pred = y_true_map[detail_split], detail_probs[detail_split].argmax(axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    report = classification_report(y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
                                   target_names=CLASS_NAMES, zero_division=0, digits=4)
    print(f"\n③ {detail_split} 每类指标")
    print("\n".join("  " + line for line in report.splitlines()))
    print(f"\n④ {detail_split} 混淆矩阵")
    ev.print_confusion(cm, CLASS_NAMES)

    # ④ 存盘
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_json = RESULT_DIR / f"evaluation_{tag}.json"
    payload = {
        "weights": [str(p) for p in weights_paths],
        "n_models": len(weights_paths),
        "tta": use_tta, "tta_strong": strong, "splits": splits, "detail_split": detail_split,
        "ensemble_metrics": results,
        "per_model_metrics": per_model_metrics,
        "class_names": CLASS_NAMES,
        f"{detail_split}_confusion_counts": cm.tolist(),
        f"{detail_split}_per_class": {k: classification_report(
            y_true, y_pred, labels=list(range(len(CLASS_NAMES))), target_names=CLASS_NAMES,
            zero_division=0, output_dict=True, digits=4)[k] for k in CLASS_NAMES},
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    out_txt = RESULT_DIR / f"evaluation_{tag}.txt"
    lines = ["FER2013 表情分类模型集成评估报告", "=" * 60,
             "模型：" + "、".join(p.name for p in weights_paths),
             f"TTA：{'强 TTA（10 视角）' if strong else ('6 视角' if use_tta else '关')}",
             f"评估 split：{splits}", "", "集成指标:"]
    for split, m in results.items():
        lines.append(f"  {split:5s}: n={m['n']}  Accuracy={m['acc']:.4f}  Macro-F1={m['macro_f1']:.4f}")
    lines += ["", "单模型指标:"]
    for x in per_model_metrics:
        lines.append(f"  {x['weights']:28s} {x['split']:5s}: acc={x['acc']:.4f}  macro-F1={x['macro_f1']:.4f}")
    lines += ["", f"{detail_split} 每类指标:", report,
              f"{detail_split} 混淆矩阵（行=真实，列=预测）:", str(cm)]
    out_txt.write_text("\n".join(lines), encoding="utf-8")

    out_png = RESULT_DIR / f"confusion_matrix_{tag}.png"
    ev.plot_confusion(cm, CLASS_NAMES, out_png)

    final = results[detail_split]
    print("\n" + "=" * 88)
    print("集成评估完成 ✅")
    print(f"  指标 JSON：{out_json}")
    print(f"  文本报告：{out_txt}")
    print(f"  混淆矩阵图：{out_png}")
    print(f"  {detail_split} 集 Accuracy = {final['acc']*100:.2f}%，Macro-F1 = {final['macro_f1']:.4f}")
    print("=" * 88)


if __name__ == "__main__":
    main()
