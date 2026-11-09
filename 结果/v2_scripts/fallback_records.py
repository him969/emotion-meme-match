# -*- coding: utf-8 -*-
"""E2 超时兜底：训练进程被提前停掉时，用已保存的最佳 ckpt + 日志里的最后一个 epoch
补写 结果\v2_records\v2_seed43.json / v2_seed44.json（记录 val 指标），再重建聚合文件。

用法：python fallback_records.py v2_seed43 v2_seed44
"""
import importlib.util
import json
import os
import pathlib
import re
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "_deps"))
import torch  # noqa: E402

REC = ROOT / "结果" / "v2_records"
LOGS = ROOT / "结果" / "v2_logs"
names = sys.argv[1:] or ["v2_seed43", "v2_seed44"]


def read_log_text(log_path: pathlib.Path) -> str:
    """读训练日志。PowerShell Tee-Object 默认写 UTF-16LE（带 BOM），这里自动识别编码。"""
    raw = log_path.read_bytes()
    if raw.startswith(b"\xff\xfe"):
        return raw.decode("utf-16-le", errors="replace")
    if raw.startswith(b"\xef\xbb\xbf"):
        return raw.decode("utf-8-sig", errors="replace")
    try:
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return raw.decode("utf-16-le", errors="replace")


def parse_epochs_run(log_path: pathlib.Path):
    """从训练日志里解析已跑完的最大 epoch 编号（epoch 行形如 '     19  3.89e-05 ...'）。"""
    if not log_path.exists():
        return None
    last = None
    for line in read_log_text(log_path).splitlines():
        m = re.match(r"\s{2,}(\d+)\s{2,}\d+(?:\.\d+)?e-?\d+", line)
        if m:
            last = int(m.group(1))
    return last


for name in names:
    ckpt_path = ROOT / "模型" / f"{name}.pth"
    rec_path = REC / f"{name}.json"
    old = None
    if rec_path.exists():
        try:
            old = json.loads(rec_path.read_text(encoding="utf-8"))
        except Exception:
            old = None
    if old is not None and old.get("epochs_run") and old.get("total_time_s"):
        print(f"{name}: 已有完整记录，跳过")
        continue
    if not ckpt_path.exists():
        print(f"{name}: ❌ 找不到权重 {ckpt_path}")
        continue
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    log_path = LOGS / f"{name}.log"
    epochs_run = parse_epochs_run(log_path)
    # 训练时长近似 = 日志文件最后写入时间 - 创建时间（进程被提前停掉时是近似值）
    approx_min = None
    if log_path.exists():
        approx_min = round((log_path.stat().st_mtime - log_path.stat().st_ctime) / 60.0, 1)
    args = ckpt.get("args", {})
    record = {
        "name": name, "group": "seed", "model": ckpt.get("model_name"),
        "img_size": ckpt.get("img_size"), "epochs": int(args.get("epochs", 30)),
        "epochs_run": epochs_run, "batch_size": int(args.get("batch_size", 64)),
        "seed": int(args.get("seed", 0)), "optimizer": args.get("optimizer"),
        "lr": args.get("lr"), "weight_decay": args.get("weight_decay"),
        "amp": bool(args.get("amp", False)), "ema_decay": args.get("ema_decay"),
        "swa_last": args.get("swa_last"), "aug": args.get("aug"), "mixup": args.get("mixup"),
        "balance": args.get("balance"), "label_smoothing": args.get("label_smoothing"),
        "warmup_epochs": args.get("warmup_epochs"), "grad_clip": args.get("grad_clip"),
        "patience": args.get("patience"),
        "best_epoch": ckpt.get("epoch"), "best_val_acc": ckpt.get("val_acc"),
        "best_val_macro_f1": ckpt.get("val_macro_f1"),
        "best_source": ckpt.get("source", "raw"),
        "total_time_s": round((approx_min or 0) * 60, 1),
        "note": f"E2 超时保护：23:45 未跑完即停，取当时最佳 ckpt（已跑 {epochs_run} 轮；"
                f"耗时约 {approx_min} min）",
        "timeout_fallback": True,
    }
    REC.mkdir(parents=True, exist_ok=True)
    rec_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{name}: 记录已写 best_epoch={record['best_epoch']} "
          f"F1={record['best_val_macro_f1']:.4f} acc={record['best_val_acc']:.4f} 已跑 {epochs_run} 轮")

spec = importlib.util.spec_from_file_location("t08", ROOT / "代码" / "08_transfer_train.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.rebuild_aggregates()
print("聚合文件已重建")
