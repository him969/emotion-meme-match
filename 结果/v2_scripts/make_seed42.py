# -*- coding: utf-8 -*-
"""E2 方案：P2（v2_effb0_160）完成后复制其权重为 模型\v2_seed42.pth，
并在 结果\v2_records\v2_seed42.json 写一条 name=v2_seed42 的实验记录（note=由 P2 复制），
再重建 experiments_v2.jsonl / v2_experiment_summary.md。
"""
import importlib.util
import json
import os
import shutil
import sys
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = __import__("pathlib").Path(r"<项目根>")
REC = ROOT / "结果" / "v2_records"

p2_path = ROOT / "模型" / "v2_effb0_160.pth"
p2_rec = REC / "v2_effb0_160.json"
if not p2_path.exists() or not p2_rec.exists():
    print("P2 尚未完成（权重或记录缺失），跳过")
    raise SystemExit(1)

# 校验 P2 记录的 seed 与 30 轮
rec_src = json.loads(p2_rec.read_text(encoding="utf-8"))
print(f"P2 记录：name={rec_src['name']} seed={rec_src['seed']} epochs_run={rec_src['epochs_run']} "
      f"F1={rec_src['best_val_macro_f1']:.4f} acc={rec_src['best_val_acc']:.4f} src={rec_src['best_source']}")
assert rec_src["seed"] == 42, f"P2 seed 应为 42，实际 {rec_src['seed']}"

dst = ROOT / "模型" / "v2_seed42.pth"
shutil.copy2(p2_path, dst)
print(f"已复制：{dst}（{dst.stat().st_size/1e6:.1f} MB）")

rec = dict(rec_src)
rec["name"] = "v2_seed42"
rec["group"] = "seed"
rec["note"] = "由 P2 v2_effb0_160 复制（effb0@160, R4 配方, 30ep, seed=42）"
rec["copied_from"] = "v2_effb0_160"
(REC / "v2_seed42.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
print("已写记录：", REC / "v2_seed42.json")

# 重建聚合文件（复用 08 的函数，带文件锁）
spec = importlib.util.spec_from_file_location("t08", ROOT / "代码" / "08_transfer_train.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.rebuild_aggregates()
print("完成时间：", time.strftime("%H:%M:%S"))
