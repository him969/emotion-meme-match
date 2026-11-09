# -*- coding: utf-8 -*-
"""最终验收：确认所有产物存在、06/07 能加载最优与 seed 权重、test 未被评估、best_model.pth 未变。"""
import importlib.util
import json
import os
import pathlib
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = pathlib.Path(__file__).resolve().parents[1]
CODE = ROOT / "代码"
sys.path.insert(0, str(ROOT / "_deps"))
import timm  # noqa: E402
import torch  # noqa: E402


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, CODE / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


print("== 权重文件 ==")
for n in ("v2_r1_amp", "v2_r2_ema", "v2_r3_mix", "v2_r4_lion", "v2_resnet34_128",
          "v2_effb0_160", "v2_seed42", "v2_seed43", "v2_seed44"):
    p = ROOT / "模型" / f"{n}.pth"
    print(f"  {n}: {'OK' if p.exists() else '缺失'} {p.stat().st_size/1e6:.1f} MB" if p.exists() else f"  {n}: 缺失")

print("\n== 结果文件 ==")
for rel in ("experiments_v2.jsonl", "v2_experiment_summary.md", "v2_recommendation.md",
            "train_v2_r1_amp.json", "train_v2_r2_ema.json", "train_v2_r3_mix.json",
            "train_v2_r4_lion.json", "train_v2_resnet34_128.json", "train_v2_effb0_160.json",
            "train_v2_seed44.json", "training_curves_v2_effb0_160.png"):
    p = ROOT / "结果" / rel
    print(f"  {rel}: {'OK' if p.exists() else '缺失'}")
logs = list((ROOT / "结果" / "v2_logs").glob("*.log"))
print(f"  日志 {len(logs)} 个：{sorted(p.name for p in logs)}")

print("\n== best_model.pth 未被覆盖 ==")
bm = ROOT / "模型" / "best_model.pth"
print(f"  best_model.pth mtime={bm.stat().st_mtime} size={bm.stat().st_size}（应仍为 2026/10/31 11:26, 44800843）")

print("\n== 06/07 加载 v2_seed42/v2_effb0_160/v2_seed43 权重 ==")
ev = load_module("fer_evaluate", "06_evaluate.py")
app = load_module("fer_demo_app", "07_demo_app.py")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for n in ("v2_effb0_160", "v2_seed42", "v2_seed43", "v2_seed44", "v2_resnet34_128", "v2_r4_lion"):
    ckpt = torch.load(str(ROOT / "模型" / f"{n}.pth"), map_location="cpu", weights_only=False)
    m1, prof = ev.build_model_from_ckpt(ckpt, dev)
    m2 = app.build_model_from_ckpt(ckpt, dev)
    size, rgb = app.model_input_spec(ckpt)
    print(f"  {n}: 06 OK({prof}) 07 OK 输入{size}x{size} rgb={rgb} "
          f"val_acc={ckpt['val_acc']:.4f} F1={ckpt['val_macro_f1']:.4f} src={ckpt.get('source')} ema={ckpt.get('ema')}")
    del m1, m2
    torch.cuda.empty_cache()

print("\n== jsonl 记录 ==")
recs = [json.loads(l) for l in (ROOT / "结果" / "experiments_v2.jsonl").read_text(encoding="utf-8").splitlines()]
print(f"  共 {len(recs)} 条：{[r['name'] for r in recs]}")
print(f"  seed42 record: seed={[r.get('seed') for r in recs if r['name']=='v2_seed42']} "
      f"group={[r.get('group') for r in recs if r['name']=='v2_seed42']}")

print("\n✅ 验收脚本执行完毕（未做任何 test 评估）")
