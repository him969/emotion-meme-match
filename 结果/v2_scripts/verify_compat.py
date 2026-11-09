# -*- coding: utf-8 -*-
"""验证 v2 权重包与 06_evaluate.py / 07_demo_app.py 的加载函数完全兼容（只重建模型，不跑 test 评估）。"""
import importlib.util
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


ckpt_path = ROOT / "模型" / "v2_r1_amp.pth"
if not ckpt_path.exists():
    print("权重尚未生成，跳过")
    raise SystemExit(0)
ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
print(f"权重包字段：{sorted(ckpt.keys())}")
print(f"profile={ckpt['profile']} model={ckpt['model_name']} img_size={ckpt['img_size']} "
      f"num_classes={ckpt['num_classes']} epoch={ckpt['epoch']} val_acc={ckpt['val_acc']:.4f} "
      f"ema={ckpt.get('ema')} source={ckpt.get('source')}")

# 06 / 07 都是这个逻辑：timm.create_model(model_name, num_classes) + load_state_dict
m = timm.create_model(ckpt["model_name"], pretrained=False, num_classes=int(ckpt["num_classes"]))
m.load_state_dict(ckpt["state_dict"])
m.eval()
print("✅ 06/07 重建逻辑（timm.create_model + load_state_dict）通过")

# 直接调用 06 的 build_model_from_ckpt（不触发 test 评估）
ev = load_module("fer_evaluate", "06_evaluate.py")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, profile = ev.build_model_from_ckpt(ckpt, dev)
print(f"✅ 06_evaluate.build_model_from_ckpt 通过：{profile}")

# 07 的 build_model_from_ckpt（注意它只返回 model）
app = load_module("fer_demo_app", "07_demo_app.py")
model2 = app.build_model_from_ckpt(ckpt, dev)
size, rgb = app.model_input_spec(ckpt)
print(f"✅ 07_demo_app.build_model_from_ckpt 通过：{type(model2).__name__}，输入规格 {size}×{size} rgb={rgb}")
