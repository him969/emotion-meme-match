# -*- coding: utf-8 -*-
"""
01_env_check.py —— 项目环境自检
用途：新会话开工第一步。回答四个问题：
  ① torch / CUDA / GPU 能不能用（本次训练是否可行）
  ② torchvision / opencv 能不能用（决定演示形式）
  ③ 数据是否就位（meme 表情包、FER2013）
  ④ 给出"当前环境下建议走哪条路线"的结论
运行（PowerShell）：
    $env:PYTHONIOENCODING='utf-8'
    python <项目根>/代码/01_env_check.py
"""
import importlib
import pathlib
import sys

# Windows 控制台默认是 GBK，直接打印 ✅/❌ 会抛 UnicodeEncodeError；
# 这里统一改成 UTF-8 输出，个别编码不了的字用 ? 替换，保证脚本不崩。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# 项目根目录 = 本文件所在目录（代码\）的上一级。用相对定位，项目整体搬家后脚本不用改路径。
ROOT = pathlib.Path(__file__).resolve().parents[1]
MEME_DIR = ROOT / "数据" / "cat_meme_extracted" / "cat_meme"
FER_DIR = ROOT / "数据" / "fer2013"

print("=" * 76)
print("① 解释器与基础环境")
print("=" * 76)
print(f"  executable : {sys.executable}")
print(f"  version    : {sys.version.split()[0]}")
print(f"  platform   : {sys.platform}")


def probe(mod_name: str, attr: str = "__version__"):
    """尝试导入模块，返回 (是否成功, 版本或错误信息)。"""
    try:
        m = importlib.import_module(mod_name)
        return True, str(getattr(m, attr, "unknown"))
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


print("\n" + "=" * 76)
print("② 关键库（能否导入 + 版本）")
print("=" * 76)


def repeated_import(mod_name: str, times: int = 3) -> tuple[bool, str]:
    """连续导入多次：用来检测「间歇性被系统策略拦截」的情况。
    每次都用全新的子进程导入，避免 Python 内部缓存掩盖失败。"""
    import subprocess

    ok = 0
    last_err = ""
    for _ in range(times):
        r = subprocess.run(
            [sys.executable, "-c", f"import {mod_name}; print(getattr({mod_name},'__version__','ok'))"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if r.returncode == 0:
            ok += 1
            ver = (r.stdout or "").strip().splitlines()[-1] if r.stdout.strip() else "ok"
        else:
            last_err = (r.stderr or "").strip().splitlines()[-1] if (r.stderr or "").strip() else "unknown"
    return ok == times, f"{ok}/{times} 次成功" + ("" if ok == times else f" | 最后一次错误: {last_err[:70]}")


libs = [
    ("torch", "torch"),
    ("cv2", "opencv"),
    ("PIL", "pillow"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("sklearn", "scikit-learn"),
    ("matplotlib", "matplotlib"),
    ("streamlit", "streamlit"),
    ("timm", "timm"),
]
status = {}
for mod, label in libs:
    ok, info = probe(mod)
    status[mod] = ok
    flag = "✅" if ok else "❌"
    print(f"  {flag} {label:14s} {info[:90]}")

# torchvision 单独做「稳定性」检测（本项目已知会被 Smart App Control 间歇拦截）
tv_ok, tv_info = repeated_import("torchvision", times=3)
status["torchvision"] = tv_ok
print(f"  {'✅' if tv_ok else '❌'} {'torchvision':14s} 稳定性检测: {tv_info}")
print("      ↑ 若显示 1/3、2/3，说明被系统策略「间歇拦截」——代码必须能在没有它的情况下工作")

print("\n" + "=" * 76)
print("③ GPU / CUDA 实测（含一次真实矩阵乘）")
print("=" * 76)
cuda_ok = False
if status.get("torch"):
    import time

    import torch

    print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"  GPU: {p.name}")
        print(f"  显存: {p.total_memory / 1e9:.1f} GB | SM: {p.multi_processor_count} | 算力: {p.major}.{p.minor}")
        try:
            a = torch.randn(2048, 2048, device="cuda")
            b = torch.randn(2048, 2048, device="cuda")
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(5):
                a @ b
            torch.cuda.synchronize()
            print(f"  2048 方阵乘 5 次耗时: {(time.time() - t0):.2f}s  → GPU 可用")
            cuda_ok = True
        except Exception as e:
            print(f"  ⚠ GPU 计算失败: {type(e).__name__}: {e}")
else:
    print("  ❌ torch 不可用，跳过 GPU 检测")

print("\n" + "=" * 76)
print("④ 数据就位情况")
print("=" * 76)
if MEME_DIR.exists():
    classes = sorted([d.name for d in MEME_DIR.iterdir() if d.is_dir()])
    print(f"  ✅ meme 目录存在：{MEME_DIR}")
    print(f"     类别（{len(classes)}）: {classes}")
    for c in classes:
        n = len([p for p in (MEME_DIR / c).iterdir() if p.is_file()])
        print(f"       {c:<10} {n:>3} 个文件")
else:
    print(f"  ❌ meme 目录不存在：{MEME_DIR}")

print(f"  FER2013 目录：{'✅ 存在' if FER_DIR.exists() else '❌ 尚未下载（阶段 1.1 的任务）'}")

print("\n" + "=" * 76)
print("⑤ 结论与建议路线")
print("=" * 76)
if cuda_ok:
    print("  ✅ 训练可行：torch + CUDA 正常，本地 GPU 足够训练 FER2013 小 CNN。")
else:
    print("  ⚠ 训练可能受阻：torch/CUDA 不可用，需要先解决（见 PROJECT.md 第 3.2 节）。")

if status.get("torchvision"):
    print("  ✅ torchvision 可用：可以直接用官方 Dataset / transform / 预训练模型。")
else:
    print("  ❌ torchvision 不可用：请自己写 Dataset 与 transform（项目已按此设计，无需依赖）。")

if status.get("cv2"):
    print("  ✅ opencv 可用：可以做摄像头实时演示与人脸检测。")
else:
    print("  ❌ opencv 不可用：演示请走「上传图片」路线（Streamlit），不要做摄像头版；")
    print("     若必须做摄像头，需新建 Python 3.11 环境并安装 opencv-python。")

print("\n  项目状态：阶段 0~5 已完成（最终模型测试集 Accuracy 64.73%，见 PROJECT.md 第 10 节）。")
print("  想看演示：python -m streamlit run <项目根>/代码/07_demo_app.py")
print("=" * 76)
