# -*- coding: utf-8 -*-
r"""
18_ferplus_labels.py —— P0：FER+ 众包标签对齐（只写新文件，不改动现有数据/模型）

做什么
------
FER+（Barsoum et al., 2016）对 FER2013 的同一批 35,887 张图重新做了众包标注：
每张图 10 个人投票，给出 10 类票数（neutral/happiness/surprise/sadness/anger/
disgust/fear/contempt/unknown/NF）。本脚本把 FER+ 标签与现有 `fer2013.npz` 按
“Usage 分块 + 行序”对齐（实测两边行数、拆分完全一致），生成：

  数据\fer2013\ferplus_labels.npz
      y_hard_{split}   (N,) int64     硬标签（7 类，-1=该图所有票都是 unknown/NF，无效）
      valid_{split}    (N,) bool      是否可用于训练/验证/评估
      p_soft_{split}   (N,7) float32  软标签（7 类票数归一化；无效行为 0）
  数据\fer2013\ferplus_labels_meta.json  统计/映射/来源/引用

映射规则（与 07_demo_app.py 的 CLASS_NAMES 顺序一致）：
    angry←anger, disgust←disgust+contempt, fear←fear, happy←happiness,
    sad←sadness, surprise←surprise, neutral←neutral；unknown/NF 不计入。
  —— contempt 并入 disgust 是常见做法（也可按需改）；unknown/NF 票不参与 argmax。

注意：测试集标签也一并生成，但**A 快筛只使用 train/val**，测试集成绩仍以原始
FER2013 标签为准（便于与 v2 的 75.93% 直接比较）。

运行：
    $env:PYTHONIOENCODING='utf-8'
    python 代码\18_ferplus_labels.py
"""
import json
import pathlib
import sys
import urllib.request

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[1]
FER_DIR = ROOT / "数据" / "fer2013"
NPZ_PATH = FER_DIR / "fer2013.npz"
CSV_PATH = FER_DIR / "fer2013new.csv"
OUT_NPZ = FER_DIR / "ferplus_labels.npz"
OUT_META = FER_DIR / "ferplus_labels_meta.json"
CSV_URL = "https://raw.githubusercontent.com/microsoft/FERPlus/master/fer2013new.csv"

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
# FER+ 列 → 我们的 7 类（contempt 并入 disgust；unknown/NF 忽略）
VOTE_COLS = {
    "angry": ["anger"],
    "disgust": ["disgust", "contempt"],
    "fear": ["fear"],
    "happy": ["happiness"],
    "sad": ["sadness"],
    "surprise": ["surprise"],
    "neutral": ["neutral"],
}
SPLITS = (("train", 0, 28709, "Training"),
          ("val", 28709, 32298, "PublicTest"),
          ("test", 32298, 35887, "PrivateTest"))


def download_csv() -> None:
    if CSV_PATH.exists() and CSV_PATH.stat().st_size > 1_000_000:
        print(f"已存在 {CSV_PATH.name}（{CSV_PATH.stat().st_size/1024/1024:.2f} MB），跳过下载")
        return
    print(f"下载 FER+ 标签：{CSV_URL}")
    with urllib.request.urlopen(CSV_URL, timeout=120) as r:
        data = r.read()
    assert len(data) > 1_000_000, f"下载内容过小（{len(data)} 字节）"
    CSV_PATH.write_bytes(data)
    print(f"已保存 {CSV_PATH}（{len(data)/1024/1024:.2f} MB）")


def main() -> None:
    import csv
    FER_DIR.mkdir(parents=True, exist_ok=True)
    download_csv()

    print("\n① 解析 FER+ CSV（按行序对齐，不依赖 Image name）")
    with open(CSV_PATH, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    print(f"   CSV 行数：{len(rows)}")
    need_cols = ["Usage", "neutral", "happiness", "surprise", "sadness", "anger", "disgust",
                 "fear", "contempt", "unknown", "NF"]
    assert set(need_cols) <= set(rows[0].keys()), f"CSV 缺列：{set(need_cols) - set(rows[0].keys())}"

    with np.load(NPZ_PATH) as z:
        orig = {s: z[f"y_{s}"].astype(np.int64) for s in ("train", "val", "test")}
        X_shapes = {s: z[f"X_{s}"].shape for s in ("train", "val", "test")}

    out = {}
    stats = {"splits": {}, "mapping": VOTE_COLS, "source": CSV_URL,
             "citation": "Barsoum E. et al. Training Deep Networks for Facial Expression "
                         "Recognition with Crowd-Sourced Label Distribution. ICMI 2016. arXiv:1608.01041",
             "alignment": "按 FER+ CSV 的 Usage 分块 + 行序（Training→PublicTest→PrivateTest）对齐"}
    for split, lo, hi, usage in SPLITS:
        n = hi - lo
        assert X_shapes[split][0] == n, f"{split} npz 数量 {X_shapes[split][0]} != {n}"
        block = rows[lo:hi]
        assert sum(1 for r in block if r["Usage"] == usage) == n, f"{split} Usage 块不匹配"
        y_hard = np.full(n, -1, dtype=np.int64)
        valid = np.zeros(n, dtype=bool)
        p_soft = np.zeros((n, len(CLASS_NAMES)), dtype=np.float32)
        for i, r in enumerate(block):
            votes = np.array([sum(int(r[c]) for c in VOTE_COLS[name]) for name in CLASS_NAMES],
                             dtype=np.float64)
            total = votes.sum()
            if total <= 0:
                continue
            p_soft[i] = (votes / total).astype(np.float32)
            y_hard[i] = int(votes.argmax())
            valid[i] = True
        out[f"y_hard_{split}"] = y_hard
        out[f"valid_{split}"] = valid
        out[f"p_soft_{split}"] = p_soft
        # 统计
        counts = np.bincount(y_hard[valid], minlength=len(CLASS_NAMES)).tolist()
        n_valid = int(valid.sum())
        agree = int((y_hard[valid] == orig[split][valid]).sum())
        stats["splits"][split] = {
            "n": n, "n_valid": n_valid, "n_dropped": n - n_valid,
            "class_counts": dict(zip(CLASS_NAMES, counts)),
            "agreement_with_fer2013": round(agree / max(n_valid, 1), 4),
        }
        print(f"   {split:5s}：有效 {n_valid}/{n}（丢弃 {n - n_valid}），"
              f"与原标签一致率 {agree / max(n_valid, 1):.1%}，"
              f"类别分布 {dict(zip(CLASS_NAMES, counts))}")

    # 训练集类别权重建议（供报告/训练参考；训练脚本会按过滤后的标签重新计算）
    tr = stats["splits"]["train"]["class_counts"]
    w = {k: round(sum(tr.values()) / (len(tr) * max(v, 1)), 3) for k, v in tr.items()}
    stats["train_class_weights_ferplus"] = w
    print(f"   FER+ 训练集类别权重建议：{w}")

    np.savez_compressed(OUT_NPZ, **out)
    OUT_META.write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n② 已保存：{OUT_NPZ.relative_to(ROOT)}")
    print(f"   元信息：{OUT_META.relative_to(ROOT)}")
    print("   提示：A 快筛只用 train/val；test 标签仅作后续辅助参考，最终成绩仍以原始 FER2013 测试标签为准。")
    print("=" * 82)


if __name__ == "__main__":
    main()
