# -*- coding: utf-8 -*-
r"""
02_meme_index.py —— 阶段 1（数据管线）：猫 meme 表情包预处理 + 元数据表

输入：数据\cat_meme_extracted\cat_meme\<情绪>\<情绪>_<来源>_<名字>.<ext>
输出：
  数据\meme_ready\<情绪>\<文件名主干>.png   统一处理后的图（256×256 RGB，白底）
  数据\meme_index.csv                       元数据表（演示程序靠它检索）
  结果\meme_preview.png                    全部 meme 的预览拼图（人工目检用）

处理规则（对应 PROJECT.md 第 4.4 节）：
  ① gif 只取首帧（静态展示足够，且演示时不需要播放动画）；
  ② RGBA / 带透明的 P 图：先合成到白底，避免透明区域变黑；
  ③ 等比缩放 + 白色填充到 256×256（不直接拉伸，避免变形）；
  ④ 去重：同一张图的 gif 与 jpeg 版本共用一个文件名主干 → 每个主干只留 1 张；
  ⑤ 元数据表列：file, emotion, source, ext, width, height, is_gif, processed。

运行（PowerShell）：
    $env:PYTHONIOENCODING='utf-8'
    python <项目根>/代码/02_meme_index.py
"""
import collections
import pathlib
import random
import sys

from PIL import Image

# Windows 控制台默认 GBK，打印 ✅/❌ 会抛 UnicodeEncodeError；统一改成 UTF-8 输出
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ---------- 路径：全部相对本文件定位 ----------
ROOT = pathlib.Path(__file__).resolve().parents[1]          # 表情识别项目\
SRC_DIR = ROOT / "数据" / "cat_meme_extracted" / "cat_meme"  # 原始表情包
READY_DIR = ROOT / "数据" / "meme_ready"                    # 统一处理后的图
INDEX_CSV = ROOT / "数据" / "meme_index.csv"                # 元数据表
PREVIEW_PNG = ROOT / "结果" / "meme_preview.png"            # 预览拼图

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
TARGET = 256                                                # 统一边长
SEED = 2026                                                 # 固定随机种子：每次运行结果可复现
# 去重选择表（可选）：{"文件名主干": "指定文件名"}；已有图沿用历史选择，避免"新增图片导致旧图版本被换"
DEDUP_CHOICES = ROOT / "数据" / "meme_dedup_choices.json"


def to_rgb_on_white(im: Image.Image) -> Image.Image:
    """把任意模式的图转成 RGB：透明区域合成到白底，避免 PNG/GIF 的透明变黑。"""
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        canvas = Image.new("RGBA", rgba.size, (255, 255, 255, 255))  # 纯白底
        canvas.alpha_composite(rgba)                                 # 把透明图叠上去
        return canvas.convert("RGB")
    return im.convert("RGB")


def fit_square(im: Image.Image, size: int = TARGET) -> Image.Image:
    """等比缩放，让长边等于 size，再居中放到白底方画布上（不拉伸变形）。"""
    w, h = im.size
    scale = size / max(w, h)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    im = im.resize(new_size, Image.Resampling.LANCZOS)               # 高质量缩小
    canvas = Image.new("RGB", (size, size), (255, 255, 255))         # 白底方画布
    canvas.paste(im, ((size - new_size[0]) // 2, (size - new_size[1]) // 2))
    return canvas


def main() -> None:
    print("=" * 82)
    print("猫 meme 表情包预处理 + 元数据表")
    print("=" * 82)
    print(f"  原始目录：{SRC_DIR}")
    print(f"  输出目录：{READY_DIR}")
    print()

    if not SRC_DIR.exists():
        raise SystemExit(f"找不到表情包目录：{SRC_DIR}")

    rng = random.Random(SEED)                    # 固定种子 → 新图去重结果可复现
    prefs = {}
    if DEDUP_CHOICES.exists():
        import json
        prefs = json.loads(DEDUP_CHOICES.read_text(encoding="utf-8"))
        print(f"  去重选择表：{DEDUP_CHOICES.name}（{len(prefs)} 条历史选择）")
    rows = []                                    # 元数据行
    total_files = 0
    per_class_unique = {}

    # ① 逐类：按「文件名主干」分组，同组（同图的 gif/jpeg/png 版本）只留 1 个
    #    老图优先沿用 meme_dedup_choices.json 的历史选择；新图才用固定种子随机选
    print("① 去重选图（同一张图的 gif/jpeg 只留一个版本）")
    for emotion in CLASS_NAMES:
        d = SRC_DIR / emotion
        if not d.exists():
            raise SystemExit(f"缺少类别目录：{d}")
        files = sorted(p for p in d.iterdir() if p.is_file())
        total_files += len(files)
        groups = collections.defaultdict(list)   # 主干 -> [同图的不同格式文件]
        for p in files:
            groups[p.stem].append(p)
        chosen = []
        for stem, versions in sorted(groups.items()):
            pref_name = prefs.get(stem)
            pick = next((p for p in versions if p.name == pref_name), None) if pref_name else None
            chosen.append(pick if pick is not None else rng.choice(versions))
        per_class_unique[emotion] = len(chosen)
        print(f"  {emotion:9s} 文件 {len(files):2d} 个 → 唯一图 {len(chosen):2d} 张")
        rows.extend(chosen)

    # ② 逐张处理并保存 256×256 PNG
    print("\n② 统一预处理：取首帧 → 白底 → 等比缩放 → 256×256")
    (READY_DIR).mkdir(parents=True, exist_ok=True)
    index_rows = []
    for p in sorted(rows, key=lambda x: (x.parent.name, x.name)):
        emotion = p.parent.name
        # 文件名格式：<情绪>_<来源>_<名字>.<ext>  →  来源 = 第 2 段
        parts = p.stem.split("_", 2)
        source = parts[1] if len(parts) >= 2 else "unknown"

        with Image.open(p) as im:
            orig_w, orig_h = im.size                 # 原始尺寸（记进元数据表）
            is_gif = p.suffix.lower() == ".gif"
            if is_gif:
                im.seek(0)                           # gif 只取首帧
            rgb = to_rgb_on_white(im)
            fixed = fit_square(rgb)

        out_dir = READY_DIR / emotion
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{p.stem}.png"
        fixed.save(out_path, format="PNG")

        index_rows.append({
            "file": p.name,
            "emotion": emotion,
            "source": source,
            "ext": p.suffix.lower().lstrip("."),
            "width": orig_w,
            "height": orig_h,
            "is_gif": int(is_gif),
            "processed": out_path.relative_to(ROOT).as_posix(),
        })

    # ③ 写元数据表（utf-8-sig：用 Excel 直接打开也不会中文乱码）
    print("\n③ 写出元数据表")
    import csv

    INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(INDEX_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"  [ok] {INDEX_CSV}（{len(index_rows)} 行）")

    # ④ 自检：每类 >=1、总数=28、处理后图能读且是 256×256
    print("\n④ 自检")
    bad = []
    for r in index_rows:
        fp = ROOT / r["processed"]
        if not fp.exists():
            bad.append((r["file"], "处理后文件不存在"))
            continue
        with Image.open(fp) as im2:
            if im2.size != (TARGET, TARGET) or im2.mode != "RGB":
                bad.append((r["file"], f"尺寸/模式不对：{im2.size} {im2.mode}"))
    total = len(index_rows)
    print(f"  唯一图总数：{total}  坏图：{len(bad)}")
    for name, why in bad[:10]:
        print(f"    ❌ {name}: {why}")
    empty = [e for e, n in per_class_unique.items() if n == 0]
    print(f"  每类唯一图数：{per_class_unique}")
    print(f"  缺失类别：{empty if empty else '无'}")
    # 项目最初入库 28 张（2026-10-30 之后又补了 30 张 pielet 来源的图，总数会更多）
    assert total >= 28, f"唯一图总数异常：{total}（至少应有 28 张）"
    assert not bad, "存在预处理失败的文件"
    assert not empty, "存在没有任何 meme 的类别"

    # ⑤ 预览拼图：7 行（情绪）× 最多 max_cols 列，一眼检查缩放/白底/是否重复
    print("\n⑤ 生成预览拼图（人工目检用）")
    import matplotlib

    matplotlib.use("Agg")                        # 不弹窗，直接存文件
    import matplotlib.pyplot as plt
    import numpy as np

    max_cols = max(per_class_unique.values())
    fig, axes = plt.subplots(len(CLASS_NAMES), max_cols,
                             figsize=(max_cols * 1.25, len(CLASS_NAMES) * 1.4))
    axes = np.atleast_2d(axes)
    for ri, emotion in enumerate(CLASS_NAMES):
        subset = [r for r in index_rows if r["emotion"] == emotion]
        for ci in range(max_cols):
            ax = axes[ri, ci]
            ax.axis("off")
            if ci < len(subset):
                with Image.open(ROOT / subset[ci]["processed"]) as im3:
                    ax.imshow(im3)
            if ci == 0:                            # 行首标情绪（用英文，避免 matplotlib 缺中文字体）
                ax.set_title(emotion, fontsize=9, color="#333333")
    fig.suptitle("Cat memes after preprocessing (256x256, white background)", fontsize=11)
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))
    PREVIEW_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PREVIEW_PNG, dpi=110)
    plt.close(fig)
    print(f"  [ok] {PREVIEW_PNG}")

    # ⑥ 结论
    print("\n" + "=" * 82)
    print("结论")
    print("=" * 82)
    print(f"  原始文件 {total_files} 个 → 去重后唯一图 {total} 张，全部处理成 {TARGET}×{TARGET} RGB 白底 PNG")
    print("  每类唯一图：")
    for emotion in CLASS_NAMES:
        n = per_class_unique[emotion]
        flag = "🔴 只有 1 张（演示会重复）" if n == 1 else ("🟡 偏少" if n <= 3 else "✅")
        print(f"    {emotion:9s} {n:2d}  {flag}")
    print("  下一步 → 03_dataset.py（FER2013 自写 Dataset）")


if __name__ == "__main__":
    main()
