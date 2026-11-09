# -*- coding: utf-8 -*-
r"""
16_meme_anim.py —— 为 GIF 表情包生成"演示用动画版"（256×256、白底、压缩）

背景
----
数据管线（02_meme_index.py）为了训练/匹配/特征一致，把所有 meme（含 GIF）统一处理成
**静态 256×256 PNG（GIF 只取首帧）**，所以 Streamlit 演示里 GIF 不会动。
本脚本**不改数据管线**，只额外产出"演示用"的动画版本：

    数据\cat_meme_extracted\cat_meme\<情绪>\<原文件>.gif
        → 数据\meme_anim\<文件名主干>.gif   （256×256 白底、≤32 帧、调色板压缩）

匹配/特征/TTA 仍然用静态 `数据\meme_ready\`（首帧），动画只影响页面展示，
因此对模型准确率、匹配分数、视觉特征/神似模式没有任何影响。

页面侧（07_demo_app.py）：侧边栏"GIF 动图播放"开关（默认开），开启时如果选中的 meme
有动画版就用 `st.image(动画路径)` 播放（Streamlit 会对 ≤1460px 的 GIF 原样透传，浏览器自动播放），
关闭时退回静态 PNG。

运行（PowerShell）：
    $env:PYTHONIOENCODING='utf-8'
    python 代码\16_meme_anim.py
    # 可选：--max-frames 32 --colors 64 --size 256 --max-kb 1500 --force
"""
import argparse
import json
import pathlib
import sys
import time

from PIL import Image, ImageSequence

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "数据"
MEME_CSV = DATA_DIR / "meme_index.csv"
SRC_ROOT = DATA_DIR / "cat_meme_extracted" / "cat_meme"
ANIM_DIR = DATA_DIR / "meme_anim"
INDEX_JSON = ANIM_DIR / "_index.json"


def find_source(file_name: str, emotion: str) -> pathlib.Path | None:
    """在原始 meme 目录里找同名文件；找不到就全目录搜一遍。"""
    p = SRC_ROOT / emotion / file_name
    if p.exists():
        return p
    matches = list(SRC_ROOT.rglob(file_name))
    return matches[0] if matches else None


def make_anim(src: pathlib.Path, dst: pathlib.Path, size: int, max_frames: int,
              colors: int, max_kb: float) -> dict | None:
    """把原始 GIF 转成 256×256 白底、压缩后的演示用 GIF；返回统计信息。"""
    im = Image.open(src)
    n_frames = getattr(im, "n_frames", 1)
    if n_frames <= 1:
        return None
    # 均匀采样：保留足够帧但控制体积；帧间隔在采样后按"原帧间隔×步长"累加
    step = max(1, n_frames // max_frames)
    frames, durations = [], []
    for idx, frame in enumerate(ImageSequence.Iterator(im)):
        if idx % step != 0 and idx != n_frames - 1:
            continue
        duration = frame.info.get("duration", im.info.get("duration", 80)) or 80
        frame = frame.convert("RGBA")
        w, h = frame.size
        scale = size / max(w, h)
        frame = frame.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                             Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (size, size), "white")
        canvas.paste(frame, ((size - frame.width) // 2, (size - frame.height) // 2), frame)
        frames.append(canvas.convert("P", palette=Image.Palette.ADAPTIVE, colors=colors))
        # 采样跳帧时，duration 按跳过的帧数累计（近似保持播放速度）
        durations.append(min(500, duration * step))
        if len(frames) >= max_frames:
            break

    def save(frames_, durations_, pal_colors: int) -> None:
        quant = [f.convert("P", palette=Image.Palette.ADAPTIVE, colors=pal_colors) for f in frames_]
        quant[0].save(dst, save_all=True, append_images=quant[1:], duration=durations_,
                      loop=0, optimize=True, disposal=2)

    save(frames, durations, colors)
    if dst.stat().st_size > max_kb * 1024 and colors > 32:
        save(frames, durations, 32)          # 体积超标 → 降到 32 色再压一次
    return {"src": str(src.relative_to(ROOT)), "frames": len(frames), "src_frames": n_frames,
            "size_kb": round(dst.stat().st_size / 1024, 1),
            "duration_ms": int(sum(durations)), "width": size, "height": size}


def main() -> None:
    ap = argparse.ArgumentParser(description="生成 GIF 表情包的演示用动画版（256×256 白底）")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--max-frames", type=int, default=32)
    ap.add_argument("--colors", type=int, default=64)
    ap.add_argument("--max-kb", type=float, default=1500, help="单张超过该体积则降到 32 色")
    ap.add_argument("--force", action="store_true", help="已存在也重新生成")
    args = ap.parse_args()

    import pandas as pd
    memes = pd.read_csv(MEME_CSV)
    gifs = memes[memes["is_gif"] == 1]
    print("=" * 84)
    print(f"GIF 表情包 → 演示动画版：共 {len(gifs)} 张，输出目录 {ANIM_DIR}")
    print("=" * 84)
    ANIM_DIR.mkdir(parents=True, exist_ok=True)

    index: dict = {}
    if INDEX_JSON.exists():
        try:
            index = json.loads(INDEX_JSON.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            index = {}
    produced, skipped, failed = 0, 0, []
    t0 = time.time()
    total_kb = 0.0
    for _, row in gifs.iterrows():
        file_name = row["file"]
        stem = pathlib.Path(file_name).stem
        dst = ANIM_DIR / f"{stem}.gif"
        if dst.exists() and not args.force:
            info = index.get(stem, {})
            if not info:
                with Image.open(dst) as chk:
                    info = {"frames": getattr(chk, "n_frames", 1), "size_kb": round(dst.stat().st_size / 1024, 1),
                            "width": chk.width, "height": chk.height}
            index[stem] = {**info, "file": file_name, "emotion": row["emotion"], "status": "cached"}
            total_kb += index[stem].get("size_kb", 0)
            skipped += 1
            print(f"  [跳过] {file_name:40s} 已存在（{index[stem].get('size_kb')} KB）")
            continue
        src = find_source(file_name, row["emotion"])
        if src is None:
            failed.append((file_name, "找不到原始文件"))
            print(f"  [失败] {file_name:40s} 找不到原始文件")
            continue
        try:
            info = make_anim(src, dst, args.size, args.max_frames, args.colors, args.max_kb)
            if info is None:
                failed.append((file_name, "不是动图"))
                print(f"  [失败] {file_name:40s} 不是动图")
                continue
            index[stem] = {**info, "file": file_name, "emotion": row["emotion"], "status": "ok"}
            total_kb += info["size_kb"]
            produced += 1
            print(f"  [OK]   {file_name:40s} {info['src_frames']:3d} 帧 → {info['frames']:3d} 帧  "
                  f"{info['size_kb']:7.1f} KB  {info['duration_ms']:5d} ms")
        except Exception as e:  # noqa: BLE001
            failed.append((file_name, repr(e)[:120]))
            print(f"  [失败] {file_name:40s} {type(e).__name__}: {e}")

    INDEX_JSON.write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    print("-" * 84)
    print(f"完成：新生成 {produced} 张，复用 {skipped} 张，失败 {len(failed)} 张，"
          f"总计统计体积 {total_kb / 1024:.1f} MB，用时 {time.time() - t0:.1f}s")
    print(f"索引：{INDEX_JSON.relative_to(ROOT)}")
    if failed:
        print("失败清单：")
        for name, reason in failed:
            print(f"  - {name}: {reason}")


if __name__ == "__main__":
    main()
