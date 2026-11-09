# -*- coding: utf-8 -*-
r"""
04_model.py —— 阶段 2：CNN 模型定义（Baseline CNN）

网络结构（对应 PROJECT.md 第 5 节阶段 2）：
    Conv(1→32, 3×3) + BN + ReLU + MaxPool     48×48 → 24×24
    Conv(32→64, 3×3) + BN + ReLU + MaxPool    24×24 → 12×12
    Conv(64→128, 3×3) + BN + ReLU + MaxPool   12×12 → 6×6
    GlobalAvgPool → FC(128→7)                 6×6 → 1×1 → 7 个 logits
参数约 10 万级，8.5 GB 显存绰绰有余。另有两档加强模型：
  wide（通道 64/128/256 + 隐藏层 + Dropout，约 44 万参数）
  flat（在 wide 基础上把 GAP 换成直接展平，保留空间位置信息，约 237 万参数）
——实测 baseline 训练 30 轮训练集准确率只到 ~53%（欠拟合），需要更宽的模型；详见 PROJECT.md 第 10 节。

几个关键字（第一次出现，后面训练还会用到）：
  logits    —— 网络最后一层直接输出的 7 个实数（没经过 softmax）
                CrossEntropyLoss 内部会做 softmax + 取 log，所以模型不要自己加 softmax
  BN        —— BatchNorm：把每个通道的激活值标准化，训练更稳、收敛更快
  GAP       —— Global Average Pooling：把 6×6 的特征图对空间取平均 → 1 个数，
               好处是参数少、对位置没那么敏感
  ReLU      —— 激活函数 max(0, x)，给网络引入非线性

直接运行本文件 = 实跑验证：
  ① 打印网络结构、每层参数量、总参数量；
  ② CPU 上前向一次，检查输出形状 (7) 且 softmax 概率和 = 1；
  ③ GPU 上前向+反向，实测「每 128 张耗时」，估算 1 个 epoch 要多久。

运行（PowerShell）：
    $env:PYTHONIOENCODING='utf-8'
    python <项目根>/代码/04_model.py
"""
import os
import pathlib
import sys
import time

# 本机 conda 与 torch 各带一份 Intel OpenMP；torch + matplotlib 一起用会报 OMP Error #15，
# 这个环境变量允许重复加载（详见 PROJECT.md 第 3.2 节第 5 条）。必须在 import torch 之前设！
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Windows 控制台默认 GBK，打印 ✅/❌ 会抛 UnicodeEncodeError；统一改成 UTF-8 输出
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
import torch.nn as nn

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 48


class EmotionCNN(nn.Module):
    """7 类表情分类 CNN。输入 (N, 1, 48, 48) 灰度图，输出 (N, 7) 的 logits。

    两个「档位」（靠 build_model(profile=...) 选择，训练脚本用 --profile 指定）：
      baseline：通道 32/64/128，GAP → 直接 FC(128→7)，约 9.4 万参数（PROJECT.md 里的 Baseline）
      wide    ：通道 64/128/256，GAP → FC(256→256) → Dropout → FC(256→7)，约 44 万参数
    实测 baseline 训练到 30 轮训练准确率只有 ~53%（明显欠拟合），所以加了 wide 档位做对照。
    """

    def __init__(self, num_classes: int = NUM_CLASSES, widths: tuple = (32, 64, 128),
                 hidden: int | None = None, dropout: float = 0.0, use_gap: bool = True):
        super().__init__()
        c1, c2, c3 = widths
        # 记下结构参数，保存权重时一起存，推理时按同样结构重建模型
        self.cfg = {"widths": tuple(widths), "hidden": hidden, "dropout": dropout, "use_gap": use_gap}
        self.use_gap = use_gap
        # 特征提取部分：3 个「卷积 + BN + ReLU + 池化」块
        self.features = nn.Sequential(
            # 块 1：1 通道 → c1 通道；3×3 卷积、padding=1 保持边长；池化后 48→24
            nn.Conv2d(1, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 块 2：c1 → c2；池化后 24→12
            nn.Conv2d(c1, c2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # 块 3：c2 → c3；池化后 12→6
            nn.Conv2d(c2, c3, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # 池化方式：
        #   use_gap=True  → 全局平均池化 (N,c3,6,6) → (N,c3)，参数少、对位置不敏感（baseline/wide）
        #   use_gap=False → 直接展平 (N,c3*6*6)，保留空间位置信息，参数多但表达能力强（flat 档位）
        self.gap = nn.AdaptiveAvgPool2d(1) if use_gap else nn.Identity()
        flat_dim = c3 if use_gap else c3 * 6 * 6
        # 分类头：baseline 直接 FC；wide/flat 多一层隐藏层 + Dropout（防止过拟合）
        self.hidden = hidden
        if hidden:
            self.fc1 = nn.Linear(flat_dim, hidden)
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden, num_classes)
        else:
            self.fc = nn.Linear(flat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)          # (N, c3, 6, 6)
        x = self.gap(x)               # use_gap 时 → (N, c3, 1, 1)；否则原样通过
        x = torch.flatten(x, 1)       # (N, c3) 或 (N, c3*6*6)
        if self.hidden:
            x = torch.relu(self.fc1(x))
            x = self.dropout(x)
            return self.fc2(x)        # (N, 7) logits（不要加 softmax！）
        return self.fc(x)             # (N, 7) logits（不要加 softmax！）


# 三档模型配置：
#   baseline 与 PROJECT.md 里写的原始方案一致（GAP + 直接 FC）
#   wide     实测欠拟合后加的加强版（更宽通道 + 隐藏层）
#   flat     GAP 换成直接展平，保留空间信息（参数量 ~237 万，表达力更强）
PROFILES = {
    "baseline": {"widths": (32, 64, 128), "hidden": None, "dropout": 0.0, "use_gap": True},
    "wide": {"widths": (64, 128, 256), "hidden": 256, "dropout": 0.3, "use_gap": True},
    "flat": {"widths": (64, 128, 256), "hidden": 256, "dropout": 0.4, "use_gap": False},
}


def build_model(num_classes: int = NUM_CLASSES, profile: str = "baseline") -> EmotionCNN:
    """给 05_train.py / 06_evaluate.py / 07_demo_app.py 复用的工厂函数。"""
    if profile not in PROFILES:
        raise ValueError(f"未知 profile：{profile}，可选 {list(PROFILES)}")
    return EmotionCNN(num_classes=num_classes, **PROFILES[profile])


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """返回 (总参数量, 可训练参数量)。"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def get_device() -> torch.device:
    """有 GPU 用 GPU，没有就用 CPU（有 torch 但 CUDA 不可用时也能跑）。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    print("=" * 82)
    print("阶段 2：CNN 模型定义（Baseline CNN）")
    print("=" * 82)
    print(f"  torch：{torch.__version__}   CUDA 可用：{torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU：{torch.cuda.get_device_name(0)}")

    model = build_model()
    total, trainable = count_parameters(model)
    print(f"\n① 网络结构（总参数 {total:,}，可训练 {trainable:,}）")
    print(model)

    # 两档配置对比：baseline 是 PROJECT.md 里的原方案；wide 是实测欠拟合后加的加强版
    print("\n  两档配置（训练脚本用 --profile 选择）：")
    for name in PROFILES:
        m = build_model(profile=name)
        t, _ = count_parameters(m)
        print(f"    {name:9s} widths={str(m.cfg['widths']):14s} hidden={str(m.cfg['hidden']):5s} "
              f"dropout={m.cfg['dropout']:.1f} → 参数 {t:,}")

    # 逐层参数明细：方便理解"参数都花在哪了"
    print("\n  每层参数量：")
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.BatchNorm2d, nn.Linear)):
            n = sum(p.numel() for p in module.parameters())
            print(f"    {name:22s} {type(module).__name__:12s} {n:>8,}")

    # ② CPU 前向：检查形状和 softmax 概率和
    print("\n② CPU 前向验证")
    model_cpu = build_model().eval()
    dummy = torch.randn(4, 1, IMG_SIZE, IMG_SIZE)     # 假装 1 个 batch 有 4 张图
    with torch.no_grad():
        logits = model_cpu(dummy)
    probs = torch.softmax(logits, dim=1)
    print(f"  输入 {tuple(dummy.shape)} → logits {tuple(logits.shape)}（应为 (4, 7)）")
    print(f"  softmax 后每行概率和 = {probs.sum(dim=1).tolist()}（应全为 1.0）")
    print(f"  第 1 张图预测类别 = {CLASS_NAMES[int(probs[0].argmax())]}（随机初始化，结果无意义，只看形状）")
    assert logits.shape == (4, NUM_CLASSES), "输出形状不对！"
    assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5), "softmax 概率和不为 1！"

    # ③ GPU 前向 + 反向测速，估算 1 个 epoch 时间（两档配置都测）
    print("\n③ GPU 前向 + 反向测速（batch=128）")
    device = get_device()
    print(f"  使用设备：{device}")
    steps_per_epoch = 28709 // 128 + 1

    def bench(profile: str) -> tuple[float, float, float]:
        """对某个档位的模型跑 20 步「前向+反向+更新」，返回 (每步秒数, 首步 loss, 末步 loss)。"""
        model = build_model(profile=profile).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        x = torch.randn(128, 1, IMG_SIZE, IMG_SIZE, device=device)
        y = torch.randint(0, NUM_CLASSES, (128,), device=device)

        def one_step() -> float:
            """做一次「前向 + 算 loss + 反向 + 更新」，返回 loss。"""
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            return loss.item()

        for _ in range(3):                 # 预热：第一次跑包含 CUDA 初始化等开销，不计入计时
            one_step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        losses = [one_step() for _ in range(20)]
        if device.type == "cuda":
            torch.cuda.synchronize()
        return (time.time() - t0) / 20, losses[0], losses[-1]

    for profile in PROFILES:
        dt, loss0, loss1 = bench(profile)
        print(f"  {profile:9s} 每步(128张) {dt*1000:6.1f} ms → 估算 1 epoch {dt*steps_per_epoch:5.1f}s"
              f" | 20 步 loss {loss0:.3f}→{loss1:.3f}（确认反向传播能更新参数）")

    print("\n" + "=" * 82)
    print("结论：模型定义完成 ✅")
    print("  验收：① baseline 9.4 万 / wide 44 万 / flat 273 万参数  ② CPU/GPU 前向形状正确  ③ 反向传播可正常更新参数")
    print("  下一步 → 05_train.py（--profile baseline|wide|flat 选择档位）")
    print("=" * 82)


if __name__ == "__main__":
    main()
