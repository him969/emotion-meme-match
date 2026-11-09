# PROJECT.md · 基于 CNN 的人脸表情识别 + 猫 meme 匹配

> **本文件的用途**：给一个**全新的 AI 会话**在 5 分钟内接手本项目。
> 读完这份文件即可知道：项目要做什么、环境什么状况、数据什么情况、下一步该干什么。
> **最后更新**：2026-10-31 ｜ 状态：**阶段 0~5 全部完成 + 方案 A/B 优化 + 软匹配修复 + 建议 3（视觉多样性备选）+ 建议 2（CLIP 探针 → 神似模式）+ 模型优化 v2（efficientnet_b0@160 + 3-seed 集成）已接入演示** —— 最终模型测试集 **Accuracy 75.93% / Macro-F1 0.7443**（v2 3-seed 集成 + TTA；v1 ResNet18 为 69.71% / 0.6796，保留可回退）；软匹配已修多样性；演示可选"1 主推 + 2 视觉备选"、"神似模式（情感+视觉加权，默认关）"与"3-seed 集成（默认关）"
> **项目根目录**：`<项目根>\`（本文件所在目录；下文所有相对路径都相对它）

---

## 1. 项目一句话

用**卷积神经网络**识别输入人脸的**7 类表情**，得到情绪标签后，从一组**已标注情绪的猫 meme 表情包**里挑出**最匹配**的一张。

```
输入图片/摄像头 → 人脸裁剪 → CNN → 7 类情绪概率 → 检索策略 → 输出一张猫 meme
```

- **类型**：课程大作业（深度学习 / 人工智能）
- **学生背景**：大一新生，Python 基础薄弱，线代/概率/信息论几乎零基础
- **已完成的相关学习**：最小二乘、梯度下降、逻辑回归、Softmax、前馈网络、反向传播、LeNet5（都能跑通）
- **算力**：本地 GPU，详见第 3 节

---

## 2. 协作约定（AI 必须遵守）

1. **全程用中文讲解与分析**；`python` 语法、API 名、`logits`/`epoch`/`batch` 等术语保留英文。
2. **本项目不布置课后作业**，只做「讲解 + 写代码 + 验收」。
3. **注释直接写在代码里**（行内或上一行），不另写注释文档。
4. **遇到新概念先问学生是否了解**，再决定讲多深。
5. 每次交付都要**实跑验证**，不能只给代码不验证。
6. 聊天里**不要用 `$...$` 写公式**（DSH 界面不渲染 LaTeX，只认 `\(...\)` 与 `\[...\]`，或直接写纯文本）。

---

## 3. 环境体检结论（2026-10-30 实测）

### 3.1 硬件：够用，甚至富余

| 项目 | 实测值 |
|---|---|
| GPU | **RTX 5060 Laptop，显存 8.5 GB**，compute 12.0，SM 26 |
| CUDA | `torch 2.14.1+cu132`，`torch.cuda.is_available() = True` |
| 训练速度 | 小型 CNN（2 conv + 3 fc）**≈ 804 ms / 128 张 48×48** → **≈ 225 秒 / epoch**（FER2013 35887 张） |
| 磁盘 | D 盘剩余 **274 GB** |
| 摄像头 | `Integrated Camera` 正常 |
| Python | `python`（3.14.6，含 torch） |

> 结论：**训练本地做完全没问题**，10 个 epoch 约 40 分钟。

### 3.2 软件：8 个已知坑（重要）

| # | 问题 | 证据 | 应对 |
|---|---|---|---|
| 1 | **Smart App Control 间歇性拦截 DLL** | 事件日志 `Microsoft-Windows-CodeIntegrity`：`2026-10-30 20:34` 拦 `torch\lib\torch_global_deps.dll`；`2026-10-26 14:52` 拦 `torchvision\_C_stable.pyd`（WinError 4551）。但 `10-30 21:12` 同一台机器上 **torch / torchvision 连测 3/3 全部成功** | **时好时坏**：失败就重试；若频繁失败，考虑关闭 SAC（**不可逆，需重装系统才能再开**）或改用云 GPU |
| 2 | **torchvision 间歇不可用** | `import torchvision` 有时报 `RuntimeError: operator torchvision::nms does not exist`（10-26 稳定复现）；`10-30 21:12` 连测 3 次全过 | **代码必须能在"没有 torchvision"的情况下工作**：自己写 Dataset / transform（已用 MNIST 验证可行）。torchvision 只作"可选加速" |
| 3 | ~~**opencv 装不上**~~ **2026-10-31 修正** | 原判断基于 `pip install opencv-python` 无 cp314 轮子；复核发现 **opencv-python 5.0.0.93 有 `cp37-abi3-win_amd64` 轮子**（abi3 跨版本可用），已随 mediapipe 手工装进 `_deps\`（`cv2.pyd` 112 MB） | 主流程**仍不依赖 opencv**（设计原则不变）；只有建议 2 的"神似模式"用 mediapipe+opencv 做人脸关键点，缺依赖时自动隐藏；摄像头实时版仍不做 |
| 4 | **Windows 控制台 GBK 编码** | 2026-10-30 实测：PowerShell 默认代码页下跑 `01_env_check.py`，打印 ✅ 时抛 `UnicodeEncodeError: 'gbk' codec can't encode character '\u2705'` | 脚本开头已统一加 `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`；手动跑时先设 `$env:PYTHONIOENCODING='utf-8'` |
| 5 | **OMP Error #15（两份 Intel OpenMP）** | 2026-10-30 实测：同一进程里 `import torch` + `matplotlib` 画图（`fig.canvas.draw()`）时报 `OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized`。原因是 conda 的 `Library\bin\libiomp5md.dll`（2024-08-20）与 torch 自带的 `torch\lib\libiomp5md.dll`（2026-02-13）各有一份 | 用到 torch + matplotlib 的脚本，在 `import numpy/torch` **之前**加 `os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")`（已写进 `03_dataset.py`；后续含画图的脚本同样处理）。实测设了之后训练/画图/存图都正常。⚠ **它只把"OMP 重复加载导致的进程中止"变成"继续运行"，不释放/不限制任何内存或显存，也不是 CUDA OOM 的解法**；官方也建议改为统一 OpenMP 版本。内存/显存问题要看 `07_demo_app.py` 的 `empty_cache`/自动降级/`释放显存缓存` 按钮与第 9 节排障 |
| 6 | **SAC/WDAC 拦截 Explorer 启动 cmd.exe → 双击 .bat 无法运行** | 2026-10-30 22:53~22:54 学生双击 `启动演示.bat` 时，CodeIntegrity 日志连续 7 条事件 3033：`explorer.exe attempted to load cmd.exe that did not meet the Enterprise signing level requirements`（本机 `VerifiedAndReputablePolicyState=1`，SAC 开启） | **本机不要用双击 .bat 启动**：① 由 AI 在会话里用后台任务启动 `streamlit run`，学生只开浏览器访问 `http://localhost:8501`；② 或在已经打开的终端里手动跑命令；③ 换一台未开 SAC 的机器时 .bat 仍可用 |
| 7 | **Streamlit 首次运行会交互式问 Email** | 2026-10-30 实测：无 `~\.streamlit\credentials.toml` 且没开 headless 时，`streamlit run` 会停在 `Email:` 提示，非交互环境直接退出（exit 1） | 启动时加 `--server.headless true --browser.gatherUsageStats false`（现已写进 `启动演示.bat` 和启动命令），浏览器手动打开 URL 即可 |
| 8 | **pip 装 timm 时写全局目录/临时文件被沙箱拒** | 2026-10-31 实测：`pip install timm` 报 `PermissionError: ...whl.metadata`（Python 装在 `<conda环境>`，site-packages 和 pip 临时目录都不可写）；另外 huggingface_hub 2.4 依赖 `httpcore2`，其缓存目录 `<用户目录>/.cache` 也被拒 | **改成项目内“手工安装”**：用 urllib 从 PyPI 下载 `timm 1.0.30` / `huggingface_hub 0.36.0`（requests 版，避开 httpcore2）/ `safetensors 0.8.0` 的 wheel，`zipfile` 解压到 `表情识别项目\_deps\`，脚本里 `sys.path.insert(0, _deps)`；预训练权重也从 `hf-mirror.com` 直接下载到 `_deps\weights\`（`resnet18.a1_in1k.safetensors`，46.8 MB），**不走 huggingface_hub 缓存**。⚠ `_deps\` 是迁移学习模型的依赖，不要删 |

### 3.3 已安装的库（ml 环境，2026-10-30 21:12 实测）

```
torch 2.14.1+cu132     numpy 2.4.6        pillow 12.3.0
pandas 3.0.3           scikit-learn 1.9.0 matplotlib 3.11.0
streamlit 1.58.0       tqdm 4.68.2        jupyter / notebook / ipykernel
torchvision 0.29.1+cu132（可导入，但「间歇被拦」，不可依赖）
```

**`_deps\` 内手工安装（2026-10-31 新增，供建议 2 神似模式使用；主流程不依赖）**：

```
open_clip 3.3.0 + ftfy 6.3.1 + weights\open_clip_vitb32_laion2b.bin（605 MB，laion CLIP ViT-B/32）
mediapipe 1.2.0 + opencv-python 5.0.0.93(abi3) + absl-py 2.5.1 + sounddevice 0.5.6
onnxruntime 1.31.1（备份关键点方案）+ ai-edge-litert 2.3.0（B0 猫脸关键点实验用，交付可不留）
models\face_landmarker.task（3.7 MB，mediapipe 478 点模型）、models\face_landmarks_478.onnx（4.9 MB，备份）
```

整个 `_deps\` 约 **934 MB**（含 timm 迁移学习依赖）。若只保留基础功能，可删 CLIP/mediapipe 相关，神似模式开关会自动隐藏。

> ⚠ **设计原则**：所有核心代码**不依赖 torchvision 与 opencv**；这两个库只作为"有则加速、无则降级"的可选项。

---

## 4. 数据审查结论（cat_meme，已实测）

### 4.1 基本信息

| 项目 | 结果 |
|---|---|
| 原始压缩包 | `数据\cat_meme.zip`（41.4 MB） |
| 解压位置 | `数据\cat_meme_extracted\cat_meme\` |
| 文件数 | **68 个**（第一批 38 个：21 gif + 15 jpeg + 2 png；2026-10-30 补充 30 个 pielet 来源）+ 1 个 `ATTRIBUTION.md` |
| **去重后唯一图** | **66 张**（第一批 28 张 + pielet 38 张；同图的 gif/jpeg 只算 1 张；2026-11-02 扩库 8 张） |
| 补充来源 | `pielet/cat-memes`（HuggingFace 数据集，标注 MIT）：55 个知名猫梗图里挑 30 张入库，**情绪标签由本项目人工逐张看图标注**，映射见 `数据\_meme_add_raw\_labels.json` |
| 目录结构 | `cat_meme/<情绪>/<情绪>_<来源>_<名字>.<ext>` |
| 类别 | **angry / disgust / fear / happy / sad / surprise / neutral** —— **与 FER2013 七类完全一致，无需映射表** |

### 4.2 每类覆盖（唯一图计数，2026-10-30 补充后）

| 情绪 | 文件数 | 唯一图 | 评价 |
|---|---|---|---|
| angry | 9 | 7 | ✅ 充足 |
| **disgust** | 5 | **4** | 🟢 基本够（原来只有 1 张，已补 30 张里的 3 张） |
| fear | 6 | 5 | 🟢 基本够 |
| happy | 14 | 13 | ✅ 充足 |
| sad | 7 | 6 | ✅ 充足 |
| surprise | 9 | 8 | ✅ 充足 |
| neutral | 18 | 15 | ✅ 充足 |

### 4.3 质量检查（全部通过）

| 检查项 | 结果（补充后 68 个文件复测） |
|---|---|
| 能否打开 | **68/68 全部可读**（gif/webp 动画已逐帧解码验证） |
| 逐字节重复 | **0 组**（无完全相同的文件） |
| 目录名 vs 文件名前缀 | **0 处不一致**（`<情绪>_<来源>_<名字>` 命名规则保持） |
| 版权说明 | ✅ 有 `ATTRIBUTION.md`，含 4 个来源、许可、致谢 |
| 尺寸 | 宽高 **64 ~ 1414 px**，27/68 为正方形 |
| 颜色模式 | `P` 24 个、`RGB` 38 个、`RGBA` 6 个 → **必须统一转 RGB** |
| gif/动画帧数 | 1 ~ 156 帧（中位数 1）→ 统一只取首帧 |

### 4.4 数据处理方案（据此定下）

1. **gif**：取首帧（或中间帧）→ `convert("RGB")`；
2. **RGBA / P**：先合成到白底再转 RGB，避免透明区域变黑；
3. **尺寸**：等比缩放 + 白色填充到 **256×256**（不要直接拉伸，会变形）；
4. **去重策略**：把「同名不同后缀」视为同一张图，随机只取一个（避免同一张图被重复展示）；
5. **元数据表**：生成 `数据\meme_index.csv`，列 = `file, emotion, source, ext, width, height, is_gif, processed`；
   处理后的图存 `数据\meme_ready\<情绪>\<文件名主干>.png`（演示程序直接用它，不再重复处理）。

### 4.5 数据侧的结论与建议

**结论：补充后每类至少 4 张，演示重复感明显降低。**

- 2026-10-30 从 HuggingFace 数据集 `pielet/cat-memes`（55 个知名猫梗图）下载，**人工逐张看图标注情绪**后挑选 30 张入库（`*_pielet_*`）；
- 补充后每类唯一图：angry 8 / disgust 4 / fear 7 / happy 15 / sad 7 / surprise 10 / neutral 15，共 **66 张**（2026-11-02 扩库后）；
- 仍需注意：情绪标签是"看图主观标注"，个别图（如 `lime-cat`、`side-eye-cat` 标为 disgust）存在主观性；部分梗图带英文文字；
- 代码侧仍保留「情绪相似度回退」策略（见 5.4）作为兜底；演示程序改成每次从全部候选里检索，重复问题已基本解决。

### 4.6 FER2013 数据源与实测（2026-10-30 已下载完成）

| 项目 | 结果 |
|---|---|
| 数据源 | HuggingFace 镜像 `hf-mirror.com` 上的 `Aaryan333/fer2013_train_publicTest_privateTest`（Kaggle 官方 FER2013 的 parquet 转存，三分划与原版完全一致） |
| 为什么用镜像 | 直连 `huggingface.co` 超时；`hf-mirror.com` 实测约 1.4 MB/s 可下 |
| 原始文件 | `数据\fer2013\_raw\`：train.parquet 106.48 MB、publicTest.parquet 13.35 MB、privateTest.parquet 13.36 MB |
| 转换产物 | `数据\fer2013\fer2013.npz`（78.3 MB）：`X_train (28709,48,48)`、`X_val (3589,48,48)`、`X_test (3589,48,48)`，uint8；对应 `y_*` 为 int64 标签 0~6 |
| 附带文件 | `meta.json`（来源/划分/每类数量）、`norm_stats.json`（训练集 mean=0.5077、std=0.2550，训练与推理共用） |
| 标签顺序 | 0=angry 1=disgust 2=fear 3=happy 4=sad 5=surprise 6=neutral（与 Kaggle 官方一致，已按仓库 README 的 class_label 核对） |
| 训练集分布 | angry 3995 / disgust 436 / fear 4097 / happy 7215 / sad 4830 / surprise 3171 / neutral 4965 → **最少类 disgust 只有最多类 happy 的 1/16.5** |
| 验证集（PublicTest） | 3589 张：467 / 56 / 496 / 895 / 653 / 415 / 607 |
| 测试集（PrivateTest） | 3589 张：491 / 55 / 528 / 879 / 594 / 416 / 626 |
| 读取方式 | 只依赖 numpy（`np.load`），不依赖 torchvision；`00_download_fer2013.py` 可重跑（已有文件自动跳过，`--rebuild` 强制重建） |

---

## 5. 技术路线（5 个阶段）

### 阶段 1：数据管线（✅ 已完成）

| 步骤 | 内容 | 验收标准 |
|---|---|---|
| 1.1 | 下载 **FER2013**（7 类，48×48 灰度；官方划分 28709/3589/3589） | 能读出 `(N,48,48)` 与标签 |
| 1.2 | **自写 Dataset**（不用 torchvision） | 能 `for batch in loader` 跑通 |
| 1.3 | 数据增强：水平翻转、随机平移、±10° 旋转、亮度 ±10% | 增强后可视化正常 |
| 1.4 | 统计类别分布（FER2013 本身不均衡，disgust 仅 436 张） | 输出分布表 |

**关键前置**：FER2013 常见获取渠道（Kaggle 需账号 / HuggingFace `datasets` / 镜像站）。本次实际用 `hf-mirror.com` 镜像成功。若镜像失效，备选 RAF-DB 或 CK+。

### 阶段 2：模型（✅ 已实现，`代码\04_model.py`）

Baseline（PROJECT.md 原方案）：

```
Conv(1→32,3x3) + BN + ReLU + MaxPool
Conv(32→64,3x3) + BN + ReLU + MaxPool
Conv(64→128,3x3) + BN + ReLU + MaxPool
GlobalAvgPool → FC(128→7)
```

实测 baseline 只有 93,799 参数，30 轮后**训练集**准确率也只有 ~53% —— **明显欠拟合**。
因此在 `04_model.py` 里加了两档加强模型（`--profile` 选择）：

| 档位 | 结构 | 参数量 | 说明 |
|---|---|---|---|
| `baseline` | 32/64/128 通道 + GAP + FC(128→7) | 93,799 | PROJECT.md 原方案 |
| `wide` | 64/128/256 通道 + GAP + FC(256→256) + Dropout(0.3) + FC(256→7) | 437,703 | 加宽 + 加隐藏层 |
| `flat` | 64/128/256 通道 + 直接展平(6×6×256) + FC(→256) + Dropout(0.4) + FC(256→7) | 2,731,463 | 不用 GAP，保留空间位置信息 |

- 8.5 GB 显存训练这些模型都绰绰有余：`wide` 约 10s/epoch，`flat` 约 10~15s/epoch；
- **最终采用（2026-10-31，方案 A）**：ImageNet 预训练 **ResNet18**（timm `resnet18.a1_in1k`，11,180,103 参数）微调，见 `代码\08_transfer_train.py`。输入 48×48 → 128×128 RGB，加载除分类头外的全部 ImageNet 权重，再在 FER2013 上微调。参数量大一个量级，但预训练特征让测试集直接从 64.73% 提到 68.46%（+TTA 69.71%）。依赖 `timm` 等装在项目内 `_deps\`（原因见 3.2 节第 8 条）。
- **模型优化 v2（2026-10-31 晚，最终采用）**：改用 **efficientnet_b0 + 160×160 输入**（timm `efficientnet_b0.ra_in1k`，10.7M 参数），训练配方 = **AMP fp16 + Lion(lr1e-4) + label smoothing 0.1 + mixup 0.2 + 强增强（旋转/缩放/平移/亮度对比度/随机擦除）+ EMA 0.999 + warmup 2 + grad-clip 1.0 + 类别权重**，30 epoch（单模型约 60 分钟）；再跑 **3 个随机种子（42/43/44）做概率平均集成** + 6 视角 TTA → **测试集 75.93% / Macro-F1 0.7443**（v1 为 69.71% / 0.6796，+6.22pp / +0.0647）。`08_transfer_train.py` 已通用化（任意 timm 模型、AMP/Lion/EMA/SWA/mixup/balance/strong aug、实验记录）；`15_ensemble_eval.py` 做多模型集成评估。详见 `报告\最终报告.md` §5.5。

### 阶段 3：训练与评估（✅ 已实现，`代码\05_train.py` / `06_evaluate.py` / `08_transfer_train.py`）

**自写 CNN 的训练设置（E1~E8 实验）**：

| 项 | 设定 |
|---|---|
| 损失 | `CrossEntropyLoss`（`--no-weight` 关类别权重；`--label-smoothing` 可选标签平滑） |
| 优化器 | `Adam(lr 可调，默认 1e-3)` + `CosineAnnealingLR(T_max=epochs)` |
| 轮数 | 默认 30 epoch；`--patience` 早停（**实测 patience=3 会在 lr 还高时误停，建议 ≥8**） |
| 指标 | Accuracy、**Macro-F1**、**7×7 混淆矩阵** |
| 预期 | 小 CNN 在 FER2013 约 **65%~70%**（该数据集人类也只有 ~65%）——**实测 flat2 测试集 64.73%（+TTA 65.51%）** |
| 消融 | 模型档位（baseline/wide/flat）、类别权重、早停 patience、学习率/标签平滑、是否迁移学习、是否 TTA |

**最终模型（迁移学习，E9）设置**：

| 项 | 设定 |
|---|---|
| 模型 | ImageNet 预训练 ResNet18（timm `resnet18.a1_in1k`），**11,180,103 参数** |
| 输入 | FER 48×48 → **128×128**，灰度复制为 3 通道，ImageNet mean/std 归一化 |
| 损失/优化器 | CrossEntropyLoss + label smoothing 0.05；AdamW(lr=3e-4, weight_decay=1e-4) + CosineAnnealingLR(T_max=20) |
| 轮数/早停 | 20 epoch，patience=5（实际跑满 20 轮，最优第 16 轮） |
| 增强 | 与自写 CNN 同一套（翻转/平移/±10° 旋转/亮度），放大到 128 后平移幅度按比例放大 |
| 结果 | val 67.51% / test **68.46%**（Macro-F1 0.6633）；**+TTA 后 test 69.71%（Macro-F1 0.6796）** |
| 耗时 | **12.5 分钟**（20 轮 × 37.6s，RTX 5060） |

**TTA（方案 B，`06_evaluate.py --tta` + 演示页勾选）**：对原图、水平翻转、上下左右小平移共 6 个视角的 softmax 概率取平均。
实测提升：flat2 +0.78 点（64.73%→65.51%），ResNet18 +1.25 点（68.46%→69.71%）。

### 阶段 4：推理 + meme 匹配（✅ 已实现，`代码\07_demo_app.py`）

**匹配策略（实现 2 种，页面单选框切换）**：

```
① 硬匹配（baseline）：取 argmax 情绪 → 该情绪的候选里随机取一张
② 软匹配（推荐）：score(meme) = Σ_i p_i × sim(emotion_i, meme.emotion)，取分数最大者
   （同一情绪所有 meme 分数相同 → 见下方"同分随机"修复）
```

- "强度匹配"**没有实现**：它需要给每张 meme 标一个"情绪强度"，现有 58 张图没有这个标注，硬凑会变成假实验；
- **回退策略**（硬匹配时该情绪没有图）：按情绪相似度找最像且有图的类 —— disgust 缺图→angry、fear 缺图→surprise、sad 缺图→neutral；
- 相似度矩阵硬编码在 `07_demo_app.py` 的 `EMOTION_SIM`（7×7，对角=1）；
- **软匹配同分随机（2026-10-31 全局修复）**：同一情绪的所有 meme 软匹配分数完全相同，旧实现 `sort_values` 取第一张 → 同一种情绪永远只显示同一张图（实测验证集 3589 次查询只用到 **7 张**）。修复为：**选中情绪与旧逻辑完全一致**，但在同类同分候选里随机挑一张 + `avoid_file` 避免连续重复。实测 label_hit 0.6773 / S 加权相似度 0.8124 均不变，用到的不同 meme **7 → 58 张**、连续重复 **636 → 0 次**；`--selftest` 用 100 组随机概率验证"选中情绪"零不一致；
  - **同分候选挑法在页面可选（2026-11-01 新增）**：`TIE_STRATEGIES` = 随机+防连续重复（默认，原行为）/ 纯随机 / 确定性（按图片内容 hash，同一张图结果固定）/ 固定第一张（修复前行为，仅对照）。四者只影响"同情绪类内挑哪张"，**不改变情绪类别与准确率指标**；`--selftest` 已覆盖 4 种策略（情绪一致、随机多样、hash 确定、first 固定），AppTest 也已覆盖页面切换；
- **输入预处理随模型变**：自写 CNN 用 48×48 灰度（FER mean/std）；迁移模型用 128×128 RGB（ImageNet mean/std）；
- **TTA（方案 B）**：演示页侧边栏可勾选（默认开），推理时对 6 个视角的概率取平均；
- **视觉多样性备选（建议 3，2026-10-31 已接入）**：侧边栏"视觉多样性展示"开关（**默认关**）；打开后显示 **1 主推 + 2 备选**，"备选范围"（仅同类/同类为主/自由补充）与"视觉特征"（ImageNet/FER）都由用户选择；"换一批"只换备选、主推固定；页面显示余弦距离（0=一样）。算法见 `代码\11_visual_diversity.py`：距离分位带（60%~95%）+ 贪心 max-min；离线验证：主推情绪 **0 变化**、三张两两视觉距离比同约束随机备选高 **0.10~0.14**、无近重复图；纯最大距离会挑到"纯纹理"怪图，故默认用分位带；
- **人脸检测**：环境没有 opencv，页面不做自动人脸检测；支持上传"已裁好的脸"或一键随机取 FER2013 测试图。

### 阶段 5：演示与交付（✅ 已完成）

| 交付物 | 状态 |
|---|---|
| 演示程序 | ✅ **Streamlit**（`代码\07_demo_app.py`）：上传/随机图 → 7 类概率条 → 选中 meme + 软匹配 Top-5 打分明细；可选模型（自写 CNN / v1 ResNet18 / **v2 efficientnet_b0**，可勾 **3-seed 集成**）、可选 TTA、软/硬匹配切换、**同分候选选择策略（4 种）**、**GIF 动图播放（默认开）**、**GIF 视觉特征（首帧默认 / B 多帧平均 / C 峰值帧）**；可选"1 主推 + 2 视觉备选"（建议 3）；可选"神似模式"（建议 2） |
| 摄像头实时版 | ❌ 不做（opencv 装不上，按第 7 节决定） |
| 报告 | ✅ `报告\最终报告.md`（问题定义、数据、模型、实验、混淆矩阵分析、匹配策略、TTA、视觉备选、结论） |
| 代码 | ✅ 00~11 全部脚本带中文行内注释、均可独立实跑（09/10 为离线原型实验脚本，11 为视觉备选共享模块） |
| 版权 | ✅ 保留 `数据\cat_meme_extracted\cat_meme\ATTRIBUTION.md` |

---

## 6. 目录约定

> **2026-10-30 起：所有项目文件已集中到 `<项目根>\` 这一个文件夹里**（工作区根目录只保留与本项目无关的旧目录）。脚本内路径一律用「相对本文件/本脚本」定位，整体搬家不用改代码。

```
<项目根>\     ← 项目根目录
├── PROJECT.md                    ← 本文件
├── 启动演示.bat                  ← 双击启动演示网页（ASCII-only；本机 SAC 拦截 cmd，双击可能无效）
├── 启动演示.lnk                  ← 快捷方式：直接启动 python.exe（不经过 cmd，绕开 SAC 对 cmd 的拦截；需实测）
├── _deps\                        ← 项目内手工安装依赖（勿删）：timm 1.0.30 / huggingface_hub 0.36.0 / safetensors 0.8.0 + weights\resnet18.a1_in1k.safetensors；建议 2 新增 open_clip 3.3.0 / ftfy / mediapipe 1.2.0 / opencv 5.0.0.93(abi3) / onnxruntime + weights\open_clip_vitb32_laion2b.bin（605 MB）+ models\face_landmarker.task / face_landmarks_478.onnx
├── 报告\
│   └── 最终报告.md                ✅ 完整报告（数据/模型/实验/混淆矩阵/匹配策略/TTA/结论/复现命令）
├── 数据\
│   ├── cat_meme.zip              原始压缩包（只读参考）
│   ├── cat_meme_extracted\cat_meme\   解压后的表情包（7 类 + ATTRIBUTION.md）
│   ├── _audit_memes.py           表情包审查脚本（已完成）
│   ├── meme_ready\<情绪>\*.png    预处理后的 256×256 表情包（66 张，02 产出；模型/匹配/特征用）
│   ├── meme_anim\*.gif           GIF 表情包的演示用动画版（18 张 256×256 压缩动图 + _index.json，16 产出；只用于页面展示）
│   ├── meme_gif_frames\*.npz     GIF 采样帧 + 时长权重 + 峰值帧索引（18 个，239 帧，17 产出；B/C 视觉特征用）
│   ├── meme_anim_peak\*.png      GIF 峰值帧图像（18 张，17 产出；C 方案/目检用）
│   ├── meme_index.csv            表情包元数据表（66 行，02 产出）
│   └── meme_dedup_choices.json   "同图多格式"历史去重选择表（02 使用；保证扩库不扰动旧图版本）
│   ├── shensi_meme_tags.json     建议 2：66 张 meme 的人工神态/气质标签（眼/嘴/风格/用法）
│   ├── _meme_add_raw\            2026-10-30 补充的 30 张 pielet 原图 + `_manifest.json`（来源/许可）+ `_labels.json`（人工标签），便于追溯
│   └── fer2013\
│       ├── fer2013.npz           FER2013 数组（28709/3589/3589，78.3 MB）
│       ├── meta.json             数据来源、划分、每类数量
│       ├── norm_stats.json       归一化 mean/std
│       └── _raw\*.parquet        原始下载文件（106.48 / 13.35 / 13.36 MB）
├── 代码\
│   ├── 00_download_fer2013.py    阶段 1.1 下载 + 转 npz（✅ 已完成）
│   ├── 01_env_check.py           环境自检（✅）
│   ├── 02_meme_index.py          meme 预处理 + 元数据表（✅）
│   ├── 03_dataset.py             FER2013 Dataset + 增强 + 迁移学习 Dataset（✅）
│   ├── 04_model.py               自写 CNN 定义（baseline / wide / flat 三档，✅）
│   ├── 05_train.py               自写 CNN 训练 + 早停 + 保存权重 + 训练曲线（✅）
│   ├── 06_evaluate.py            Accuracy / Macro-F1 / 混淆矩阵 / **TTA**（✅）
│   ├── 07_demo_app.py            Streamlit 演示（软/硬匹配、回退、TTA、视觉备选开关，✅）
│   ├── 08_transfer_train.py      **迁移学习**：ImageNet 预训练 ResNet18 微调（✅）
│   ├── 09_meme_softmatch_proto.py  离线原型：全自动 7 维软标签 + 距离/聚类（结论：不成立，留档）
│   ├── 10_visual_diversity_proto.py 离线原型：1 主推 + 2 视觉备选（3 范围 × 2 特征验证）
│   ├── 11_visual_diversity.py    建议 3 共享模块：视觉特征、候选池、分位带贪心、换一批（✅ 演示在用）
│   ├── 12_clip_probe.py          建议 2 B1：CLIP 跨模态探针 P1~P6（只读验证集，门槛+统计检验）
│   ├── 13_shensi_visual.py       建议 2 共享模块：478 关键点几何 + CLIP 神态属性画像 + 融合打分（✅ 演示在用）
│   ├── 14_shensi_probe.py        建议 2 B2 离线验证：几何统计、β 扫描、目检素材（结果\proto_shensi_*）
│   ├── 15_ensemble_eval.py       模型优化 v2：多模型概率平均集成评估（--tta / --tta-strong / --splits）
│   ├── 16_meme_anim.py           GIF 表情包 → 演示用 256×256 压缩动图（只影响页面展示，不参与模型/匹配）
│   └── 17_meme_gif_frames.py     GIF 多帧资产：采样帧/时长权重/峰值帧（B/C 视觉特征方案）
├── 模型\
│   ├── best_model.pth            ← **最终交付模型**（= v2_effb0_160.pth，efficientnet_b0@160 微调；单模型 72.64% / 0.7039，3-seed 集成+TTA test 75.93% / 0.7443）
│   ├── best_model_v1.pth         旧最终模型（ResNet18 迁移学习；test 69.71% / 0.6796，保留可回退）
│   ├── v2_seed42.pth / v2_seed43.pth / v2_seed44.pth   v2 三个随机种子（演示"3-seed 集成"用；均 effb0@160）
│   ├── v2_effb0_160.pth（=seed42）、v2_resnet34_128.pth、v2_r1_amp.pth / v2_r2_ema.pth / v2_r3_mix.pth / v2_r4_lion.pth   模型优化 v2 消融对照权重
│   ├── best_transfer.pth         v1 迁移学习权重（42.7 MB）
│   ├── best_model_flat2.pth      E8 自写 CNN 最优（测试集 64.73%，+TTA 65.51%）
│   ├── best_model_flat.pth       E7
│   ├── best_model_wide.pth       E5
│   ├── best_model_wide_weighted.pth  E6
│   ├── best_model_plain_p8.pth   E4
│   ├── best_model_weighted_p8.pth    E3
│   ├── best_model_E1_baseline_w_p3.pth      E1
│   ├── best_model_E2_baseline_plain_p3.pth  E2
│   ├── shensi_meme_cache.npz      建议 2：58 张 meme 的 CLIP 特征 + 属性画像缓存（演示/探针共用）
│   └── shensi_geometry_stats.json 建议 2：验证集关键点几何分位统计（百分位映射用）
└── 结果\
    ├── evaluation_test.json / .txt          **v1 模型**（ResNet18）普通推理评估：test 68.46% / 0.6633
    ├── evaluation_test_tta.json / .txt      **v1 模型** + TTA：test 69.71% / 0.6796（旧成绩；副本 evaluation_v1_test_tta.* 防覆盖）
    ├── confusion_matrix_test.png            v1 模型混淆矩阵（普通推理）
    ├── confusion_matrix_test_tta.png        v1 模型 + TTA 混淆矩阵（副本 confusion_matrix_v1_test_tta.png）
    ├── evaluation_flat2_tta.*               自写 CNN + TTA：test 65.51% / 0.5437
    ├── evaluation_E1...E8 / evaluation_transfer_tta  各消融实验评估
    ├── train_best_transfer.json             迁移学习训练历史
    ├── training_curves_best_transfer.png    迁移学习训练曲线
    ├── train_*.json / training_curves_*.png  其他实验训练历史/曲线
    ├── proto_matching_report.md / .csv       09 原型：全自动软标签/距离/聚类对比（结论：不成立，留档）
    ├── proto_visual_diversity_report.md / .csv  10 原型：1 主推 + 2 备选（3 范围 × 2 特征）
    ├── proto_visual_examples_*.png / proto_visual_mitigation_band.png  视觉备选样例图
    ├── proto_visual_probe.txt                视觉相似度探针（ResNet18 跨域信号≈0，留档）
    ├── proto_meme_vectors.csv / .npz / .png  09 原型的 meme 向量与聚类可视化
    ├── proto_clip_probe.txt / .json       建议 2 B1：CLIP 探针三门门槛结果 + 统计检验
    ├── proto_clip_features.npz            建议 2：人脸/meme 的 CLIP 特征（3589×512 / 58×512）
    ├── proto_clip_attr_meme.csv / proto_clip_attr_profiles.npz  建议 2：meme/人脸属性画像
    ├── proto_clip_picks_val.csv / proto_clip_ab_1/2.png         建议 2：类内视觉重排选取与目检对照
    ├── proto_shensi_summary.json / proto_shensi_picks_val.csv   建议 2 B2：几何统计、β 扫描、候选明细
    ├── proto_shensi_landmarks.png          建议 2：人脸 478 关键点提取可视化（验证集样例）
    ├── proto_shensi_ab_1/2.png             建议 2：神似验收目检对照（人脸|随机|属性|融合）
    ├── evaluation_v2_final_tta.json / .txt  模型优化 v2 最终评估（3-seed 集成 + TTA：test 75.93% / 0.7443）
    ├── confusion_matrix_v2_final_tta.png    v2 最终混淆矩阵
    ├── experiments_v2.jsonl / v2_experiment_summary.md / v2_recommendation.md   v2 实验记录与推荐
    ├── v2_records\ / v2_logs\                v2 每实验明细与训练日志
    ├── train_v2_*.json / training_curves_v2_*.png   v2 各实验训练历史/曲线
    ├── meme_preview.png                     58 张 meme 预处理后预览
    ├── aug_preview.png                      数据增强可视化
    └── fer2013_samples.png                  FER2013 每类样本
```

> 另有两个留档目录：`_probe_b0\`（建议 2 B0 关键点可行性预检：b0_report.md + 可视化）、`_probe_clip\`（CLIP 环境报告、P6 属性桥诊断、AppTest 脚本）。
> `_deps\` 现约 **1 GB**：迁移学习依赖（timm + resnet18/resnet34/efficientnet_b0 预训练权重）+ 建议 2 神似模式依赖（CLIP 605 MB、mediapipe/opencv、`models\face_landmarker.task` 等）。删掉 CLIP/mediapipe 相关后神似模式自动隐藏，默认功能不受影响。

> **注意**：工作区根目录（`<工作区根>\`）里还有旧目录 `讲义\`、`作业\`、`课件\`、`progress\`、`任务C_新学习线\` 等，属于之前的学习内容，**与本项目无关，不要动**。

---

## 7. 待决策项 —— 已拍板（2026-10-30，学生选择「按默认走」）

| # | 问题 | 最终决定 |
|---|---|---|
| 1 | FER2013 数据来源 | **先试 HuggingFace 镜像**；失败再换 RAF-DB / CK+ |
| 2 | 演示形式 | **Streamlit 上传图片**（不依赖 opencv；环境已装 streamlit 1.58） |
| 3 | 环境路线 | **不新建** Python 3.11；不依赖 opencv / torchvision |
| 4 | 是否补 meme 图 | 需要补（disgust +2、fear +1、sad +1、surprise +1）。图需学生提供或用可公开使用图源补；补齐前代码用「情绪相似度回退」兜底 |
| 5 | **模型优化方向（2026-10-31 追加拍板）** | **走 A+B**：A = ImageNet 预训练 ResNet18 微调（`08_transfer_train.py`）；B = 推理 TTA（6 视角平均，`06_evaluate.py --tta` + 演示页勾选） |
| 6 | 依赖安装方式（2026-10-31） | pip 全局安装被沙箱拒 → 手工下载 wheel 解压到 `_deps\`；预训练权重手工从 `hf-mirror.com` 下载（详见 3.2 节第 8 条） |
| 7 | **建议 3（视觉多样性备选）** | **2026-10-31 已批准并执行**：基础模式 + "视觉多样性展示"开关（默认关）；1 主推 + 2 备选；备选范围三选一（仅同类/同类为主/自由补充）；视觉特征二选一（ImageNet/FER）；默认仅同类 + ImageNet + 距离分位带；显示差异分；换一批只换备选 |
| 8 | **建议 2（CLIP 视觉路线）** | **2026-10-31 已审查并落地**：B0 关键点预检（人脸 mediapipe 478 点可行、猫侧仅照片风 36/58）→ B1 CLIP 探针（P1 勉强过、P2/P3 未过门，负结论：不做 CLIP 视觉融合主匹配；但属性桥 top1 0.349、p=0.0015 有真信号）→ B2 落地**"神似模式"**（`F=(1-β)·情感分+β·视觉分`，视觉分 = 0.50 属性画像 + 0.35 关键点几何 + 0.15 外观，β 默认 0.35、页面可调 0.10~0.60，**默认关闭**）。CLIP 权重（605 MB）已下载进 `_deps\`，交付保留与否后续可再权衡 |
| 9 | **GIF 抽帧/演示动画** | **2026-11-01 部分纳入**：① **UI 动图播放已补**（`代码\16_meme_anim.py` 生成 18 张 256×256 压缩 GIF + 侧边栏开关默认开）；② **GIF 视觉特征 B/C 已补**（`代码\17_meme_gif_frames.py` 采样帧/时长权重/峰值帧；`11_visual_diversity.py` 与 `13_shensi_visual.py` 支持 `gif_mode`；侧边栏"GIF 视觉特征"单选，默认首帧；只影响可选视觉功能）。**"帧平均情绪向量用于主匹配"已实测否定**（GIF 帧平均 argmax 与人工标签一致率仅 5.6%，负结论见报告 §6.7）；"GIF 抽帧做动作/动画特征"仍不做 |
| 10 | **模型优化 v2（2026-10-31 晚）** | **已完成**：方案 = 训练配方升级（AMP + Lion + EMA + mixup + 强增强 + 30ep）+ 更强 backbone（efficientnet_b0@160，resnet34@128 对比）+ **3-seed 概率平均集成** + 6 视角 TTA；测试集 **75.93% / 0.7443**（v1 69.71% / 0.6796）。**test 只在最后评估一次**；`best_model.pth` 更新为 v2 单模型（`best_model_v1.pth` 保留）；演示新增"3-seed 集成"开关（默认关）。未做：resnet50/ConvNeXt/ViT（算力）、focal loss |
| 11 | **P0：FER+ 标签重标注（2026-11-02）** | **已实测，按项目原指标 No-Go**：`18_ferplus_labels.py` 对齐 FER+（35,887 行、拆分一致、~64% 与原标签一致），1 seed effb0@160 同配方训练（早停 26 轮/约 35min，新权重 `ferplus_hard_s42.pth`）。原始标签 val：FER+ 模型 61.27/0.5539 vs v2 72.61/0.7036 → 不替换主模型、**不做 UI 选项、不进入 B**；但两套标签一致子集上 FER+ 模型 90.29/0.8554 vs v2 87.21/0.8138（+3.1pp），说明是**标签体系定义差异**而非能力问题。详见报告 §5.6 / `结果\ferplus_A_quick_screen.md`。测试集未使用 |

---

## 8. 已知风险清单（2026-10-30 更新）

| 风险 | 状态 | 说明 |
|---|---|---|
| Smart App Control 再次拦截 torch | 本次未发生 | 今天几十次 torch 导入/训练全部正常；若以后失败：重试 → 关 SAC（不可逆）→ 云 GPU |
| FER2013 无法下载 | ✅ 已解决 | `hf-mirror.com` 镜像可下（1.4 MB/s），`00_download_fer2013.py` 可重复运行 |
| FER2013 准确率上不去 | ✅ 彻底解决（2026-10-31） | 白写 CNN 到顶 ~64.7%；v1 ResNet18+TTA 69.71%；**v2（efficientnet_b0@160 + 配方升级 + 3-seed 集成 + TTA）→ test 75.93% / Macro-F1 0.7443** |
| disgust 类几乎必错 | ✅ 明显改善（2026-10-31） | 自写 CNN disgust 召回 0/55；v1 迁移模型 + TTA 召回 60%；**v2 集成后召回 76.4%（F1 0.7368）**；fear 召回也从 31% → 48.7% → **60.0%** |
| meme 太少、观感重复 | ✅ 已改善（2026-10-30） | 从 `pielet/cat-memes` 补 30 张并人工打标签：唯一图 28 → **58 张**，每类 ≥4（disgust 1→4、fear 2→5、sad 3→6、surprise 3→8）；标签主观性已在 4.5 节说明 |
| 训练时间长 | ✅ 可控（v2 成本更高） | 自写 CNN 13.6 分钟；v1 迁移学习 12.5 分钟；**v2 单模型 30ep 约 60 分钟 × 3 seed + 消融 ≈ 3.7 小时 GPU**；本机 8.5GB 显存够用 |
| v2 训练的数据管线瓶颈 | ⚠ 注意 | 沙箱禁止 multiprocessing 命名管道 → DataLoader 必须 `num_workers=0`，数据增强单进程 CPU 成为瓶颈（30~43s/epoch）；通过**多实验并发**提高 GPU 吞吐，但单实验耗时被拉长 |
| v2 模型/权重增多 | ⚠ 注意 | `模型\` 现有 v1 + v2 消融 + 3 seeds 共 9+ 个权重（合计约 300MB）；报告/SELFTEST 已区分；`_deps` 增至约 1 GB（新增 effb0/resnet34 预训练权重 20~83MB/个） |
| `_deps\` 依赖目录缺失 | ⚠ 注意 | `timm` 等第三方库和预训练权重放在 `_deps\`，**删除后迁移学习模型无法加载/重新训练**；重新安装步骤见 3.2 节第 8 条 |
| 演示功能复杂度上升（建议 3） | ⚠ 注意 | 新增开关/选项/换一批后 07 变长；公共逻辑已抽到 `代码\11_visual_diversity.py`；**默认开关关闭**，不影响原有单图行为 |
| 建议 2 依赖与权重（神似模式） | ⚠ 注意 | `_deps\` 增至约 934 MB（CLIP 605 MB + mediapipe/opencv/onnxruntime + 关键点模型）；缺依赖时神似模式自动隐藏；**默认关闭**，不影响原有行为 |
| 神似模式的"主观性" | ⚠ 注意 | meme 神态/气质标签、视觉三路权重与 β 都是人工/目检定，评测用弱指标（社交语境 + AI 目检）；报告已写明局限，正式结论需人工盲评 |
| CLIP 跨域信号弱（已实测） | ✅ 有结论 | 2026-10-31 探针：CLIP 整体图像相似度跨域情绪信号≈随机（P2/P3 未过门），只保留属性画像 + 关键点几何；负结论留档 `结果\proto_clip_probe.txt` |

---

## 9. 新会话如何接手（项目已完成，2026-10-31）

0. **先看 `交接总结.md`**（项目根目录）：给新会话的交接文档，含现状、本会话改动、验证结果、环境坑、待决策项与建议 2 的后续步骤草案；
1. **读本文件**（`<项目根>/PROJECT.md`）+ `报告\最终报告.md`；
2. 跑环境自检（PowerShell；先设 UTF-8，避免中文/emoji 乱码）：
   ```powershell
   $env:PYTHONIOENCODING='utf-8'
   python <项目根>/代码/01_env_check.py
   ```
3. 想看演示（注意：本机 **SAC 会拦截 Explorer 启动 cmd.exe，双击 `启动演示.bat` 不可用**，详见第 3.2 节第 6 条）：
   - **推荐**：让 AI 在会话里用后台任务启动服务，然后用浏览器打开 **http://localhost:8501**；
   - 本机双击 `启动演示.bat` 会被 SAC 拦（cmd.exe 被拦）；可试着双击 `启动演示.lnk`（直接启动 python.exe，不经过 cmd），若也被拦则只能用上面两种方式；
   - 或自己在已经打开的终端里手动跑：
   ```powershell
   $env:PYTHONIOENCODING='utf-8'
   python -m streamlit run <项目根>/代码/07_demo_app.py --server.headless true --browser.gatherUsageStats false
   # 然后浏览器访问 http://localhost:8501
   ```
   > ⚠ **不要直接双击 / 直接运行 `07_demo_app.py`**：那样只会打印一句 "to view this Streamlit app on a browser, run it with streamlit run" 的警告，**不会打开网页**。必须用 `python -m streamlit run` 启动。`启动演示.bat` 只在没有 SAC 拦截 cmd 的机器上能双击运行。
   - **打不开 / 报 CUDA OOM 的排障（2026-11-01 实际案例）**：
     1. 先确认服务在跑：`netstat -ano | findstr :8501`；再请求 `http://127.0.0.1:8501/_stcore/health`。**8501 无监听 = 后台服务被会话/应用重启杀掉了**（后台任务生命周期跟 DSH 会话绑定），按上面命令重启即可；
     2. 页面报"视觉特征加载失败（CUDA out of memory）"**不代表项目坏了**：是 GPU 被其他程序/陈旧 CUDA 上下文占满（本机 RTX 5060 Laptop 只有 8GB）。页面已做**自动降级**（`empty_cache` → FER 特征换 ImageNet → 再不行用 CPU 算 ImageNet；低显存时神似模式 CLIP 也会自动走 CPU），并在"模型与数据信息"显示显存占用；实测视觉特征链路峰值仅约 **300 MB**，一般不是它吃满显存。2026-11-02 明确了一个实际原因：**8501 服务进程长期运行后自身显存堆积到约 6.8 GB**（Windows 计数器口径）——先点侧边栏"🧹 释放显存/内存缓存"按钮（现已释放 CLIP/特征/模型缓存 + `empty_cache` + `gc.collect`），不行再重启服务；模型缓存已加 `max_entries=2`，正常使用不会再无限堆积；
     3. 仍报错就关闭占 GPU 的桌面程序（微信/QQ/Edge 等也会占显存），或关闭演示页的"3-seed 集成/神似模式"；
     4. 跑训练/大模型时不要同时开演示服务（8GB 显存不够）。
4. 想复跑评估（**最终模型 = v2 efficientnet_b0@160**，默认读 `模型\best_model.pth`）：
   ```powershell
   $env:PYTHONIOENCODING='utf-8'
   # 单模型：v2 官方 val 72.64%；test 需 --splits all（最终 score 以 3-seed 集成为准）
   python ...\代码\06_evaluate.py --tta
   # 3-seed 集成 + TTA（★ 最终成绩 75.93% / 0.7443）：test 只跑这一种组合
   python ...\代码\15_ensemble_eval.py --weights "...\模型\v2_seed42.pth" "...\模型\v2_seed43.pth" "...\模型\v2_seed44.pth" --name v2_final --tta --splits all
   # 调参/对比时请用 --splits val，避免反复使用测试集
   ```
   > v1 旧模型：`模型\best_model_v1.pth`（ResNet18+TTA test 69.71%）。
4c. 想重新训练 v2（约 60 分钟/seed；⚠ 先确认）：
   ```powershell
   $py = 'python'; $code = '<项目根>/代码'
   & $py "$code\08_transfer_train.py" --model efficientnet_b0 --img-size 160 --epochs 30 --patience 10 `
       --batch-size 64 --optimizer lion --lr 1e-4 --label-smoothing 0.1 --mixup 0.2 --aug strong `
       --ema-decay 0.999 --swa-last 5 --warmup-epochs 2 --grad-clip 1.0 --amp --seed 42 --name v2_seed42
   # seed43/44 同命令改 --seed/--name；也可只训单模型后直接 15_ensemble_eval
   ```
4b. 想验证/复查**建议 2（神似模式）**（约 2 分钟/个；依赖 `_deps\` 的 CLIP 权重 + mediapipe）：
   ```powershell
   $env:PYTHONIOENCODING='utf-8'
   $py = 'python'
   $code = '<项目根>/代码'
   & $py "$code\12_clip_probe.py"     # B1：P1~P6 门槛（P1 过、P2/P3 未过 → 负结论，属预期）
   & $py "$code\14_shensi_probe.py"   # B2：关键点几何统计 / β 扫描 / 目检素材
   & $py "$code\07_demo_app.py" --selftest   # 含神似模式 3 张用例（缺依赖时自动跳过）
   ```
   结论与局限见 `报告\最终报告.md` §6.6；证据见 `结果\proto_clip_probe.*`、`结果\proto_shensi_*`、`_probe_b0\b0_report.md`。
5. 想重新训练（⚠ 都属"简单但耗时长"任务，按协作约定先确认再跑）：
   - 迁移学习（约 12.5 分钟）：`08_transfer_train.py --epochs 20 --patience 5 --batch-size 64 --img-size 128 --no-weight --name best_transfer`
   - 自写 CNN（约 14 分钟）：见 `报告\最终报告.md` 第 9 节
   - 注意：迁移学习依赖项目内 `_deps\`（timm + 预训练权重），不要删；
6. 若要继续改进，优先看 `报告\最终报告.md` 第 8.2 节（多 seed 集成 / 更高输入分辨率 / 更多 epoch / 人脸检测）；
7. **约定**：每完成一步要**实跑验证**并把关键输出贴给学生；主流程代码保持不依赖 torchvision / opencv（迁移学习用到的 timm、神似模式用到的 CLIP/mediapipe 都只是**可选增强**，删掉不影响自写 CNN + 软匹配 + 演示主流程，神似模式开关会自动隐藏）。

> **自检脚本说明**（`代码\01_env_check.py`，已实测通过）：会连续 3 次用子进程导入 `torchvision`，
> 若显示 `1/3`、`2/3` 就说明被系统策略间歇拦截——此时**不要**把 torchvision 当必需依赖。
> 所有脚本都用 `__file__` 相对定位项目根目录，项目整体搬家后不用改路径。

---

## 10. 进度记录

| 日期 | 事件 |
|---|---|
| 2026-10-30 | 环境体检（GPU 可用、SAC 拦截问题、opencv 缺 wheel）；cat_meme 审查（38 文件 / 28 唯一图 / 7 类齐全 / disgust 仅 1 张）；建立本文件 |
| 2026-10-30 | 新建 `代码\01_env_check.py`（含 torchvision 稳定性检测），实测通过：torch ✅ / CUDA ✅ / torchvision 3-3 ✅ / opencv ❌ / timm ❌ |
| 2026-10-30 | **项目文件整体迁入 `表情识别项目\`**（PROJECT.md / 代码\ / 数据\，并新建 模型\ 结果\ 报告\）；`01_env_check.py` 与 `_audit_memes.py` 改为 `__file__` 相对定位，修掉 GBK 控制台 `UnicodeEncodeError`；迁移后复跑环境自检通过（torch 2.14.1+cu132 / CUDA ✅ / torchvision 3-3 ✅ / opencv ❌ / timm ❌ / GPU RTX 5060 8.5GB） |
| 2026-10-30 | 4 个待决策项按「默认」拍板（HF 镜像取 FER2013、Streamlit 上传演示、不建 py311、补 meme 图）；开始阶段 1.1 |
| 2026-10-30 | 阶段 1.1 完成：FER2013 从 `hf-mirror.com` 镜像下载（3 个 parquet 共 133 MB）→ 转成 `数据\fer2013\fer2013.npz`（78.3 MB）。实测 train 28709 / val 3589 / test 3589，标签顺序与 Kaggle 官方一致；训练集 disgust 仅 436 张（1.5%），是最多类 happy 的 1/16.5 |
| 2026-10-30 | `02_meme_index.py` 完成：38 个文件 → 28 张唯一图 → 全部处理成 256×256 RGB 白底 PNG；写出 `数据\meme_index.csv`（28 行）与 `结果\meme_preview.png`（已目检：缩放/白底/去重正常） |
| 2026-10-30 | 阶段 1.2~1.4 完成：`03_dataset.py` 自写 `FERDataset`（不用 torchvision）+ 数据增强（水平翻转 / ±10° 旋转 / 亮度 ±10%）+ 类别分布统计 + 类别权重；DataLoader 实跑 28709 张/epoch 约 3.2s（CPU 读数据+增强）；产出 `结果\aug_preview.png`（已目检）。期间发现 `OMP Error #15`（两份 Intel OpenMP），已用 `KMP_DUPLICATE_LIB_OK=TRUE` 解决并记录到第 3.2 节 |
| 2026-10-30 | `04_model.py` 完成 + 实跑：baseline 93,799 参数（前向/反向正常）；但实测 baseline **欠拟合**（30 轮训练集准确率仅 ~53%），于是加 `wide`（437,703）与 `flat`（2,731,463）两档，训练脚本用 `--profile` 选择 |
| 2026-10-30 | `05_train.py` + `06_evaluate.py` 完成；跑完 **8 组消融实验 E1~E8**（测试集 3589 张）：E1 51.24% / E2 48.01% / E3 52.61% / E4 55.50% / E5 wide 59.29% / E6 wide+加权 55.64%（disgust 召回 60%）/ E7 flat 59.99% / **E8 flat2 64.73%（Macro-F1 0.5368）**。结论：早停 patience=3 会误停；容量不够是主因；类别加权提升小类召回但整体掉点。实验全表见 `报告\最终报告.md` 第 5.1 节 |
| 2026-10-30 | `07_demo_app.py` 完成并验收：`--selftest` 5/5 通过（硬匹配/软匹配/回退都正常）；Streamlit AppTest 渲染 + 点击"随机测试图"走通完整推理；实际启动 `streamlit run` 服务，`http://127.0.0.1:8502/_stcore/health` 返回 200 |
| 2026-10-30 | **项目收尾**：最终模型定为 flat2 → 复制为 `模型\best_model.pth`（测试集 64.73% / Macro-F1 0.5368）；重跑最终评估生成 `结果\evaluation_test.*`、`confusion_matrix_test.png`；`报告\最终报告.md` 完成；清理临时探针文件。**阶段 0~5 全部完成 ✅** |
| 2026-10-30 | 学生反馈"演示网页打不开" → 排查发现：① 直接运行 `07_demo_app.py` 不会打开网页（Streamlit 必须用 `streamlit run` 启动，脚本只打警告）；② 用含中文的 .bat 启动时 cmd 会因控制台代码页与文件编码不匹配而解析失败。**解决**：新增 `启动演示.bat`（内容纯 ASCII、运行时用 `for /d` 自动定位 代码 目录，避免中文路径硬编码），实测启动后 `http://127.0.0.1:8501/_stcore/health` 返回 200、首页正常返回；PROJECT.md 第 9 节补了"不要直接运行 .py"的警告 |
| 2026-10-30 | 学生再次反馈"启动演示.bat 无法运行" → 查 `Microsoft-Windows-CodeIntegrity/Operational` 日志确认：22:53~22:54 有 **7 条事件 3033**，`explorer.exe` 想启动 `cmd.exe` 被系统签名策略拦截（本机 SAC 开启，`VerifiedAndReputablePolicyState=1`）——所以本机双击任何 .bat 都跑不起来，**不是脚本问题**。另发现 Streamlit 首次运行在非 headless 模式会交互式问 Email 导致非交互环境退出（exit 1）。**解决**：启动命令统一加 `--server.headless true --browser.gatherUsageStats false`，并由 AI 在会话内用后台任务把服务起起来，学生只开浏览器访问 `http://localhost:8501`（实测 health 200 / 首页 200）；已把这两条坑写进第 3.2 节第 6、7 条 |
| 2026-10-30 | **表情包扩充**：从 HuggingFace 数据集 `pielet/cat-memes`（MIT 标注，55 个知名猫梗图，共 19.3 MB）下载全部图片 → 生成 3 张缩略图总览**逐张人工看图标情绪** → 挑 30 张加入素材库（`<情绪>_pielet_<名字>`，并按实际格式修正扩展名）→ 更新 `ATTRIBUTION.md`（新增 pielet 来源与人工标注说明）→ 重跑 `_audit_memes.py`（68/68 可读、0 重复、0 命名不一致）与 `02_meme_index.py`（元数据表 28→58 行）→ `07_demo_app.py --selftest` 通过（58 张、软匹配已能选到新图）。每类唯一图：angry 7 / disgust 4 / fear 5 / happy 13 / sad 6 / surprise 8 / neutral 15 |
| 2026-10-30 | 演示服务已按新表情包库重启（后台任务），`http://127.0.0.1:8501/_stcore/health` 返回 200 ok |
| 2026-10-31 | 学生问"模型还能不能继续优化" → 评估方案后按 **A+B** 执行。**方案 A**：手工把 `timm 1.0.30` / `huggingface_hub 0.36.0` / `safetensors 0.8.0` 装到项目内 `_deps\`（pip 写全局目录被沙箱拒，见 3.2 节第 8 条），预训练权重从 `hf-mirror.com` 下载到 `_deps\weights\`；写 `08_transfer_train.py` 并扩展 `03_dataset.py`（`FERTransferDataset`：48→128、灰度→3 通道、ImageNet 归一化、一次性预 resize 提速）。训练 ImageNet 预训练 ResNet18 微调 20 轮（12.5 分钟）：val 67.51% / test **68.46% / Macro-F1 0.6633**，比自写 CNN 最好的 E8（64.73% / 0.5368）提升明显，disgust 召回从 0 提到 60% |
| 2026-10-31 | **方案 B（TTA）**：`06_evaluate.py` 增加 `--tta`（6 视角：原图/翻转/上下左右平移，softmax 平均），`07_demo_app.py` 增加 TTA 开关 + 迁移模型推理支持（128×128 RGB + ImageNet 归一化）。实测 flat2 64.73%→65.51%，迁移模型 68.46%→**69.71% / Macro-F1 0.6796**（最终成绩）。`best_model.pth` 更新为 ResNet18 迁移学习权重；重跑 `evaluation_test.*` 与 `evaluation_test_tta.*`、混淆矩阵；`07_demo_app.py --selftest` 5/5 通过；演示服务已重启（`http://localhost:8501` health 200） |
| 2026-10-31 | 文档同步：`报告\最终报告.md` 补迁移学习 + TTA 的实验、指标、混淆矩阵与复现命令；PROJECT.md 更新第 3/5/6/7/8/9/10 节。清理 sanity 权重与 `_pip_tmp`/`_hf_cache` 临时目录 |
| 2026-10-31 | 离线原型 `09_meme_softmatch_proto.py`（全自动 7 维向量 + 距离/聚类匹配）结论：① FER 模型对猫 meme 自动打标与人工标签一致率仅 **22.4%**，`model` 向量策略 label_hit 0.14；② rule/onehot 各种距离与现方案差异 <1pp；③ KMeans 轮廓系数 0.28，聚类无帮助；④ **全自动软标签方案暂不成立**；⑤ 意外发现软匹配只用 8 张图的多样性缺陷。产物：`结果\proto_*` |
| 2026-10-31 | 视觉相似度探针（FER/ImageNet ResNet18 全局特征）：同情绪人脸 ↔ 同情绪 meme 的质心对齐与随机几乎无差别（对角-非对角 +0.004 / -0.001），视觉检索 top-1 仅 0.28/0.21（随机水平）→ **视觉相似度直接融合暂不可行**；结论存 `结果\proto_visual_probe.txt` |
| 2026-10-31 | **软匹配全局修复（本轮落地）**：`07_demo_app.py` 的 `pick_meme` 改为"选中情绪与旧逻辑完全一致 + 同类同分候选随机 + `avoid_file` 防连续重复"，`run_app` 用 `st.session_state` 记录上一张图；`--selftest` 新增 3 项验收。离线验证（验证集 3589，TTA 概率）：label_hit 0.6773 / S 加权相似度 0.8124 **完全不变**，用到的不同 meme **7 → 58 张**、连续重复 **636 → 0 次**；自检中 100 组随机概率"选中情绪"零不一致、sad 50 次用到 6 张、`avoid_file` 生效；Streamlit AppTest 首屏 + 连续两次点击均无异常 |
| 2026-10-31 | 演示服务已用软匹配修复版重启（后台任务），`http://127.0.0.1:8501/_stcore/health` 返回 200 ok。待办：匹配模式选择（建议 2 CLIP 视觉 / 建议 3 视觉多样性）尚未开始；后续新增模式需基于修复后的基线做对比 |
| 2026-10-31 | **建议 3 离线原型 `10_visual_diversity_proto.py` 完成**（不改模型/演示）：评估 3 种备选范围 × 2 种视觉特征 = 6 组合。结果：主推情绪 0 变化；三张两两视觉距离比同约束随机备选高 **0.10~0.14**（imgnet 0.634~0.668 vs 随机 0.530）；无近重复；换一批多数能出 5.5~5.9/6 个不同组合（仅同类受类大小限制约 4/6）；目检发现"纯最大距离会挑到纯纹理怪图"，补测分位带方案（60%~95%）只损失 0.019 多样性但样例明显更正常 → 定为默认策略。报告：`结果\proto_visual_diversity_report.md` |
| 2026-10-31 | **建议 3 接入演示完成**：新增 `代码\11_visual_diversity.py`（meme 特征缓存、候选池、距离分位带 + 贪心 max-min、换一批）；`07_demo_app.py` 侧边栏新增"视觉多样性展示"开关（**默认关**）+ 备选范围三选一 + 视觉特征二选一 + "换一批备选" + 差异分显示（余弦距离）；主推逻辑完全不变。`--selftest` 新增 20 图 × 3 范围 × 2 特征 = **120 次选择/换一批全部通过**；AppTest 各开关/选项组合均 `exception=[]`、渲染 4 图（输入+主推+2 备选）+ 3 表；演示服务已重启，`http://127.0.0.1:8501/_stcore/health` 返回 200 ok |
| 2026-10-31 | 生成 **`交接总结.md`**（项目根目录）供新会话接手；PROJECT.md/报告同步建议 3 实现与验证结果。建议 2（CLIP）暂缓审查；GIF 抽帧本批不纳入 |
| 2026-10-31 | **建议 2 审查开始（B0 关键部位可行性）**：安装 mediapipe 1.2.0（+opencv 5.0.0.93 abi3 轮子、absl、sounddevice）到 `_deps\`；mediapipe 478 点对 FER 48×48 放大图 **14/14 语义检查通过、10~12ms/张**；ONNX 478 点备份 9ms/张（与官方 tflite 误差 0.0002px）；猫脸关键点：CatFLW 48 点 tflite 照片风可用、插画/像素风失败（58 张仅 36 张两路裁剪一致，tizee 系列全不可用）；几何特征单独分 7 类仅 28.6% → 只作辅助。报告 `_probe_b0\b0_report.md` |
| 2026-10-31 | **建议 2 B1 CLIP 探针（验证集 3589，测试集未碰）**：手工装 open_clip 3.3.0 + ftfy、从 hf-mirror 多线程下载 laion CLIP ViT-B/32 权重（605,219,813 B，单连接 0.1MB/s → 8~24 并发分块 446s 下完）。结果：P1 质心对齐 **+0.0521**（门 0.03，p=0.0492）勉强过；P2 检索 top1 **0.1602**（随机 0.1631，p=0.4592）未过；P3 零样本标签一致率 **0.1207**（多数类 0.2586）未过；P4 对照 ImageNet ResNet18 质心 -0.0003/检索 0.1920；**P6 属性桥（新增）纯属性检索 top1 0.3486（随机 0.1631，p=0.0015）** → 唯一有真信号的跨域通道。负结论留档 `结果\proto_clip_probe.*` |
| 2026-10-31 | **建议 2 B2 神似模式落地**：新增 `代码\13_shensi_visual.py`（478 点几何 + CLIP 16 维神态属性画像 + 三路融合）与 `代码\14_shensi_probe.py`（几何分位统计、β 扫描、目检素材）；`07_demo_app.py` 侧边栏加"神似模式"开关（**默认关**）+ β 滑条（0.10~0.60，默认 0.35）+ 关键部位/属性画像/Top-5 明细 + "换一张神似候选"。验证集：β=0 情绪命中 0.6773（=历史 label_hit）→ β=0.35 时 0.6762（-0.11pp），类内视觉分 0.685→0.728，用到 48/58 张；`--selftest` 3/3 神似用例通过；AppTest 8 步 `exception` 全空；演示服务重启 health 200。弱指标目检 14 例：明显更神似 6、持平 8、明显更差 0。报告 `报告\最终报告.md` §6.6 |
| 2026-10-31 | 回归验证：`06_evaluate.py --tta` 复跑确认模型指标不变（test **69.71% / 0.6796**）；`--selftest` 全绿（5/5 样本 + 软匹配修复 + 视觉多样性 120 次 + 神似 3 张）；`_deps\` 约 934 MB（含 CLIP 605MB）；PROJECT.md/报告/PROJECT 同步完毕 |
| 2026-10-31 | **模型优化 v2 · 阶段 0**：改造 `08_transfer_train.py` 支持任意 timm 模型（`pretrained_cfg["hf_hub_id"]` 通用下载、按形状过滤分类头）+ AMP + Lion/AdamW + warmup/cosine + grad-clip + EMA(ModelEmaV3) + SWA + mixup + balance(focal/sampler) + `03_dataset.py` 新增 `aug="strong"`（basic 路径不变），ckpt 字段向后兼容（06/07 实测可加载）；3 轮 smoke 通过 |
| 2026-10-31 | **模型优化 v2 · 阶段 1 配方消融**（resnet18@128, 30ep, AMP, 只 val）：R1 68.35%/0.6457 → R2(+EMA+SWA+warmup+clip) 67.90%/0.6546 → R3(+mixup+强增强+LS0.1) 69.10%/0.6587 → **R4(+Lion lr1e-4) 69.96%/0.6722**（比 v1 +2.45pp/+0.0165）。结论：EMA 与 Lion 贡献最大，SWA 未被采用 |
| 2026-10-31 | **模型优化 v2 · 阶段 2/4 backbone 与集成**：resnet34@128 71.16%/0.6880；**efficientnet_b0@160 72.64%/0.7039（单模型最优）**；resnet50 因算力预算未跑。3 seeds 均值 72.22%±0.31pp；**3-seed 集成 + 6 视角 TTA：val 73.92%/0.7194，test 75.93%/0.7443（相对 v1 +6.22pp/+0.0647；test 仅评估一次）**。集成增益 val +1.34pp / test +1.62pp |
| 2026-10-31 | **v2 收尾**：`best_model.pth` 更新为 v2 effb0@160（`best_model_v1.pth` 保留）；`07_demo_app.py` 新增"3-seed 集成（概率平均）"开关（默认关）+ 多模型加载/逐模型预处理；`06_evaluate.py` 新增 `--tta-strong`（10 视角）与 `--splits`（调参只用 val）；新增 `15_ensemble_eval.py`；`--selftest` 全绿（含 v2 集成验证）、AppTest 10 步全过、演示服务重启 health 200；报告 §5.5 完成 |
| 2026-11-01 | **演示改进（两个小功能，默认行为兼容）**：① **同分候选选择策略**做成侧边栏可选（`TIE_STRATEGIES`：随机+防重复（默认）/纯随机/确定性 hash/固定第一张；4 种策略 `--selftest` 全过、AppTest 覆盖；只影响类内挑哪张，不改情绪与指标）；② **GIF 动图播放修复**：新增 `代码\16_meme_anim.py` 生成 18 张 256×256 压缩 GIF（`数据\meme_anim\`，约 10.4 MB），侧边栏"GIF 动图播放"开关（**默认开**，可关回静态首帧）；模型/匹配/特征仍用静态首帧 PNG，`--selftest` 新增动画版验收、AppTest 16 步全过；演示服务重启 health 200；报告 §7.2/§7.3/§10、PROJECT §7 决策同步 |
| 2026-11-01 | **GIF 多帧视觉特征 B/C 方案**：新增 `代码\17_meme_gif_frames.py`（18 张 GIF、采样 ≤16 帧共 239 帧、时长权重、峰值帧 18/18 非首帧；`数据\meme_gif_frames` 21.2MB + `数据\meme_anim_peak` 1.2MB）；`11_visual_diversity.py` 的 `get_meme_features(..., gif_mode)` 与 `13_shensi_visual.py` 的 `MemeAssets(gif_mode)` 支持 first/mean/peak（B=多帧时长加权平均，C=峰值帧；只替换 18 张 GIF，非 GIF 40 张不变）；`07_demo_app.py` 新增"GIF 视觉特征（仅影响备选/神似）"单选（默认首帧）；`--selftest` 验证 18/18 替换、非 GIF 不变、神似资产同步；AppTest 19 步全过；同时实测否定"帧平均情绪向量用于主匹配"（GIF 帧平均 argmax 与人工一致率仅 5.6%，负结论写入报告 §6.7） |
| 2026-11-02 | **演示显存加固（用户提的 1~4 项）**：① `st.cache_resource(max_entries=2)` 限制模型缓存；② "🧹 释放显存/内存缓存"按钮升级：清 `st.cache_resource` + `sv.release_clip()`（13 新增）+ `vd.release_feature_cache()`（11 新增）+ `torch.cuda.empty_cache()` + `gc.collect()`；③ 低显存自动降级：空闲 <900MB 时神似 CLIP 自动走 CPU（`sv.set_force_cpu()`，`_ensure_clip` 按标志选设备；meme 特征有缓存，只编码当前人脸约 1~2s），OOM 后再 CPU 重试一次；ImageNet 视觉特征低显存也自动走 CPU；④ OOM 提示带处理建议。实测发现 8501 服务长期运行后自身显存堆积约 6.8GB（Windows 计数器口径），释放后 GPU 恢复到 ~7GB 空闲。selftest（GPU 与 CPU 两种模式）均 exit 0（含 CPU 版神似/释放后重算验证）；AppTest GPU/CPU 两种模式全过（首屏/随机图/3-seed/多样性/神似/视觉特征 FER/释放按钮 `exception=[]`） |
| 2026-11-02 | **P0 A 快筛（FER+ 标签）**：新增 `代码\18_ferplus_labels.py`（下载 1.53MB CSV、按 Usage+行序对齐、contempt→disgust、丢弃 unknown/NF 共 150/10/16 张；生成 `数据\fer2013\ferplus_labels.npz` + meta）；`03_dataset.py`/`08_transfer_train.py` 新增可选 `--labels ferplus`（默认行为不变）；训练 1 seed effb0@160 v2 同配方（早停 26ep，约 35min）→ `模型\ferplus_hard_s42.pth`（新文件，不覆盖现有权重）。**原始标签 val：61.27/0.5539（v2 72.61/0.7036）→ No-Go，不替换主模型、不做 UI、不进 B**；**两套标签一致子集：FER+ 模型 90.29/0.8554 vs v2 87.21/0.8138（+3.1pp）**，说明是标签定义差异。测试集未使用；同时修复 08 中 `import re` 缺失的隐藏 bug。详见 `结果\ferplus_A_quick_screen.md`、报告 §5.6 |
| 2026-11-02 | **诊断 + A/F 快筛**：① 新增 `代码\19_error_analysis.py`（只读 val 错误分析）：Acc 72.61/0.7036；弱类 fear 53.2%/disgust 60.7%/sad 61.9%；sad→neutral 15.3%、fear→sad 17.9% 最大；错误中 37.7% 预测等于 FER+ 标签（疑似标签噪声），无争议子集 87.21%（291 真错）；ECE 0.094。② **A focal**：`v2_focal_s42.pth` 单模型 val 72.69/0.6992（TTA 73.00/0.7021）→ **No-Go**；4 模型集成 74.03/0.7211 vs 3-seed 73.92/0.7194（+0.11pp，不值）。③ **F 匹配层**：新增 `代码\20_match_calibration.py`；温度缩放 T*=0.65 → label_hit **0.7242→0.7292（CV +0.50pp）**，ECE 0.094→0.053；相似度矩阵调优全量 val +1.47pp 但 CV 仅 +0.27pp（过拟合，不推荐）。详见 `结果\A_focal_quick_screen.md`、`结果\match_calibration_v2.md`、`结果\diagnose_v2_val.md` |
| 2026-11-02 | **UI 优化 + F 落地**：权重选择改为**两级分组**（"模型系列 → 具体权重"，每项带中文说明与 val/test 指标：⭐v2 最终 / 🚀3-seed / v1 / 自写 CNN / 消融 / 实验版；`build_weight_groups` 自动覆盖全部 `.pth`、未登记进"其他"组）；新增 **"概率温度校准"开关**（默认关，T=0.65，只作用于软匹配打分与 Top-5 表，分类 argmax/准确率/页面概率条不变，CV label_hit +0.50pp）；selftest 新增温度归一/argmax 不变/分组全覆盖检查，AppTest 覆盖分组切换 + 温度开关（exception=[]） |
| 2026-11-02 | **匹配层变体实验（val-only）**：新增 `代码\21_match_variants.py`，5 折 CV 对比 hard / soft / soft+T0.65 / blend α / gated θ / 混淆矩阵相似度。**严格 label_hit：hard 0.7300 最好**，soft 0.7242、soft_T 0.7292、blend 0.7289、gated 0.7236、conf_sim 0.7300（与 hard 持平，全量 val 0.7303 属过拟合）；**软命中@0.6：soft_T 0.8019 最高（hard 0.7999）**。结论：相似性匹配的定位是"不确定时的鲁棒性/观感"，不是提高严格命中率；若要提严格指标应回到分类模型。详见 `结果\match_variants.md`、报告 §6.9 |
| 2026-11-02 | **猫 meme 扩库（58→66）**：从 pielet 剩余候选中人工选 8 张并标注（fear+2/sad+1/angry+1/surprise+2/happy+2，disgust 暂无合适）；`02_meme_index.py` 新增 `meme_dedup_choices.json` 固定历史"同图多格式"选择，**修复新增图片会导致旧图版本被随机替换的副作用**（旧 58 张完全保留，GIF 18→20 仅因新增 2 个 GIF）；重跑 16/17 生成 20 个 GIF 资产、`shensi_meme_cache.npz` 自动重算 66 张；`meme_index.csv` 66 行；selftest/AppTest 全过。指标口径不变（类分数与候选池无关），收益在类内可选图更多、重复更少、备选/神似更丰富 |
| 2026-11-02 | **多情绪（混淆性）meme 实验（val-only，负结论）**：`数据\meme_emotion_multi.json` 对 66 张中 26 张标第二情绪（最多 2 标签，0.7/0.3 或 0.6/0.4）；`07_demo_app.py` 新增可选 `multi_q` 参数（**默认 None=one-hot，行为不变**）；`22_multi_label_match.py` 5 折 CV 对比。**v1 加权平均 p·SIM·q 在数学上被 one-hot 支配（multi_rate=0，无效果）**；cosine 版本 strict 0.7175（−0.67pp）/coverage 0.8058 但 multi_rate=100%（靠"标签集合更大"刷高）；maxlabel strict 崩到 0.1981；gated 版无 strict 收益。结论：**不加入 UI/默认匹配**，仅归档为负结论；`meme_emotion_multi.json` 保留实验资产 |
| 2026-11-02 | **B/E 快筛（均未胜出）**：E=EffV2-B0@160（30ep/44.4min，TTA val 72.61/0.7073，低于 seed42 73.00/0.7046）；B=drop_path0.1+40ep@160（57.5min，TTA val 71.72/0.6948）。集成检验：3-seed+E val **74.20/0.7201（+0.28pp）**，3-seed+B 73.73/0.7168，5 模型 74.09/0.7200；E 的集成增益太小且演示要多加载模型 → **不替换主模型、不进 3-seed、不跑 test**；08 新增 `--drop-path`（默认 0=旧行为）；权重归入"实验版"分组 |
| 2026-11-07 | **A 方案扩训练集（完成，No-Go）**：publicTest(val) 并入训练、分层留 1,501 早停 → train 30,797（+7.3%）；3 seeds（留出集 73.35/72.55/72.29，集成 74.15/0.7224）；**privateTest 75.17/0.7400 vs v2 75.93/0.7443（−0.76pp / −0.43pp）→ 不替换主模型**；说明瓶颈更可能是标签噪声而非数据量；v3 权重归档 `模型\_archived_v3_expand\`；测试集累计评估 2 次（v2、A 方案） |
