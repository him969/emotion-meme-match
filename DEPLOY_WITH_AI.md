# 用 AI 助手部署本项目（提示词模板）

> 适用对象：拿到本项目但**不熟悉 Python / 深度学习环境配置**的人。
> 用法：复制下面「提示词正文」整段发给任意支持终端执行的 AI 编程助手（Cursor、Claude Code、Codex、通义灵码、DSH 等），把 `<仓库地址>` 和 `<数据资产来源>` 两处补上即可。

---

## 提示词正文（复制以下全部内容）

```text
你是一名擅长 Python / 深度学习环境部署的工程师。请在本机把下面这个开源项目部署到「能启动 Streamlit 演示页面、并完成一次推理」的状态。
请严格按步骤执行：每一步先验证再继续，遇到错误先贴原始报错、再尝试修复；不要伪造运行结果，不要跳过验证。

========== 项目信息 ==========
- 仓库地址：https://github.com/him969/emotion-meme-match
- 项目：基于 CNN 的人脸表情识别（FER2013 七类：angry/disgust/fear/happy/sad/surprise/neutral）+ 猫 meme 匹配，Streamlit 演示
- 演示入口：代码/07_demo_app.py
- 自检命令：python 代码/07_demo_app.py --selftest
- 开发环境：Windows + Python 3.14 + torch（CUDA）；理论上 Python 3.10+ 均可
- 核心依赖：torch、numpy、pandas、pillow、scikit-learn、matplotlib、streamlit、tqdm、timm（v2 权重是 EfficientNet-B0，必须能 import timm）、pyarrow（读 parquet 需要）
- 不依赖 torchvision / opencv（仅「神似模式」等可选功能用 opencv/mediapipe/CLIP，属于 _deps/ 可选项）
- 最终模型指标：测试集 Accuracy 75.93% / Macro-F1 0.7443（v2 3-seed 集成 + TTA）

========== 必须准备的资产（不在 git 仓库里） ==========
1) 模型权重：GitHub Release（v1.0.0）附件 `model-weights-v1.0.0.zip`（约 546 MB），解压后把 .pth 放到仓库根目录的 `模型/` 下。
   必需文件至少包括：`模型/best_model.pth`、`模型/v2_seed42.pth`、`模型/v2_seed43.pth`、`模型/v2_seed44.pth`
2) 数据目录 `数据/`：
   - FER2013：可运行 `代码/00_download_fer2013.py` 自动从 hf-mirror 下载并生成 `数据/fer2013/fer2013.npz`
   - 猫 meme 库：`数据/meme_ready/<7 个情绪>/ 的 256x256 PNG` + `数据/meme_index.csv`（演示程序靠它检索）。
     这两项仓库未包含，需从作者提供的资产包获取（询问用户存放在哪里），或让用户向仓库作者索取。
3) （可选）`_deps/`：CLIP / mediapipe / 关键点模型等，约 1.1 GB，只影响「神似模式」；缺失时页面会自动隐藏该开关，不影响基础演示。若用户提供了就解压到仓库根目录。

========== 前置检查（先做，不要急着安装） ==========
1. 告知我当前系统（Windows/macOS/Linux）、Python 版本（python --version）、是否有 NVIDIA GPU（nvidia-smi）、磁盘剩余空间
2. 检查项目是否已 clone；磁盘剩余空间建议 ≥ 5 GB
3. 检查 `数据/`、`模型/`、`_deps/` 是否已存在；缺什么就明确告诉我需要哪些资产、从哪里获取，不要自行编造或下载来路不明的数据
4. 先给我一份执行计划（环境方案 + 步骤清单），再开始执行

========== 部署步骤（按顺序执行，每步验证） ==========
A. 获取代码
   git clone https://github.com/him969/emotion-meme-match
   cd emotion-meme-match
B. 创建隔离环境
   Windows: python -m venv .venv && .venv\Scripts\activate
   macOS/Linux: python3 -m venv .venv && source .venv/bin/activate
   若 Python 版本过新导致依赖装不上，优先改用 Python 3.12 重新建环境，不要用 conda 混装
C. 安装依赖
   1) torch：有 NVIDIA GPU → 按 https://pytorch.org 选择与驱动匹配的 CUDA 版本命令；无 GPU → 安装 CPU 版
   2) pip install numpy pandas pillow scikit-learn matplotlib streamlit tqdm timm pyarrow
D. 准备资产
   1) 模型权重：下载 Release 附件 `model-weights-v1.0.0.zip` → 解压到 `模型/`；确认上述 4 个 .pth 存在
   2) FER2013：python 代码/00_download_fer2013.py（Windows 先 $env:PYTHONIOENCODING='utf-8'）
   3) 猫 meme 库：按用户提供的资产包/下载地址，解压到 `数据/`，确认 `数据/meme_ready/` 有 7 个情绪子目录、`数据/meme_index.csv` 存在
   4) （可选）_deps/：用户提供则解压到项目根目录，否则跳过
E. 冒烟测试（必须全部通过才继续）
   python 代码/01_env_check.py
   python 代码/07_demo_app.py --selftest
F. 启动演示
   python -m streamlit run 代码/07_demo_app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
   验证：访问 http://localhost:8501/_stcore/health 返回 ok；
   打开 http://localhost:8501 ，点「随机来一张测试图」，确认出现 7 类情绪概率条 + 1 张猫 meme 图片
G. 收尾
   给出：完整复现命令、环境说明、遇到问题的排查记录、后台长期运行/改端口的方法

========== 验收标准 ==========
- [ ] selftest 全部通过（无报错）
- [ ] http://localhost:8501/_stcore/health 返回 200/ok
- [ ] 页面完成一次推理：上传一张已裁好的人脸图或点「随机来一张测试图」，能看到 7 类概率与匹配的 meme
- [ ] 未修改项目代码逻辑（只允许为兼容性做最小改动，并说明）
- [ ] 未把 数据/、模型/、_deps/ 提交进 git
- [ ] 明确告知用户：因缺 _deps 而不可用的功能（神似模式）或因缺数据而无法启动的部分

========== 已知坑（按此优先排查） ==========
1. Windows 控制台 GBK 编码：先设 $env:PYTHONIOENCODING='utf-8'
2. OMP Error #15（torch + matplotlib）：代码已设 KMP_DUPLICATE_LIB_OK=TRUE；若仍出现，在启动前设该环境变量
3. 加载 v2 权重报 "No module named 'timm'"：pip install timm；不要用 torchvision 替代
4. 项目路径含中文/空格：尽量放到纯英文路径（mediapipe 等库对中文路径敏感）
5. Streamlit 首次运行卡在 Email 输入：必须带 --server.headless true
6. Windows 双击 .bat 没反应（Smart App Control 拦截 cmd）：改用命令行启动
7. 端口被占用：换 --server.port 8502
8. 显存不足（CUDA out of memory）：关闭「3-seed 集成」「神似模式」，或直接用 CPU 重新启动
9. 缺数据资产时 demo 无法启动：回到用户确认，不要自行用随机图片冒充 meme 数据（除非用户明确同意只做 UI 演示）
10. 若你的执行环境是受限沙箱（无网络 / 无系统写权限）：不要强行安装；改为生成完整命令清单交给用户本机执行

========== 交互与约束 ==========
- 先汇报计划再动手；不要一上来批量 pip install
- 每一步执行后给出验证结果；失败时提供原始报错 + 尝试过的方案
- 不要伪造截图/运行结果；不确定时询问用户
- 大文件（数据、模型权重）只放本地，禁止 git add
- 最终交付《部署报告》：环境信息、执行的命令、验证结果、遗留问题、后续使用方式
```

---

## 使用说明（给部署者）

1. 把上面「提示词正文」整段复制；
2. 把 `<仓库地址>`、`<数据资产来源>` 换成实际信息（仓库地址已填好；数据资产填 Release 附件链接、网盘链接或“向作者索取”）；
3. 发给支持终端操作的 AI 助手，并把终端/编辑器权限交给它；
4. 它会在本机建环境、装依赖、准备数据、跑自检；遇到需要授权/下载时会询问你。

## 给仓库维护者的提醒

- 若希望别人**开箱即跑**，最好在 Release 里同时提供数据资产包（如 `data-assets-v1.0.0.zip`，含 `数据/meme_ready/`、`数据/meme_index.csv` 及必要缓存），并在 README 里写明下载方式；
- 猫 meme 图片版权归原作者，公开分发前请确认 `数据/cat_meme_extracted/cat_meme/ATTRIBUTION.md` 中的许可与署名要求；
- 权重已作为 Release 附件提供，部署者无需重新训练。
