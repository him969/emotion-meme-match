# 多情绪 meme 匹配实验 v2（val 3589 张，6 视角 TTA；val-only）

- 多标签 meme：**26** 张；歧义照片：pmax<0.5 **691** 张、pmax<0.6 **1114** 张

## 全量 val（θ 为 gated 门限；真实增益看 CV）

| 方案 | strict | coverage | multi_rate |
|---|---|---|---|
| baseline | 0.7242 | 0.7428 | 0.2705 |
| weighted | 0.7242 | 0.7242 | 0.0000 |
| cosine | 0.7175 | 0.8058 | 1.0000 |
| maxlabel | 0.1981 | 0.7763 | 1.0000 |
| gated_cosine_theta=0.50 | 0.7205 | 0.7668 | 0.1925 |
| gated_cosine_theta=0.55 | 0.7180 | 0.7740 | 0.2441 |
| gated_cosine_theta=0.60 | 0.7175 | 0.7782 | 0.3104 |
| gated_cosine_theta=0.65 | 0.7175 | 0.7835 | 0.5038 |
| gated_cosine_theta=0.70 | 0.7175 | 0.7896 | 0.6523 |
| gated_cosine_theta=0.75 | 0.7175 | 0.7991 | 0.8172 |
| gated_cosine_theta=0.80 | 0.7175 | 0.8025 | 0.9434 |
| gated_cosine_theta=0.85 | 0.7175 | 0.8050 | 0.9900 |
| gated_cosine_theta=0.90 | 0.7175 | 0.8055 | 0.9916 |
| gated_maxlabel_theta=0.50 | 0.6857 | 0.7473 | 0.1925 |
| gated_maxlabel_theta=0.55 | 0.6662 | 0.7517 | 0.2441 |
| gated_maxlabel_theta=0.60 | 0.6364 | 0.7565 | 0.3104 |
| gated_maxlabel_theta=0.65 | 0.4865 | 0.7621 | 0.5038 |
| gated_maxlabel_theta=0.70 | 0.3814 | 0.7651 | 0.6523 |
| gated_maxlabel_theta=0.75 | 0.2990 | 0.7726 | 0.8172 |
| gated_maxlabel_theta=0.80 | 0.2424 | 0.7751 | 0.9434 |
| gated_maxlabel_theta=0.85 | 0.2062 | 0.7754 | 0.9900 |
| gated_maxlabel_theta=0.90 | 0.2056 | 0.7760 | 0.9916 |

## 歧义子集（pmax<0.5）

| 方案 | strict | coverage | multi_rate |
|---|---|---|---|
| baseline | 0.3864 | 0.4284 | 0.3502 |
| weighted | 0.3864 | 0.3864 | 0.0000 |
| cosine | 0.3676 | 0.6078 | 1.0000 |
| maxlabel | 0.1867 | 0.5065 | 1.0000 |
| gated_cosine_theta=0.50 | 0.3676 | 0.6078 | 1.0000 |
| gated_cosine_theta=0.55 | 0.3676 | 0.6078 | 1.0000 |
| gated_cosine_theta=0.60 | 0.3676 | 0.6078 | 1.0000 |
| gated_cosine_theta=0.65 | 0.3676 | 0.6078 | 1.0000 |
| gated_cosine_theta=0.70 | 0.3676 | 0.6078 | 1.0000 |
| gated_cosine_theta=0.75 | 0.3676 | 0.6078 | 1.0000 |
| gated_cosine_theta=0.80 | 0.3676 | 0.6078 | 1.0000 |
| gated_cosine_theta=0.85 | 0.3676 | 0.6078 | 1.0000 |
| gated_cosine_theta=0.90 | 0.3676 | 0.6078 | 1.0000 |
| gated_maxlabel_theta=0.50 | 0.1867 | 0.5065 | 1.0000 |
| gated_maxlabel_theta=0.55 | 0.1867 | 0.5065 | 1.0000 |
| gated_maxlabel_theta=0.60 | 0.1867 | 0.5065 | 1.0000 |
| gated_maxlabel_theta=0.65 | 0.1867 | 0.5065 | 1.0000 |
| gated_maxlabel_theta=0.70 | 0.1867 | 0.5065 | 1.0000 |
| gated_maxlabel_theta=0.75 | 0.1867 | 0.5065 | 1.0000 |
| gated_maxlabel_theta=0.80 | 0.1867 | 0.5065 | 1.0000 |
| gated_maxlabel_theta=0.85 | 0.1867 | 0.5065 | 1.0000 |
| gated_maxlabel_theta=0.90 | 0.1867 | 0.5065 | 1.0000 |

## 5 折交叉验证（θ 只在训练折选）

- **gated_cosine**：strict **0.7175**，coverage **0.8055**，multi_rate 0.9916，θ/折=[0.9, 0.9, 0.9, 0.9, 0.9]
- **gated_maxlabel**：strict **0.2059**，coverage **0.7754**，multi_rate 0.9905，θ/折=[0.85, 0.9, 0.9, 0.9, 0.9]

口径说明：strict=dominant 命中；coverage=真实情绪在选中 meme 的标签集合（q≥0.3）；multi_rate=选中多标签 meme 比例。多标签会天然抬高 coverage，因此必须结合 multi_rate 与目检判断。