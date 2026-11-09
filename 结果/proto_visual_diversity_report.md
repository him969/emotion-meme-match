# 建议 3 离线验证：1 主推 + 2 视觉备选

- 数据：58 张 meme；验证集 3589 张（复用 09 缓存概率）
- 主推逻辑：软匹配选中情绪 + 同类随机（与 07 修复后一致）
- 备选：贪心 max-min（备选1 离主推最远；备选2 到已选两张的最小距离最大）
- 特征：imgnet = ImageNet ResNet18；fer = FER 微调 ResNet18
- 距离 = 1 - 余弦相似度（0 = 一样，2 = 完全相反）

## 1. 6 组合对比

| feature | mode | main_emotion_mismatch | same_emotion_alt_ratio | alt_s_weighted | pairwise_dist_mean | random_pairwise_dist_mean | near_dup_pairs | reroll_mean_distinct_sets | reroll_changed_ratio |
|---|---|---|---|---|---|---|---|---|---|
| fer | free | 0 | 0.363 | 0.710 | 0.623 | 0.504 | 0 | 5.790 | 0.985 |
| fer | majority | 0 | 0.536 | 0.795 | 0.613 | 0.506 | 0 | 5.535 | 0.973 |
| fer | same | 0 | 1.000 | 1.000 | 0.578 | 0.501 | 0 | 4.355 | 0.801 |
| imgnet | free | 0 | 0.371 | 0.692 | 0.668 | 0.530 | 0 | 5.915 | 0.992 |
| imgnet | majority | 0 | 0.590 | 0.799 | 0.663 | 0.535 | 0 | 5.730 | 0.982 |
| imgnet | same | 0 | 1.000 | 1.000 | 0.634 | 0.531 | 0 | 3.985 | 0.696 |

## 2. 解读

- `main_emotion_mismatch` 应全为 0：备选策略没有改变主推情绪；
- `pairwise_dist_mean` 越大说明三张越“不重样”；与 `random_pairwise_dist_mean` 对比看视觉选择是否有增量；
- `same_emotion_alt_ratio`：备选里同情绪比例（仅同类模式应=1.00）；
- `near_dup_pairs`：三张两两距离 <0.05（相似度>0.95）的对数，越少越好；
- `reroll_*`：模拟 1 次初始 + 5 次换一批，平均不同组合数 / 换一批成功率；
- 结论之后再定默认值（当前默认：same + imgnet）。

> 注：same 模式下 `reroll_mean_distinct_sets` 只有 ~4/6，是受"同类候选数量"限制（例如 disgust 4 张 → 主推 + 3 张候选 → 最多 3 对），属预期行为；majority/free 模式约 5.5~5.9。

## 3. 补充实验：避免备选挑到“视觉极端但内容奇怪”的图

目检样例（`proto_visual_examples_imgnet.png`）发现一个问题：**纯最大距离贪心**可能选中视觉上很远、但内容很奇怪/低信息量的图（例如只有一团绒毛纹理的 `cat-vibe`，被当作"最不同"的备选）。
于是在 majority 模式下测试了 3 种缓解方案（验证集 3589 张）：

| 方案 | pairwise_dist | same_ratio | near_dup |
|---|---|---|---|
| A. imgnet 最大距离（原方案） | **0.663** | 0.59 | 0 |
| C. imgnet 距离分位带 60%~95% | 0.644 | 0.55 | 0 |
| B. imgnet + FER 拼接特征后最大距离 | 0.607 | 0.55 | 0 |
| D. fer 最大距离 | 0.613 | 0.54 | 0 |

- 方案 C（分位带）只损失 0.019 的视觉多样性，但目检样例明显更“正常”（备选是 Nyan Cat、戴锥帽的猫等可辨认梗图，而不是纯纹理）；样例见 `proto_visual_mitigation_band.png`；
- 建议实现时用 **ImageNet 特征 + 距离分位带（60%~95%）** 作为默认备选选择策略，FER 特征作为用户可选项；
- 若要更简单，也可以保留最大距离 + “换一批”给用户兜底，但目检体验不如分位带稳。