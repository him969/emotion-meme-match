# 匹配层变体对比（val 3589 张，6 视角 TTA；val-only）

| 方案 | 严格 label_hit（5 折 CV） | 软命中@0.6 | 软命中@0.7 |
|---|---|---|---|
| hard | 0.7300 ± 0.0099 | 0.7999 | 0.7481 |
| soft | 0.7242 ± 0.0070 | 0.7994 | 0.7451 |
| soft_T0.65 | 0.7292 ± 0.0070 | 0.8019 | 0.7484 |
| blend_alpha | 0.7289 ± 0.0080 | 0.7997 | 0.7470 |
| gated_theta | 0.7236 ± 0.0079 | 0.7994 | 0.7445 |
| conf_sim | 0.7300 ± 0.0082 | 0.8011 | 0.7484 |

全量 val 上的乐观结果：

- hard: strict=0.7300, soft@0.6=0.7999, soft@0.7=0.7481
- soft: strict=0.7242, soft@0.6=0.7994, soft@0.7=0.7451
- soft_T0.65: strict=0.7292, soft@0.6=0.8019, soft@0.7=0.7484
- blend_a=0.00: strict=0.7300, soft@0.6=0.7999, soft@0.7=0.7481
- gated_theta=0.45: strict=0.7244, soft@0.6=0.7994, soft@0.7=0.7451
- conf_sim_lambda=0.6: strict=0.7303, soft@0.6=0.8013, soft@0.7=0.7487

说明：
- hard=取 argmax；soft=当前默认软匹配（手工相似矩阵）；soft_T0.65=温度缩放后软匹配；
- blend_alpha 在 argmax one-hot 与概率之间插值；gated_theta 高置信用 hard、低置信用 soft；
- conf_sim 用训练折混淆矩阵估计相似度并向手工矩阵收缩；
- 所有超参只在训练折选，报告为 5 折 CV；测试集未参与。