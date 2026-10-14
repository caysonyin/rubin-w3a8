# Plan B 固定消融协议

本轮回答固定顺序下 H64、GPTQ 和 LUT Refitting 对 W3A16/W3A8 的条件增量收益。
工程与实验基于已核实的 main `e6a3adc1ac7f52fa37fb47d00f87c418dcdc5b31`。
数值格式、模型、数据预算和历史 B0/B1 算法保持原有定义，不根据 Test 结果调参。

## Recipe 与公平性

| ID | weight_method | 变换与权重量化 | Calibration |
|---|---|---|---|
| B0 | plain_lloyd_max | 原始坐标 Lloyd-Max | 不使用 |
| A1 | h64_lloyd_max | 相同 signed H64 后 Lloyd-Max | 不使用 |
| A2 | h64_gptq | 相同 H64、初始化 LUT 和 K64 GPTQ；关闭所有 refit 轮次 | train 前 512 tokens |
| B1 | h64_gptq_refit | 原有 H64 + GPTQ + LUT Refitting | 原有 train 前 512 tokens |

沿用 N8×K64、FP32 max-abs/448 scales、八项 E4M3 LUT、50 次 Lloyd-Max
上限、B1 的阻尼、列处理顺序、死通道规则和三个 refit 轮次。A2 是 B1
首次 GPTQ sweep 的冻结结果；不做 nearest recoding，不以修改其他超参数来代替关闭 refit。
W3A16/W3A8 共用每种 Recipe 的同一份 payload。A8 在适用坐标中调用原有 K64
激活量化，最终输出依旧转换到输入 BF16 dtype。

A1/A2 的初始 LUT 必须逐层完全相同；A1/A2/B1 的 signs、scales 必须完全相同；
A2 的首轮 proxy objective 必须与历史 B1 trace 的首项完全相同。
B0/B1 保留原有 metadata/config 与 payload 指纹。额外 Recipe 使用独立 metadata，
不能被自动标记为历史 Baseline。

## 数据与执行

模型和数据 revision 沿用
[资源记录](../results/historical_rerun/resource_provenance.json)，所有模型/数据访问离线。
数值依赖和 `uv.lock` 不变。准备阶段 4 个 CPU threads；诊断和 Test 评测 20 个；
seed 42、BF16 原始 Teacher、`use_cache=False`。

- Calibration：WikiText-2 raw train 前 512 tokens，仅 A2/B1 使用。
- Validation：前 4,096 tokens；前 1,024 tokens 为独立 Smoke。
- Test：32,768 tokens，32 个 1,024-token block，32,736 个预测位置。

准备阶段不加载 Test 数据。所有 Recipe 配置在任何 Test 评测之前冻结；本轮只执行
上述四种 Recipe，不用 Validation 或 Test 分数修改量化算法。
B0/B1 的历史冻结 payload、BF16 参考分数及 Plan A 的局部诊断均可复用，
必须保留原始来源和验收证据。B0/B1 两种精度的 Test 分数在当前代码下重新评测，
与历史值严格逐项比较；不得用已有分数伪装为本次执行。A1/A2 各执行两种精度的完整 Test。

## 局部测量与独立验收

复用 [Plan A](update_plan_a.md) 的 `collect_recipe`、FP64 SSE、权重独立重建、
固定输出公式、FP32/H64 controls、hash 和 Smoke-prefix 核验。
四 Recipe 使用相同 Teacher 输入，局部输出不反馈进 Teacher。
原 Plan A 的默认两个 Recipe、Schema 和计算路径保持兼容。

Plan B 四份诊断 CSV 使用 `ablation-diagnostics-csv-v1`，字段名称/顺序与 Plan A
一致；Recipe 枚举显式扩展为 B0、A1、A2、B1。full 行数分别为：
`layer_errors.csv` 784、`projection_summary.csv` 28、
`precision_controls.csv` 392、`block_stats.csv` 3,136。
层表/投影表中的 comparison 字段统一表示各非 B0 Recipe 相对 B0。
另存相邻 Recipe 的逐层与投影增量表，避免将条件增量误认为 B0 对比。
FP32 controls 跨四 Recipe、H64 controls 跨三种旋转 Recipe 必须完全一致，再各保留一份。
新版本两种 control 的 `cross_recipe_check` 均为 PASS。

额外输入误差统计在 A1 的已有 full Teacher hook 内同时计算原始/H64 坐标的
`||Q_A8(X)-X||²`、参考能量、元素数和 hash，存为 block 与 layer 表，
不增加 Forward。H64 输入量化算子相同的 A1/A2/B1 共用该输入误差描述；
输出误差仍分别测量，因为其重建权重不同。
输入误差也按充分统计量 pooled；不同特征维度的 MSE 不直接混作同一总体。

工程验收先运行完整测试，保留独立 log 和源码 hash。
实验验收要求资源与历史身份校验、新增 payload 结构/公平性验证、
Smoke、full、独立 weight/reduction/control 检查，以及八个完整 Test 结果全部通过。
缺资源记 BLOCKED，身份或数值异常记 FAIL，保留失败记录。
历史工程证据不因新增代码而覆盖；当前工程证据保存在 Plan B 目录。

## 分析与交付

主指标为每种 Recipe 的 Mean NLL 与 PPL；报告 A8−A16 NLL 增量。
条件对比顺序为 B0→A1、A1→A2、A2→B1，各报告原始 NLL/PPL 差值和 PPL 比率。
PPL 差值只作描述，不作可加的误差归因；NLL 条件差值可按该顺序相加。
局部误差沿用 Plan A 分类阈值
`tau=1e-8+1e-6*max(abs(E_before),abs(E_after))`，保留原始差值。
该阈值为数值分类规则，不是统计显著性检验。

报告结合 `o_proj`、activation increment MSE、signed Teacher MSE delta 与
输入误差解释已观察到的现象。明确区分实验观察、机制推断和尚未验证的假设；
单一模型/语料/seed 无跨 seed 方差估计，顺序消融不证明独立因果贡献。
后续算法建议只形成研究建议，不在本轮追加新 Recipe、模型、格式或硬件开发。

交付包含代码与正确性/回归测试、四 Recipe 对比和条件增量 CSV、
输入与输出诊断、来源/配置/验收 manifest、完整日志、PNG/PDF 和研究报告。
执行后本地提交可审查结果；不执行 Git Push 或 PR Merge。
