# Update Plan A 最终修订版

## 实验目标与固定范围

在完全相同的 BF16 Teacher 输入上，比较现有两套量化 recipe 的独立层输出误差、权重重建误差和 A8 激活量化增量，定位误差集中位置及 B1 的适用边界。本轮完整实现 Isolated Layer Error，保留现有模型、数值定义、数据协议、实验规模和交付物。

| 标识 | 固定实现 | 作用 |
|---|---|---|
| Teacher | 原始 BF16 Qwen3-0.6B-Base | 提供统一的层输入和实际输出 |
| B0 | `plain_lloyd_max` | 现有 E4M3 约束 Lloyd-Max LUT-W3 |
| B1 | `h64_gptq_refit` | signed H64 + K64 GPTQ + Hessian LUT refit |

B0、B1 均诊断 W3A16 和 W3A8，同一 recipe 的两种激活模式使用相同 payload。运行限定为 CPU 数值参考，沿用 N8×K64、FP32 scales、现有 E4M3 激活规则及 `src/experiment.py` 中的 `FORMAT`、`CONFIG`。

目标集合为 28 个 Transformer block、每个 block 七类 projection，共 196 层。projection 顺序固定为：

```text
q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
```

通过现有 `eligible()` 枚举目标层，校验完整模块名、block 编号、projection 类型、形状及每类数量；缺层、重复层或额外目标层均停止实验验收。

历史 W3A8 PPL 为 B0 29.0666、B1 22.2245，仅作为研究背景。本轮不重算 PPL，不进行 Propagated Error、逐层 PPL 敏感性测试、H64/GPTQ/refit 分步骤消融、参数搜索或硬件性能实验。局部误差的变化不能直接解释为逐层 PPL 敏感性，也不能把组合 recipe 的收益归因于单个步骤。

既有数值协议见 [docs/numerical_spec_v1.md](numerical_spec_v1.md)，历史比较见 [docs/quantization_comparison.md](quantization_comparison.md)。

## Preflight 与两类验收

Preflight 在任何 payload 准备、校准或 Teacher forward 前运行。它只检查资源、配置和可验证的指纹，分开报告实现测试条件与实验运行条件；缺失资源不阻止编写实现和执行不依赖这些资源的独立测试。

| 检查项 | 检查内容 | 缺失或不一致时的处理 |
|---|---|---|
| 实现测试环境 | Python、锁定 CPU 依赖、pytest；图表测试另需 `plots` extra | 对应独立测试不能声称通过；仍可完成实现和静态检查 |
| 历史对照记录 | 五行历史结果、quantization JSON、配对 payload 指纹、固定配置 | 缺失为 BLOCKED；记录冲突为 FAIL，停止历史基线验收 |
| 本地 checkpoint | 完整模型文件、tokenizer、本地加载、模型与 tokenizer 指纹 | 缺失为 BLOCKED；指纹不符为 FAIL |
| 诊断数据 | 所选 split 缓存、固定文本处理、至少所需 token 数 | 缺失为 BLOCKED；禁止静默替换或缩短数据 |
| 校准数据 | 仅缺失 B1 payload 需要重建时，检查 train 前 512 tokens 及历史 hash | 缺失为 BLOCKED；hash 不符为 FAIL |
| Payload | 自校验、历史指纹、配置、校准元数据、196 层及形状 | 缺失且未启用准备选项为 BLOCKED；不一致为 FAIL |
| 执行配置 | CPU、BF16 Teacher、seed 42、准备 4 threads、诊断 20 threads、`use_cache=False` | 不符合固定协议为 FAIL |
| 输出与预算 | 新输出目录、写入条件、固定实验工作量 | 非空目录拒绝覆盖；不自动扩大工作量 |

缺失 payload 且显式启用准备选项时，只有 checkpoint、所需数据和配置检查通过，才允许准备缺失 payload。准备后再次运行 payload 和历史基线检查；全部通过后才能进行 1,024-token 验证和正式诊断。复用 payload 时，B1 的校准信息与历史记录比较，不要求额外运行校准 forward。

### Implementation Acceptance

Implementation Acceptance 表示工程实现通过，不代表正式实验完成。必须满足：

- 独立的小模型或合成张量测试通过，覆盖数值路径、共享权重重建、统计、schema、阈值、hook 生命周期和失败分支。
- 现有不依赖完整 checkpoint 或数据集的回归测试通过。
- 图表的合成数据测试通过；CSV 与图表入口可以分开使用。
- 资源缺失、指纹不符和未运行实验能按本节正确报告，不伪造通过标记。

模型、真实 payload 或 WikiText 缓存缺失时，Implementation Acceptance 可以为 PASS，Experiment Acceptance 必须保留为 BLOCKED。实际测试环境缺失或测试未执行时，Implementation Acceptance 为 NOT_RUN；不能以文档完成或静态检查代替测试通过。

### Experiment Acceptance

Experiment Acceptance 只有在 Preflight、历史基线验收、1,024-token 验证、4,096-token 正式统计和结果核验全部完成后才能为 PASS。`--smoke-only` 通过只表示验证阶段通过，正式 Experiment Acceptance 为 NOT_RUN。

`manifest.json` 分别记录以下字段，不使用单一的总体 PASS 覆盖不同阶段：

```text
implementation_acceptance: PASS | FAIL | NOT_RUN
historical_baseline_acceptance: PASS | FAIL | BLOCKED | NOT_RUN
smoke_acceptance: PASS | FAIL | BLOCKED | NOT_RUN
experiment_acceptance: PASS | FAIL | BLOCKED | NOT_RUN
```

各字段附实际执行证据和原因。资源缺失使用 BLOCKED，已执行的校验或数值检查失败使用 FAIL，尚未执行使用 NOT_RUN；报告必须保留阶段区别。

## 历史 Payload 的严格校验

以 `results/quantization_comparison/` 内已记录的结果为历史对照。分别读取 `plain_w3a16.json`、`plain_w3a8.json`、`h64_gptq_refit_w3a16.json`、`h64_gptq_refit_w3a8.json`，先确认每个 recipe 的 A16/A8 历史指纹相同，并与对应 quantization JSON 一致。

| Recipe | 历史 payload fingerprint |
|---|---|
| B0 | `b51d1413b50282f1062fe02645caacb6893a3d37d34d8daeca582ba173453de9` |
| B1 | `93da19247272e6ce0b6a7f0fd17c9c49b6f8462c34b3ddd8b3de842d8c68f043` |

使用 `src/experiment.py` 的 `validate_payload()` 和 `payload_hash()` 检查 payload 自身完整性，并独立比较历史模型、tokenizer、配置、校准和 payload 指纹。payload 自校验通过不等于历史基线验收通过。

优先复用已有 `.pt`。缺失时，仅在显式指定 `--prepare-payloads` 后调用现有 `build_payload()`；每个 recipe 最多准备一次。新 payload 保存到本次输出目录，不覆盖历史记录或已有 payload，不修改 payload 元数据以适配新诊断。

跨平台、依赖环境或重建导致指纹不同，均不得自动接受为原始 baseline。记录 expected/actual fingerprint、平台与数值依赖、准备线程数、配置和校准 hash，将 `historical_baseline_acceptance` 置为 FAIL，停止该次历史基线验收及正式诊断。Implementation Acceptance 可继续独立完成。不得改写 expected fingerprint、放宽比较、添加新指纹白名单或自动重试调参；接受新的复现实例需要另行决定，不属于本轮范围。

准备流程直接调用校准与 payload API。现有 `scripts/eval_qwen.py` 的 `--prepare-only` 仍会先加载 test tokens，且 B1 被限制为 test split，因此不借用该入口执行开发集诊断或新增准备流程。

## 数据协议与执行流程

固定数据集为 `Salesforce/wikitext`，配置 `wikitext-2-raw-v1`。以两个换行连接 `text`，一次性 tokenize，`add_special_tokens=False`，再按 token 索引截取。所有区间为半开区间。

| 用途 | 固定 token 区间 |
|---|---|
| B1 calibration | train `[0:512]`，保持历史校准协议和 token hash |
| 默认诊断 | validation `[0:4096]` |
| 默认验证 | 上述序列的前 1,024 tokens |
| 显式备用诊断 | train `[512:4608]` |
| 备用验证 | train `[512:1536]` |
| 历史最终 PPL | test 前 32,768 tokens，本轮不读取其文本或重新评测 |

默认选择 validation。缓存缺失时停止；只有显式指定 `--diagnostic-source train-holdout` 才使用备用区间。备用区间按位置不与校准 tokens 重叠，来源仍为 train，报告中保留这一区别。禁止静默替换 split、缩短 token 数或更换模型。

把选定的 4,096 个 token IDs 保存为冻结 JSON，记录 split、截取区间、token 数、tokenizer 指纹和 SHA256。验证和正式统计读取同一冻结序列。

正式统计使用四个独立的 1,024-token block，batch size 为 1，block 间重新建立上下文，`use_cache=False`。所有层输出统计包含全部 1,024 个 token 位置，不使用 PPL 每个 block 的 1,023 个预测位置计数。

执行顺序固定为：

1. Preflight；必要且显式授权时准备缺失 payload，再校验历史指纹。
2. 冻结诊断 tokens，运行 B0、B1 的 1,024-token 验证。
3. 验证通过后，运行 B0、B1 的 4,096-token 正式统计。
4. 重算输出统计和权重误差，完成两类验收记录、数据表、图表与报告。

Teacher 使用原始 BF16 模型，在 `eval()`、`torch.inference_mode()` 下运行。按 recipe 分轮，每轮仅加载一份 recipe payload，不通过 `apply_payload()` 替换 Teacher。对目标原始 Linear 注册 forward hook，读取原始输入和实际输出；hook 返回 `None`。所有 handles 在 `finally` 中移除，不改变 Teacher 参数、输入或输出。

每个目标层、每个 token block 记录 Teacher 输入和输出 hash，在 B0/B1 间逐项比较。hash、实际输出形状和计数任何一项不一致都停止 Experiment Acceptance。

## 数值路径与权重重建

令 `X` 为 Teacher 的 BF16 输入，`Y` 为原始 Linear 的实际 BF16 输出，`W` 为原始 BF16 权重提升到 FP32 后的数值。bias 保留原值并提升到 FP32；无 bias 时省略。`C` 表示输出转换回输入 dtype，本协议中为 BF16。

两个未量化控制保持不变：

\[
Y_{\mathrm{FP32}}=C\bigl(\operatorname{Linear}(X_{\mathrm{FP32}},W,b)\bigr)
\]

\[
Y_{\mathrm{H64}}=C\bigl(\operatorname{Linear}(XR,WR,b)\bigr)
\]

H64 control 使用 B1 对应层的固定 signs，与 B0 是否使用旋转无关。它用于观察配对旋转的有限精度影响。两个控制均相对实际 Teacher 输出计算误差，不要求误差为零，也不从总误差中直接相减来宣称纯量化损失。

量化路径为：

\[
U_{B0}=X_{\mathrm{FP32}},\quad U_{B1}=XR
\]

\[
Y_{m,16}=C\bigl(\operatorname{Linear}(U_m,\hat W_m,b)\bigr)
\]

\[
Y_{m,8}=C\bigl(\operatorname{Linear}(Q_8(U_m),\hat W_m,b)\bigr)
\]

`Q_8` 使用现有 K64 E4M3 fake quantization。B1 的 W3A16 保留 BF16 输入经旋转后的 FP32 数值，不增加旋转后的 BF16 舍入。所有主输出指标基于最终 BF16 输出。

每次处理一个目标层和 token block 时，只实例化一次 `src/linear.py` 中的 `CachedLUTLinear`。其构造函数执行一次 LUT 权重重建，然后用同一实例、同一 `weight_hat` 分别执行 A16 和 A8：依次设置 `quantize_activations=False`、`True`，调用现有 `forward()`。不创建第二个 Cached 实例，也不先单独重建再实例化。权重误差使用该实例已经重建的 `weight_hat`。

当前层统计完成后释放临时 Linear、重建权重和激活。不同 token block 可以重新重建当前层；保证的是每次 A16/A8 配对共享一次重建，不缓存全模型 FP32 权重或长期保存激活。每层权重误差只在首次处理时计算一次，后续 block 不重复累加。

B1 重建权重位于旋转坐标系，需映射回原坐标：

\[
R=DH_{64}/8,\quad R^T=H_{64}D/8,\quad
\hat W_{\mathrm{eff}}=\hat W_{B1}R^T
\]

在 `src/hadamard.py` 新增 `inverse_rotate(x, signs)`，先做 normalized Hadamard，再乘 signs，保持 FP32。不能直接再次调用相同 signed `rotate()`。生产实现按 K64 block 操作，不构造完整稠密 `R`；小矩阵显式 `R^T` 只用于独立测试。

## 指标及固定判定规则

统一 `epsilon = 1e-12`。误差统计在相减、平方之前把张量提升到 FP64；平方和与能量使用 FP64，计数使用整数。零参考能量按公式保留有限数值，并额外标记，不删除对应记录。

### 权重重建误差

\[
S_W=\|W-\hat W_{\mathrm{eff}}\|_F^2,\quad
T_W=\|W\|_F^2,\quad
E_W=\frac{\sqrt{S_W}}{\sqrt{T_W}+\epsilon}
\]

B0 直接使用重建权重，B1 使用上述逆变换。只统计原始有效权重形状，不包含 padding 和 bias。`weight_numel = in_features × out_features`，不随 token block 数增加。

### 输出 MSE 与 Relative Output Error

对每个层、recipe 和模式，先跨 token block 累积：

\[
S_{m,a}=\sum_t\|Y_{m,a,t}-Y_t\|_F^2,\quad
T_Y=\sum_t\|Y_t\|_F^2,\quad N=\sum_t\operatorname{numel}(Y_t)
\]

\[
\operatorname{MSE}_{m,a}=S_{m,a}/N,\quad
E_{\mathrm{rel},m,a}=\sqrt{S_{m,a}/(T_Y+\epsilon)}
\]

每层正式 `N = 4096 × out_features`。输出能量可由 `T_Y/N` 重算。Relative Output Error 是跨层比较和热力图的主指标。

### Activation Quantization Increment

\[
S_{A,m}=\sum_t\|Y_{m,8,t}-Y_{m,16,t}\|_F^2,\quad E_{A,m}=S_{A,m}/N
\]

它表示固定 W3 权重后 A8 引起的输出变化，不是 A8 相对 Teacher 的总 MSE 增量，不能与 W3A16 的总误差相加解释 W3A8 总误差。

### A8 相对 Teacher 的 MSE 变化

新增派生字段 `a8_teacher_mse_delta`：

\[
\Delta\operatorname{MSE}_{A8\mid\mathrm{Teacher},m}
=\operatorname{MSE}_{m,8}-\operatorname{MSE}_{m,16}
=\frac{S_{m,8}-S_{m,16}}{N}
\]

正值表示 A8 增大了相对 Teacher 的 MSE，负值表示减小，零表示相同。直接从已有平方和计算，保留正负号，不截断为非负值，不调用额外 Linear 或 Teacher forward。它与非负的 `activation_increment_mse` 分开报告，避免忽略交叉项。

### B1 相对 B0 的改善 恶化 持平

分别对 A16、A8 的 Relative Output Error 定义：

\[
\Delta E=E_{\mathrm{rel},B1}-E_{\mathrm{rel},B0},\quad
\tau=10^{-8}+10^{-6}\max\bigl(|E_{\mathrm{rel},B0}|,|E_{\mathrm{rel},B1}|\bigr)
\]

| 固定条件 | `comparison` 值 |
|---|---|
| `delta < -tau` | `improved` |
| `delta > tau` | `worsened` |
| `abs(delta) <= tau`，包含边界 | `tied` |

`comparison_atol = 1e-8`、`comparison_rtol = 1e-6` 在观察结果前固定，写入 manifest，不按结果或平台调整。保留未截断的原始 `delta` 和实际 `tau`；持平记录也不把原始差值置零。Figure A2 展示原始差值，层数分类、projection 汇总和报告统一使用这条判定规则。这里的阈值用于数值分类，不代表统计显著性或科研收益门槛。

### Projection 汇总

每类 projection 在每个 recipe 内汇总 28 层。先加和各层输出平方和、Teacher 能量和元素数，再计算 pooled MSE、pooled Relative Output Error、activation increment 及 `a8_teacher_mse_delta`。逐层 Relative Output Error 中位数单独报告。

权重汇总只加和每层一次的 `S_W`、`T_W` 和 `weight_numel`，再使用权重误差公式。中位数使用每层 `relative_weight_error`。不平均逐 block 比值作为 pooled 指标，不把重复权重统计或跨 recipe 重复控制结果累加。

## 固定 CSV Schema 与独立重算来源

四份 CSV 均使用 `schema_version = layerwise-diagnostics-csv-v1`。以下 header 的字段名称与顺序固定，不自动增加、删除或重排字段。CSV 为 UTF-8、逗号分隔、标准 quoting；整数为十进制，布尔值为 `0` 或 `1`，浮点数使用可 round-trip FP64 的 17 位有效数字，禁止 NaN/Inf。hash 为小写 64 位 SHA256。

字段类型固定如下：`schema_version`、`module_name`、`projection`、`weight_method`、`payload_fingerprint`、所有 hash 字段、`control_type`、`cross_recipe_check`、`comparison_a16`、`comparison_a8` 为字符串；`teacher_zero_energy`、`weight_zero_energy` 为布尔值，分别表示对应参考能量严格等于零；层编号、token block 编号、特征维度、block/token/元素/层计数以及六个改善/恶化/持平计数为整数；其余字段均为 FP64 数值。projection、recipe 和 control 的枚举取本文件中的固定集合，comparison 仅取 `improved|worsened|tied`，`cross_recipe_check` 仅取 `PASS|NOT_APPLICABLE`。

各阶段使用独立目录；阶段由父目录与 manifest 标识，不混合 smoke/full。除了明确列出的可空列，所有字段必须填写。模块全名使用 `model.layers.<transformer_block>...`，所有输出路径和文档引用使用仓库相对路径。

### layer_errors.csv

每个 `(module_name, weight_method)` 一条记录；smoke/full 各 392 行。顺序为 Transformer block 升序、固定 projection 顺序、B0/B1 顺序。

```csv
schema_version,module_name,transformer_block,projection,weight_method,payload_fingerprint,in_features,out_features,num_token_blocks,num_tokens,output_numel,teacher_output_energy,teacher_zero_energy,sse_a16,mse_a16,relative_error_a16,sse_a8,mse_a8,relative_error_a8,sse_a8_vs_a16,activation_increment_mse,a8_teacher_mse_delta,weight_numel,weight_sse,weight_reference_energy,weight_zero_energy,relative_weight_error,delta_relative_error_a16,comparison_tolerance_a16,comparison_a16,delta_relative_error_a8,comparison_tolerance_a8,comparison_a8
```

`weight_method` 仅为 `plain_lloyd_max` 或 `h64_gptq_refit`。`teacher_output_energy` 为跨 block 的 `T_Y`，不是均值。`sse_a8_vs_a16` 为 `S_A`。`weight_sse`、`weight_reference_energy` 为 `S_W`、`T_W`。

最后六个 comparison 字段仅填写在 B1 行；B0 行这六列为空。B1 行保留 B1−B0 的原始差值、实际阈值和分类。不在 B0 行复制 comparison 数据。

### projection_summary.csv

每个 `(projection, weight_method)` 一条记录，共 14 行。顺序为固定 projection 顺序、B0/B1 顺序。

```csv
schema_version,projection,weight_method,num_layers,num_token_blocks_per_layer,num_tokens_per_layer,output_numel,teacher_output_energy,sse_a16,mse_a16,relative_error_a16,median_relative_error_a16,sse_a8,mse_a8,relative_error_a8,median_relative_error_a8,sse_a8_vs_a16,activation_increment_mse,a8_teacher_mse_delta,weight_numel,weight_sse,weight_reference_energy,relative_weight_error,median_relative_weight_error,improved_layers_a16,worsened_layers_a16,tied_layers_a16,improved_layers_a8,worsened_layers_a8,tied_layers_a8
```

`num_layers = 28`。输出与权重充分统计量各自按前节规则累积。最后六个分类计数仅填写在 B1 行，B0 行为空；每个模式的三个计数之和必须为 28。全模型计数从对应 B1 行加和，每个模式总和为 196。

### precision_controls.csv

每个 `(module_name, control_type)` 只保留一条记录，两个控制各 196 行，共 392 行；不以 recipe 为行维度。顺序为 Transformer block 升序、固定 projection 顺序、FP32/H64 control 顺序。

```csv
schema_version,module_name,transformer_block,projection,control_type,num_token_blocks,num_tokens,output_numel,teacher_output_energy,teacher_zero_energy,control_sse,control_mse,control_relative_error,cross_recipe_check
```

`control_type` 仅为 `fp32_unquantized` 或 `h64_unquantized`。`control_mse = control_sse/output_numel`，`control_relative_error = sqrt(control_sse/(teacher_output_energy+epsilon))`。

FP32 control 在 B0/B1 处理中均出现，必须逐 token block 比较最终 BF16 控制输出 hash、Teacher hash、输出元素数、Teacher 能量及控制 SSE，要求完全一致。两轮核验通过后只选一份统计量汇总，写一行，`cross_recipe_check=PASS`；不得把两份重复控制相加或平均。

H64 control 按固定流程只在 B1 中计算，写一行，`cross_recipe_check=NOT_APPLICABLE`。若内部实现出现重复观察，也必须以同一 B1 signs 逐 block 完全一致后才能去重，检查不通过则失败，不能保留两行规避核验。

### block_stats.csv

每个 `(module_name, weight_method, token_block_index)` 一条记录；smoke 392 行，full 1,568 行。顺序为 Transformer block 升序、固定 projection 顺序、B0/B1 顺序、token block 升序。`token_block_index` 为该阶段内的 0-based 编号，与 Transformer block 编号不同。

```csv
schema_version,module_name,transformer_block,projection,weight_method,payload_fingerprint,token_block_index,num_tokens,output_numel,teacher_input_hash,teacher_output_hash,teacher_output_energy,sse_a16,sse_a8,sse_a8_vs_a16,sse_fp32_control,fp32_control_output_hash,sse_h64_control,h64_control_output_hash
```

`num_tokens = 1024`，各 SSE 和能量只覆盖当前 token block。B0 行的最后两列 H64 control 字段为空，B1 行必须填写。FP32 control 的重复 block 统计用于跨 recipe 核验，不使 `precision_controls.csv` 增加记录。

本表不包含权重统计：权重不随 token block 变化，不把 `weight_sse` 或 `weight_numel` 重复加入本表。

### 独立重算来源

| 指标 | 独立重算来源 | 重算方法 |
|---|---|---|
| Layer Output MSE、Relative Output Error、输出能量 | `block_stats.csv` | 按层与 recipe 加和 SSE、Teacher 能量和元素数，再使用固定公式 |
| Activation Increment、A8 Teacher MSE Delta | `block_stats.csv` 中三类输出 SSE | 分别计算 `sum(sse_a8_vs_a16)/N`、`(sum(sse_a8)-sum(sse_a16))/N` |
| Precision Controls | `block_stats.csv` 的控制 SSE、能量、计数和 hash | 先核验重复 FP32 control，再去重汇总；H64 control 使用 B1 block 记录 |
| B1−B0 原始差值、分类、projection 输出汇总 | 上述重算后的层指标 | 按模块名配对、应用固定阈值，再按 projection 汇总 |
| 每层 Weight Error | manifest 指定的原始 checkpoint、冻结 payload、固定逆变换 | 从原始 BF16 权重提升到 FP32；由 LUT/indices/scales 重建、必要时逆旋转，再按有效形状用 FP64 计算 `S_W`、`T_W`、`E_W`；无需 forward |
| Projection Weight Error | 独立重算后的每层权重统计 | 每层只计一次，再 pooled 或取中位数 |

由 `block_stats.csv` 重算输出指标验证的是充分统计量到汇总指标的计算链，不等于独立重现原始 forward 或原始输出 SSE。原始 SSE 的张量级重现需要相同 checkpoint、冻结 tokens、payload 和执行配置；本轮不自动增加重跑，不保存完整激活。

`layer_errors.csv` 中的权重统计可以用于核验 CSV 汇总，但不能作为自身权重误差的唯一独立来源。正式验收按 checkpoint/payload 独立重算权重，记录引用指纹与一致性检查；这不增加 forward。汇总指标重算及 weight 充分统计量比较使用 `atol=1e-12, rtol=1e-10`，该核验容差与 B1−B0 的分类阈值分开，不用于放宽 Teacher hash 或重复控制的严格一致性检查。

## 实现接口与交付物

新增 `src/diagnostics.py` 作为数值诊断模块，`scripts/diagnose_layers.py` 作为独立入口。复用现有量化、payload、hash、Cached Linear 和激活量化实现，保持历史评测入口与量化 recipe 行为。

入口参数固定为：

| 参数 | 行为 |
|---|---|
| `--model-path` | 默认 `models/Qwen3-0.6B-Base`，仅本地加载 |
| `--payload-dir` | 默认 `results/quantization_comparison`，读取两份冻结 `.pt` |
| `--output-dir` | 必填的新运行目录；非空目录拒绝写入 |
| `--diagnostic-source validation\|train-holdout` | 默认 validation，无自动 fallback |
| `--prepare-payloads` | 每个 recipe 只准备一次缺失 payload，保存到本次目录 |
| `--smoke-only` | 只运行验证，不生成正式实验通过标记 |
| `--preflight-only` | 只检查并记录前置条件，不准备 payload、不运行 forward；与前两项执行开关互斥 |

准备阶段在独立进程中使用 4 个 CPU threads，完成后释放该进程。诊断固定 20 个 CPU threads，统一设置并记录 OMP、MKL 和 PyTorch thread 数；seed 42。模型和数据使用本地离线加载。

验证与正式统计分别保存在本次输出目录下的 `smoke/`、`full/`。保留四份固定 CSV、`manifest.json`、`report.md`、冻结 token JSON，以及以下两张图，各导出 PNG 和 PDF：

- Figure A1：2×2 Relative Output Error 热力图；列为 B0/B1，行为 A16/A8，纵轴 block 0–27，横轴固定 projection 顺序。四个面板共享 `viridis` 线性色阶 `[0, 全局最大值]`。
- Figure A2：A16/A8 两个 B1−B0 原始 Relative Output Error 差值面板，共享以零为中心的对称色阶；负值为蓝、正值为红。改善、恶化、持平计数使用固定阈值；差值不按分类截断。全零色阶范围使用 `[-1e-12, 1e-12]`。

新增 `scripts/plot_layer_diagnostics.py`，仅读取已核验 CSV 和 manifest。引入 Matplotlib 的 `plots` 可选依赖，范围 `>=3.10.7,<4`；更新依赖锁时保留当前数值依赖版本。使用无界面后端，输出 300 dpi PNG 和 PDF。新增派生 MSE 变化只进入已有表格和报告，不增加图表或实验。

manifest 记录相对路径、模型/tokenizer/tokens/payload/代码与依赖指纹、recipe 配置、CPU 与数值执行选项、schema 版本、epsilon、comparison 阈值、阶段计数、逐项 Preflight、两类验收及阶段验收、实际耗时和峰值 RSS。外部本地资源在 manifest 中记录为相对仓库根目录的路径，不写入机器特定的绝对路径。

report 说明主要误差分布、A8 增量较大的层、A8 Teacher MSE Delta、B1 改善/恶化/持平数量、数值控制与结论边界。预置空结果或失败运行不能写成正式结论。

## 测试与 Experiment Acceptance 条件

### 独立实现测试

- 用非平凡 signs 验证正交性、配对旋转输出保持关系、逆变换顺序和小矩阵显式 `R^T` oracle；FP32 检查 `rtol=1e-5, atol=1e-6`。
- 小矩阵覆盖 bias、padding、BF16 输入以及 B0/B1 的 A16/A8，与现有 Cached/lookup 路径比较；固定 CPU 条件下最终 BF16 输出逐元素一致。
- 计数 `reconstruct_weight()` 调用，证明一次 A16/A8 配对只重建一次，使用同一 Cached 实例及同一 `weight_hat`，权重误差不触发第二次重建。
- 用可手工计算的数值检查 FP64 误差、能量、元素计数、zero-energy 标记、跨 block/projection pooling 和权重不重复累加。
- 检查 `a8_teacher_mse_delta` 的正、负、零结果，与 `activation_increment_mse` 分开；确认派生计算不增加任何 forward。
- 检查 delta 为零、阈值内、恰好 `±tau`、刚越过阈值及不同误差量级时的三类判定，保留原始差值。
- 检查固定 CSV header、主键、顺序、可空列、FP64 round-trip、row count 与拒绝 NaN/Inf；检查 Precision Controls 重复观察的一致性和去重。
- 检查 hook 不改变 Teacher 参数、输入和输出，正常与异常路径均释放 handles；校验 Teacher hashes。
- 覆盖缺失模型/数据/payload、历史与实际指纹不符、阶段未运行等情况，确保两类验收不会被合并或虚报。
- 用合成 CSV 检查图表轴顺序、共享色阶、差值符号、分类计数、全零差值处理及 PNG/PDF 导出。

### 正式实验通过条件

1. Preflight 与历史基线验收通过，两份 payload 严格匹配历史指纹。
2. 两种 recipe、每个阶段均覆盖完整 196 层；full 每层处理 4 个 token block、4,096 个 token 位置。
3. 两种 recipe 对应的 Teacher 输入、输出 hash 完全一致，重复 FP32 controls 的输出与统计完全一致。
4. 所有指标有限，schema、主键、行数和元素计数正确；precision controls 每层每种只保留一条。
5. 输出汇总能从 `block_stats.csv` 重算；权重误差能从原始 checkpoint 和冻结 payload 独立重算。
6. smoke 与 full 的第一个 token block hashes、计数及原始统计完全一致。
7. 数据表、manifest、图表与报告完整，保留原始差值、固定阈值分类及误差恶化记录。

工程通过与科研收益分开。B1 未普遍改善或某些 projection 的 A8 增量更大，仍是有效结果，不触发自动调参、换数据或扩展实验。

## 工作量预算与停止条件

沿用按工作量限制的预算，不设置时间或内存硬上限，记录实际资源消耗：

- 每个 recipe 最多重建一次缺失 payload。
- B0/B1 各完成一次 1,024-token 验证和一次 4,096-token 正式统计，合计 10 次 Teacher 的 1,024-token block forward。
- 只有重建 B1 时增加既定的一次 512-token train 校准 forward。
- 同一层的 A16/A8 使用已有两次局部计算；新增 Teacher MSE Delta 不增加局部 Linear 或 Teacher forward。
- 独立权重重算无需 forward；不增加完整实验重复、tokens、模型、recipe、参数搜索或 GPU 运行。

资源缺失、历史指纹不符、数值失败、Teacher 或控制不一致、schema/层集合/计数不完整时，停止相应实验阶段，保留失败日志及已完成证据。资源阻塞不自动阻止独立工程实现与测试，失败或阻塞也不等于科研结论。

完成后回传既定数据表、manifest、报告、图表、独立测试结果及任何协议偏离；分别报告 Implementation Acceptance 与 Experiment Acceptance。
