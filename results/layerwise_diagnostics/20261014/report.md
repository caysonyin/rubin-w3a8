# Plan A execution report

Implementation Acceptance: PASS
Historical Baseline Acceptance: PASS
Smoke Acceptance: PASS
Experiment Acceptance: PASS


Local relative output errors and signed A8 Teacher MSE changes:

| Recipe | Projection | A16 relative error | A8 relative error | Activation MSE | A8 Teacher MSE delta |
|---|---|---:|---:|---:|---:|
| plain_lloyd_max | q_proj | 0.13975838 | 0.14069895 | 0.011075436 | 0.011526606 |
| h64_gptq_refit | q_proj | 0.11995405 | 0.12164233 | 0.017704706 | 0.017822952 |
| plain_lloyd_max | k_proj | 0.146862 | 0.14780787 | 0.0085013095 | 0.0091138941 |
| h64_gptq_refit | k_proj | 0.11627817 | 0.1179718 | 0.012862481 | 0.012972837 |
| plain_lloyd_max | v_proj | 0.20791108 | 0.20932137 | 0.0076814697 | 0.0078609294 |
| h64_gptq_refit | v_proj | 0.18983147 | 0.19207552 | 0.011374426 | 0.011449254 |
| plain_lloyd_max | o_proj | 0.17832616 | 0.1797144 | 0.00070412067 | 0.00073552353 |
| h64_gptq_refit | o_proj | 0.19661876 | 0.19802423 | 0.00082402519 | 0.00082078299 |
| plain_lloyd_max | gate_proj | 0.13058067 | 0.13125489 | 0.0008546744 | 0.00092011823 |
| h64_gptq_refit | gate_proj | 0.090165629 | 0.091669913 | 0.0014020406 | 0.0014256809 |
| plain_lloyd_max | up_proj | 0.16906078 | 0.17012128 | 0.00072917023 | 0.00076012771 |
| h64_gptq_refit | up_proj | 0.14281301 | 0.14491626 | 0.001263495 | 0.0012788527 |
| plain_lloyd_max | down_proj | 0.19167023 | 0.1919777 | 0.0013066518 | 0.0013375177 |
| h64_gptq_refit | down_proj | 0.10539717 | 0.10778707 | 0.0060350595 | 0.0057769831 |

B1 − B0 a16: {'improved': 170, 'worsened': 26, 'tied': 0}

B1 − B0 a8: {'improved': 170, 'worsened': 26, 'tied': 0}

Largest A8 activation increments for plain_lloyd_max:

- model.layers.26.self_attn.q_proj: activation MSE=0.059129534, A8 Teacher MSE delta=0.059987102
- model.layers.25.self_attn.q_proj: activation MSE=0.055320698, A8 Teacher MSE delta=0.05856935
- model.layers.27.self_attn.q_proj: activation MSE=0.050829031, A8 Teacher MSE delta=0.053849922
- model.layers.27.self_attn.k_proj: activation MSE=0.050430397, A8 Teacher MSE delta=0.057970901
- model.layers.26.self_attn.v_proj: activation MSE=0.046126612, A8 Teacher MSE delta=0.047073929
- model.layers.24.self_attn.q_proj: activation MSE=0.044766525, A8 Teacher MSE delta=0.046538096
- model.layers.27.self_attn.v_proj: activation MSE=0.041991682, A8 Teacher MSE delta=0.042305958
- model.layers.25.self_attn.v_proj: activation MSE=0.040956428, A8 Teacher MSE delta=0.042431469
- model.layers.26.self_attn.k_proj: activation MSE=0.040554881, A8 Teacher MSE delta=0.041715164
- model.layers.25.self_attn.k_proj: activation MSE=0.038773899, A8 Teacher MSE delta=0.042152101

Largest A8 activation increments for h64_gptq_refit:

- model.layers.27.mlp.down_proj: activation MSE=0.11746467, A8 Teacher MSE delta=0.11000267
- model.layers.26.self_attn.q_proj: activation MSE=0.098354028, A8 Teacher MSE delta=0.097536703
- model.layers.25.self_attn.q_proj: activation MSE=0.085305721, A8 Teacher MSE delta=0.086304809
- model.layers.27.self_attn.q_proj: activation MSE=0.082445974, A8 Teacher MSE delta=0.08433635
- model.layers.27.self_attn.k_proj: activation MSE=0.074935696, A8 Teacher MSE delta=0.0777387
- model.layers.27.self_attn.v_proj: activation MSE=0.069343939, A8 Teacher MSE delta=0.070152883
- model.layers.24.self_attn.q_proj: activation MSE=0.068854587, A8 Teacher MSE delta=0.069226235
- model.layers.26.self_attn.k_proj: activation MSE=0.059475598, A8 Teacher MSE delta=0.05935829
- model.layers.26.self_attn.v_proj: activation MSE=0.058771757, A8 Teacher MSE delta=0.059401171
- model.layers.25.self_attn.k_proj: activation MSE=0.055164995, A8 Teacher MSE delta=0.055101804

Local errors do not establish layer PPL sensitivity, attribution to individual H64/GPTQ/refit steps, or hardware performance.

## Main findings from the full validation diagnostic

B1 reduces relative output error in 170 of 196 layers (86.73%) in both A16 and A8. The 26 worsened layers consist of 24 o_proj and two down_proj layers; no layers are classified as tied. All q_proj, k_proj, v_proj, gate_proj and up_proj layers improve. At projection level, six families improve while o_proj worsens: its pooled A8 relative error rises from 0.17971440 to 0.19802423.

Relative weight reconstruction error is higher for B1 in all 196 layers, even though local output error improves in 170. Thus, on this fixed model/protocol, unweighted Frobenius weight error alone does not predict teacher-input output error. This is a finding for the combined recipe and does not attribute a mechanism to an individual optimization step.

The A8-versus-A16 activation increment MSE is higher for B1 in all 196 matched layers and in every pooled projection family. B1 still has lower total output error in most layers and lower historical held-out PPL. The activation perturbation and total quantization error must be interpreted separately.

The signed Teacher MSE delta also differs from the nonnegative activation increment. For B0 model.layers.2.mlp.down_proj, activation increment MSE is 0.00050573926, while A8 Teacher MSE delta is -0.00066208143. This observed cancellation illustrates the error cross term without additional forwards.

The maximum relative error among the unquantized numerical controls is 0.00028153578. These controls, fixed inputs, historical payload fingerprints and independent reductions support the numerical measurement validity. Local error reductions do not establish PPL sensitivity.
