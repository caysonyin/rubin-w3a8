# Plan A execution report

Implementation Acceptance: PASS
Historical Baseline Acceptance: PASS
Smoke Acceptance: PASS
Experiment Acceptance: NOT_RUN


Local relative output errors and signed A8 Teacher MSE changes:

| Recipe | Projection | A16 relative error | A8 relative error | Activation MSE | A8 Teacher MSE delta |
|---|---|---:|---:|---:|---:|
| plain_lloyd_max | q_proj | 0.13847759 | 0.13940505 | 0.010671563 | 0.011344768 |
| h64_gptq_refit | q_proj | 0.12203164 | 0.12365974 | 0.017423303 | 0.017607918 |
| plain_lloyd_max | k_proj | 0.14595394 | 0.14689818 | 0.0081663612 | 0.0089634704 |
| h64_gptq_refit | k_proj | 0.11931078 | 0.1209376 | 0.012604942 | 0.012669017 |
| plain_lloyd_max | v_proj | 0.21022301 | 0.21161711 | 0.0074573917 | 0.0075576546 |
| h64_gptq_refit | v_proj | 0.19609715 | 0.1984158 | 0.011599256 | 0.01175544 |
| plain_lloyd_max | o_proj | 0.18098828 | 0.18237237 | 0.00073996947 | 0.0007601736 |
| h64_gptq_refit | o_proj | 0.20504019 | 0.20642021 | 0.00086190772 | 0.00085827239 |
| plain_lloyd_max | gate_proj | 0.12909501 | 0.12973063 | 0.00083339718 | 0.00086930684 |
| h64_gptq_refit | gate_proj | 0.091725762 | 0.093191567 | 0.0014114834 | 0.0014322689 |
| plain_lloyd_max | up_proj | 0.1686321 | 0.16966462 | 0.00071453512 | 0.00074137819 |
| h64_gptq_refit | up_proj | 0.14578491 | 0.14785412 | 0.0012840466 | 0.0012896138 |
| plain_lloyd_max | down_proj | 0.19363036 | 0.19393811 | 0.0012552205 | 0.0013036558 |
| h64_gptq_refit | down_proj | 0.11388715 | 0.11610927 | 0.0058100982 | 0.005586015 |

B1 − B0 a16: {'improved': 158, 'worsened': 38, 'tied': 0}

B1 − B0 a8: {'improved': 158, 'worsened': 38, 'tied': 0}

Largest A8 activation increments for plain_lloyd_max:

- model.layers.26.self_attn.q_proj: activation MSE=0.057037828, A8 Teacher MSE delta=0.059073953
- model.layers.25.self_attn.q_proj: activation MSE=0.052577718, A8 Teacher MSE delta=0.056639675
- model.layers.27.self_attn.q_proj: activation MSE=0.049028263, A8 Teacher MSE delta=0.054197094
- model.layers.27.self_attn.k_proj: activation MSE=0.048931541, A8 Teacher MSE delta=0.0630292
- model.layers.26.self_attn.v_proj: activation MSE=0.044790587, A8 Teacher MSE delta=0.045282058
- model.layers.24.self_attn.q_proj: activation MSE=0.043102727, A8 Teacher MSE delta=0.047132871
- model.layers.27.self_attn.v_proj: activation MSE=0.041364957, A8 Teacher MSE delta=0.040868768
- model.layers.25.self_attn.v_proj: activation MSE=0.039439496, A8 Teacher MSE delta=0.040594049
- model.layers.26.self_attn.k_proj: activation MSE=0.038534936, A8 Teacher MSE delta=0.038617959
- model.layers.25.self_attn.k_proj: activation MSE=0.03686784, A8 Teacher MSE delta=0.040343921

Largest A8 activation increments for h64_gptq_refit:

- model.layers.27.mlp.down_proj: activation MSE=0.11232141, A8 Teacher MSE delta=0.10421513
- model.layers.26.self_attn.q_proj: activation MSE=0.094982817, A8 Teacher MSE delta=0.095647655
- model.layers.25.self_attn.q_proj: activation MSE=0.081605033, A8 Teacher MSE delta=0.081840863
- model.layers.27.self_attn.q_proj: activation MSE=0.081401248, A8 Teacher MSE delta=0.083921161
- model.layers.27.self_attn.k_proj: activation MSE=0.07337382, A8 Teacher MSE delta=0.07713343
- model.layers.27.self_attn.v_proj: activation MSE=0.070831934, A8 Teacher MSE delta=0.072760412
- model.layers.24.self_attn.q_proj: activation MSE=0.066651426, A8 Teacher MSE delta=0.067211041
- model.layers.26.self_attn.v_proj: activation MSE=0.058233773, A8 Teacher MSE delta=0.059746984
- model.layers.26.self_attn.k_proj: activation MSE=0.056758219, A8 Teacher MSE delta=0.05410408
- model.layers.25.self_attn.k_proj: activation MSE=0.052805317, A8 Teacher MSE delta=0.053335227

Local errors do not establish layer PPL sensitivity, attribution to individual H64/GPTQ/refit steps, or hardware performance.
