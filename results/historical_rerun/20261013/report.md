# Historical experiment reproduction

Acceptance: PASS

| Group | Row | Δmean NLL | ΔPPL |
|---|---|---:|---:|
| original | bf16 | 0 | 0 |
| original | w3a16 | 0 | 0 |
| original | w3a8 | 0 | 0 |
| comparison | bf16 | 0 | 0 |
| comparison | plain_w3a16 | 0 | 0 |
| comparison | plain_w3a8 | 0 | 0 |
| comparison | h64_gptq_refit_w3a16 | 0 | 0 |
| comparison | h64_gptq_refit_w3a8 | 0 | 0 |

Original sanity_linear and sanity_block: PASS. All eight historical mean NLL/PPL deltas are exactly zero.
