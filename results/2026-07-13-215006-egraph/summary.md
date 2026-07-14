# Benchmark run summary — 2026-07-13-215006-egraph

Benchmarks with results: **43**   |   candidates evaluated: **860**

## Programs (all candidates the model produced)
- valid (proven equivalent): **860 (100.0%)**
- invalid — missing e-graph rule (numerically equal, unproven): 0 (0.0%)
- invalid — non-equivalent (model error): 0 (0.0%)
- indeterminate (no finite sample point): 0 (0.0%)

## Benchmarks
- produced >=1 valid rewrite: **43/43 (100.0%)**
- accuracy improved over reference: **26/43 (60.5%)**
- valid rewrite but no accuracy gain: 12/43
- best valid rewrite was worse than reference: 3/43
- unmeasurable (box straddles zero / singularity): 2/43
- no valid rewrite found: 0/43
- had a missing-rule candidate: 0/43 (0.0%)

## Per-benchmark
(abs) = rel error undefined over the box; compared on absolute error instead

| benchmark | candidates | valid | best rel (ulp) | vs reference |
|---|--:|--:|--:|---|
| asymptote_c | 20 | 20 | 2.4 | improved |
| beta_b | 20 | 20 | - | improved (abs) |
| cancel_sqrt_2var | 20 | 20 | 2.2 | improved |
| cancel_sqrt_shift3 | 20 | 20 | 1.8 | improved |
| cancel_sqrt_sum | 20 | 20 | 1.5 | improved |
| complex_square_real | 20 | 20 | - | improved (abs) |
| conte_near_pole | 20 | 20 | 151.0 | improved |
| expand_square | 20 | 20 | - | improved (abs) |
| fastmath_dist4 | 20 | 20 | - | improved (abs) |
| floudas1 | 20 | 20 | - | improved (abs) |
| floudas3 | 20 | 20 | - | improved (abs) |
| himmilbeau | 20 | 20 | - | improved (abs) |
| martel_p6 | 20 | 20 | - | improved (abs) |
| nmse_example_3_1 | 20 | 20 | 1.6 | improved |
| nmse_p42_positive | 20 | 20 | 221505.2 | improved (abs) |
| nmse_problem_3_3_3 | 20 | 20 | - | improved (abs) |
| nonlin2 | 20 | 20 | 1.3 | improved |
| pbrt_cone_z | 20 | 20 | - | improved (abs) |
| rigidbody1 | 20 | 20 | - | improved (abs) |
| rigidbody2 | 20 | 20 | - | improved (abs) |
| sine | 20 | 20 | - | improved (abs) |
| som_setup_w | 20 | 20 | - | improved (abs) |
| sum | 20 | 20 | 0.8 | improved |
| test05_nonlin1_r4 | 20 | 20 | 0.8 | improved |
| triangle | 20 | 20 | 13.8 | improved |
| triangle1 | 20 | 20 | 4.4 | improved |
| beta_a | 20 | 20 | - | no-change (abs) |
| conte_x_minus_sqrt | 20 | 20 | - | no-change (abs) |
| delta4 | 20 | 20 | - | worse (abs) |
| excel_x0 | 20 | 20 | 7687.7 | no-change (abs) |
| jetengine | 20 | 20 | - | no-change (abs) |
| kahan_p9 | 20 | 20 | - | no-change (abs) |
| kepler0 | 20 | 20 | 6.1 | worse |
| kepler2 | 20 | 20 | 61.1 | unmeasurable |
| matrixdeterminant | 20 | 20 | - | no-change (abs) |
| matrixdeterminant2 | 20 | 20 | - | no-change (abs) |
| nmse_example_3_6 | 20 | 20 | - | no-change (abs) |
| nmse_p42_negative | 20 | 20 | - | no-change (abs) |
| nmse_problem_3_2_1_negative | 20 | 20 | 233367.0 | no-change (abs) |
| nmse_problem_3_2_1_positive | 20 | 20 | - | no-change (abs) |
| nmse_problem_3_3_1 | 20 | 20 | 6312.5 | worse (abs) |
| test03_nonlin2 | 20 | 20 | - | no-change (abs) |
| test04_dqmom9 | 20 | 20 | - | unmeasurable |
