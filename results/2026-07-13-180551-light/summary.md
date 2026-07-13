# Benchmark run summary — 2026-07-13-180551-light

Benchmarks with results: **43**   |   candidates evaluated: **860**

## Programs (all candidates the model produced)
- valid (proven equivalent): **233 (27.1%)**
- invalid — missing e-graph rule (numerically equal, unproven): 53 (6.2%)
- invalid — non-equivalent (model error): 462 (53.7%)
- indeterminate (no finite sample point): 112 (13.0%)

## Benchmarks
- produced >=1 valid rewrite: **36/43 (83.7%)**
- accuracy improved over reference: **24/43 (55.8%)**
- valid rewrite but no accuracy gain: 5/43
- best valid rewrite was worse than reference: 6/43
- unmeasurable (box straddles zero / singularity): 1/43
- no valid rewrite found: 7/43
- had a missing-rule candidate: 15/43 (34.9%)

## Per-benchmark
(abs) = rel error undefined over the box; compared on absolute error instead

| benchmark | candidates | valid | best rel (ulp) | vs reference |
|---|--:|--:|--:|---|
| cancel_sqrt_2var | 20 | 12 | 2.2 | improved |
| cancel_sqrt_shift3 | 20 | 13 | 1.1 | improved |
| cancel_sqrt_sum | 20 | 14 | 1.4 | improved |
| complex_square_real | 20 | 9 | - | improved (abs) |
| conte_x_minus_sqrt | 20 | 11 | - | improved (abs) |
| excel_x0 | 20 | 6 | 116.1 | improved (abs) |
| expand_square | 20 | 8 | - | improved (abs) |
| fastmath_dist4 | 20 | 9 | - | improved (abs) |
| floudas1 | 20 | 1 | - | improved (abs) |
| floudas3 | 20 | 8 | - | improved (abs) |
| kahan_p9 | 20 | 3 | - | improved (abs) |
| kepler0 | 20 | 4 | 2.5 | improved |
| kepler1 | 20 | 1 | 14.0 | improved |
| matrixdeterminant | 20 | 2 | - | improved (abs) |
| nmse_example_3_1 | 20 | 14 | 1.6 | improved |
| nmse_p42_positive | 20 | 10 | 2.0 | improved (abs) |
| nmse_problem_3_2_1_negative | 20 | 6 | 2.0 | improved (abs) |
| nmse_problem_3_3_3 | 20 | 1 | 3.0 | improved (abs) |
| nonlin2 | 20 | 5 | 1.3 | improved |
| rigidbody1 | 20 | 7 | - | improved (abs) |
| sine | 20 | 7 | - | improved (abs) |
| som_setup_w | 20 | 6 | - | improved (abs) |
| sum | 20 | 12 | 0.8 | improved |
| test05_nonlin1_r4 | 20 | 6 | 0.8 | improved |
| asymptote_c | 20 | 0 | - | no-valid |
| beta_a | 20 | 4 | - | no-change (abs) |
| beta_b | 20 | 1 | - | worse (abs) |
| conte_near_pole | 20 | 0 | - | no-valid |
| delta4 | 20 | 3 | - | worse (abs) |
| himmilbeau | 20 | 3 | - | no-change (abs) |
| kepler2 | 20 | 0 | - | no-valid |
| martel_p6 | 20 | 9 | - | no-change (abs) |
| matrixdeterminant2 | 20 | 0 | - | no-valid |
| nmse_example_3_6 | 20 | 4 | 126.1 | worse (abs) |
| nmse_p42_negative | 20 | 2 | - | unmeasurable |
| nmse_problem_3_2_1_positive | 20 | 8 | 2.0 | no-change (abs) |
| nmse_problem_3_3_1 | 20 | 9 | 119306.5 | worse (abs) |
| pbrt_cone_z | 20 | 8 | - | worse (abs) |
| rigidbody2 | 20 | 5 | - | worse (abs) |
| test03_nonlin2 | 20 | 0 | - | no-valid |
| test04_dqmom9 | 20 | 0 | - | no-valid |
| triangle | 20 | 2 | 17.6 | no-change |
| triangle1 | 20 | 0 | - | no-valid |
