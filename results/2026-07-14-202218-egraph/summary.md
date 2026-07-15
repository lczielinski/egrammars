# Benchmark run summary — 2026-07-14-202218-egraph

Benchmarks with results: **43**   |   candidates evaluated: **860**

## Programs (all candidates the model produced)
- valid (proven equivalent): **860 (100.0%)**
- invalid — missing e-graph rule (numerically equal, unproven): 0 (0.0%)
- invalid — non-equivalent (model error): 0 (0.0%)
- indeterminate (no finite sample point): 0 (0.0%)

## Benchmarks
- produced >=1 valid rewrite: **43/43 (100.0%)**
- accuracy improved over reference: **36/43 (83.7%)**
- valid rewrite but no accuracy gain: 3/43
- best valid rewrite was worse than reference: 2/43
- unmeasurable (box straddles zero / singularity): 2/43
- no valid rewrite found: 0/43
- had a missing-rule candidate: 0/43 (0.0%)

## Per-benchmark
(abs) = rel error undefined over the box; compared on absolute error instead

| benchmark | candidates | valid | best rel (ulp) | vs reference |
|---|--:|--:|--:|---|
| asymptote_c | 20 | 20 | 3.0 | improved |
| beta_a | 20 | 20 | - | improved (abs) |
| beta_b | 20 | 20 | - | improved (abs) |
| cancel_sqrt_2var | 20 | 20 | 2.2 | improved |
| cancel_sqrt_shift3 | 20 | 20 | 3.3 | improved |
| cancel_sqrt_sum | 20 | 20 | 1.8 | improved |
| complex_square_real | 20 | 20 | - | improved (abs) |
| conte_near_pole | 20 | 20 | 125.4 | improved |
| conte_x_minus_sqrt | 20 | 20 | - | improved (abs) |
| delta4 | 20 | 20 | - | improved (abs) |
| excel_x0 | 20 | 20 | 126.8 | improved (abs) |
| expand_square | 20 | 20 | - | improved (abs) |
| fastmath_dist4 | 20 | 20 | - | improved (abs) |
| floudas1 | 20 | 20 | - | improved (abs) |
| floudas3 | 20 | 20 | - | improved (abs) |
| himmilbeau | 20 | 20 | - | improved (abs) |
| kahan_p9 | 20 | 20 | - | improved (abs) |
| kepler0 | 20 | 20 | 2.1 | improved |
| kepler1 | 20 | 20 | 12.4 | improved |
| martel_p6 | 20 | 20 | - | improved (abs) |
| matrixdeterminant | 20 | 20 | - | improved (abs) |
| matrixdeterminant2 | 20 | 20 | - | improved (abs) |
| nmse_example_3_1 | 20 | 20 | 1.6 | improved |
| nmse_p42_positive | 20 | 20 | - | improved (abs) |
| nmse_problem_3_2_1_positive | 20 | 20 | 509098.5 | improved (abs) |
| nmse_problem_3_3_1 | 20 | 20 | 83474.9 | improved (abs) |
| nmse_problem_3_3_3 | 20 | 20 | 285193.5 | improved (abs) |
| nonlin2 | 20 | 20 | 1.3 | improved |
| pbrt_cone_z | 20 | 20 | - | improved (abs) |
| rigidbody1 | 20 | 20 | - | improved (abs) |
| rigidbody2 | 20 | 20 | - | improved (abs) |
| sine | 20 | 20 | - | improved (abs) |
| som_setup_w | 20 | 20 | - | improved (abs) |
| sum | 20 | 20 | 0.8 | improved |
| test05_nonlin1_r4 | 20 | 20 | 0.8 | improved |
| triangle | 20 | 20 | 12.8 | improved |
| kepler2 | 20 | 20 | - | unmeasurable |
| nmse_example_3_6 | 20 | 20 | 51.7 | no-change (abs) |
| nmse_p42_negative | 20 | 20 | - | worse (abs) |
| nmse_problem_3_2_1_negative | 20 | 20 | 2.5 | worse (abs) |
| test03_nonlin2 | 20 | 20 | - | no-change (abs) |
| test04_dqmom9 | 20 | 20 | - | unmeasurable |
| triangle1 | 20 | 20 | 5.2 | no-change |
