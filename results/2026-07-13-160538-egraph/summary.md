# Benchmark run summary — 2026-07-13-160538-egraph

Benchmarks with results: **45**   |   candidates evaluated: **880**

## Programs (all candidates the model produced)
- valid (proven equivalent): **863 (98.1%)**
- invalid — missing e-graph rule (numerically equal, unproven): 17 (1.9%)
- invalid — non-equivalent (model error): 0 (0.0%)
- indeterminate (no finite sample point): 0 (0.0%)

## Benchmarks
- produced >=1 valid rewrite: **44/45 (97.8%)**
- accuracy improved over reference: **26/45 (57.8%)**
- valid rewrite but no accuracy gain: 10/45
- best valid rewrite was worse than reference: 3/45
- unmeasurable (box straddles zero / singularity): 5/45
- no valid rewrite found: 1/45
- had a missing-rule candidate: 5/45 (11.1%)

## Per-benchmark
| benchmark | candidates | valid | best rel (ulp) | vs reference |
|---|--:|--:|--:|---|
| beta_a | 20 | 20 | - | improved |
| beta_b | 20 | 20 | - | improved |
| complex_square_real | 20 | 20 | - | improved |
| conte_near_pole | 20 | 20 | 40.6 | improved |
| expand_square | 20 | 20 | - | improved |
| fastmath_dist4 | 20 | 20 | - | improved |
| floudas1 | 20 | 20 | - | improved |
| floudas3 | 20 | 20 | - | improved |
| himmilbeau | 20 | 20 | - | improved |
| martel_p6 | 20 | 20 | - | improved |
| nmse_example_3_1 | 20 | 19 | 1.6 | improved |
| nmse_example_3_6 | 20 | 20 | 151.0 | improved |
| nmse_p42_negative | 20 | 20 | 8.3 | improved |
| nmse_p42_positive | 20 | 18 | 2.0 | improved |
| nmse_problem_3_2_1_positive | 20 | 20 | - | improved |
| nmse_problem_3_3_3 | 20 | 20 | - | improved |
| nonlin2 | 20 | 20 | 1.3 | improved |
| pbrt_cone_z | 20 | 20 | - | improved |
| rigidbody1 | 20 | 20 | - | improved |
| rigidbody2 | 20 | 20 | - | improved |
| sec4_example | 20 | 20 | 1.3 | improved |
| sine | 20 | 20 | - | improved |
| som_setup_w | 20 | 20 | - | improved |
| sum | 20 | 20 | 0.8 | improved |
| test02_sum8 | 20 | 20 | 1.5 | improved |
| test05_nonlin1_r4 | 20 | 20 | 0.8 | improved |
| asymptote_c | 20 | 8 | 69.4 | worse |
| cancel_sqrt_2var | 20 | 20 | 2.2 | unmeasurable |
| cancel_sqrt_shift3 | 20 | 19 | 1.8 | unmeasurable |
| cancel_sqrt_sum | 20 | 20 | 1.8 | unmeasurable |
| conte_x_minus_sqrt | 20 | 19 | - | no-change |
| delta4 | 20 | 20 | - | no-change |
| excel_x0 | 20 | 20 | 7687.7 | no-change |
| i4 | 20 | 20 | 1.0 | no-change |
| kahan_p13 | 20 | 20 | 2.1 | no-change |
| kahan_p9 | 20 | 20 | - | no-change |
| kepler0 | 20 | 20 | 7.4 | worse |
| kepler1 | 0 | 0 | - | no-valid |
| matrixdeterminant | 20 | 20 | - | no-change |
| matrixdeterminant2 | 20 | 20 | - | no-change |
| nmse_problem_3_2_1_negative | 20 | 20 | - | no-change |
| nmse_problem_3_3_1 | 20 | 20 | 192317.4 | no-change |
| test01_sum3 | 20 | 20 | 0.8 | unmeasurable |
| test03_nonlin2 | 20 | 20 | - | worse |
| test04_dqmom9 | 20 | 20 | - | unmeasurable |
