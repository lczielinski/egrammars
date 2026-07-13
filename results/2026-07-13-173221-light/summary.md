# Benchmark run summary — 2026-07-13-173221-light

Benchmarks with results: **48**   |   candidates evaluated: **960**

## Programs (all candidates the model produced)
- valid (proven equivalent): **278 (29.0%)**
- invalid — missing e-graph rule (numerically equal, unproven): 60 (6.2%)
- invalid — non-equivalent (model error): 496 (51.7%)
- indeterminate (no finite sample point): 126 (13.1%)

## Benchmarks
- produced >=1 valid rewrite: **41/48 (85.4%)**
- accuracy improved over reference: **21/48 (43.8%)**
- valid rewrite but no accuracy gain: 7/48
- best valid rewrite was worse than reference: 1/48
- unmeasurable (box straddles zero / singularity): 12/48
- no valid rewrite found: 7/48
- had a missing-rule candidate: 20/48 (41.7%)

## Per-benchmark
| benchmark | candidates | valid | best rel (ulp) | vs reference |
|---|--:|--:|--:|---|
| beta_b | 20 | 5 | - | improved |
| complex_square_real | 20 | 6 | - | improved |
| conte_near_pole | 20 | 2 | 60.3 | improved |
| conte_x_minus_sqrt | 20 | 14 | - | improved |
| expand_square | 20 | 10 | - | improved |
| fastmath_dist4 | 20 | 5 | - | improved |
| floudas3 | 20 | 9 | - | improved |
| kahan_p9 | 20 | 2 | - | improved |
| kepler0 | 20 | 6 | 2.5 | improved |
| matrixdeterminant2 | 20 | 10 | - | improved |
| nmse_problem_3_2_1_negative | 20 | 12 | 2.0 | improved |
| nonlin2 | 20 | 4 | 1.3 | improved |
| pbrt_cone_z | 20 | 8 | - | improved |
| rigidbody1 | 20 | 7 | - | improved |
| sec4_example | 20 | 9 | 1.3 | improved |
| som_setup_w | 20 | 9 | - | improved |
| sum | 20 | 8 | 0.8 | improved |
| test01_sum3 | 20 | 11 | 0.8 | improved |
| test03_nonlin2 | 20 | 5 | - | improved |
| test04_dqmom9 | 20 | 1 | - | improved |
| test05_nonlin1_r4 | 20 | 11 | 0.8 | improved |
| asymptote_c | 20 | 0 | - | no-valid |
| beta_a | 20 | 5 | - | no-change |
| cancel_sqrt_2var | 20 | 11 | 2.2 | unmeasurable |
| cancel_sqrt_shift3 | 20 | 10 | 1.8 | unmeasurable |
| cancel_sqrt_sum | 20 | 13 | 1.3 | unmeasurable |
| delta | 20 | 1 | - | unmeasurable |
| delta4 | 20 | 2 | 3.6 | unmeasurable |
| excel_x0 | 20 | 7 | 126.8 | unmeasurable |
| floudas1 | 20 | 0 | - | no-valid |
| himmilbeau | 20 | 6 | - | no-change |
| i4 | 20 | 7 | 1.0 | no-change |
| kahan_p13 | 20 | 1 | - | unmeasurable |
| kepler1 | 20 | 0 | - | no-valid |
| martel_p6 | 20 | 5 | - | worse |
| matrixdeterminant | 20 | 3 | - | unmeasurable |
| nmse_example_3_1 | 20 | 12 | 1.6 | unmeasurable |
| nmse_example_3_6 | 20 | 3 | 123.3 | no-change |
| nmse_p42_negative | 20 | 0 | - | no-valid |
| nmse_p42_positive | 20 | 8 | 2.5 | unmeasurable |
| nmse_problem_3_2_1_positive | 20 | 9 | - | unmeasurable |
| nmse_problem_3_3_1 | 20 | 8 | 83474.9 | unmeasurable |
| nmse_problem_3_3_3 | 20 | 0 | - | no-valid |
| rigidbody2 | 20 | 4 | - | no-change |
| sine | 20 | 0 | - | no-valid |
| test02_sum8 | 20 | 4 | 2.1 | no-change |
| triangle | 20 | 5 | 17.5 | no-change |
| triangle1 | 20 | 0 | - | no-valid |
