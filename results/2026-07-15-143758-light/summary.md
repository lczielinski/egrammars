# Benchmark run summary — 2026-07-15-143758-light

Benchmarks with results: **42**   |   candidates evaluated: **420**

## Programs (all candidates the model produced)
- valid (proven equivalent): **130 (31.0%)**
- invalid — missing e-graph rule (numerically equal, unproven): 17 (4.0%)
- invalid — non-equivalent (model error): 271 (64.5%)
- indeterminate (no finite sample point): 2 (0.5%)

## Benchmarks
- produced >=1 valid rewrite: **37/42 (88.1%)**
- accuracy improved over reference: **21/42 (50.0%)**
- valid rewrite but no accuracy gain: 6/42
- best valid rewrite was worse than reference: 7/42
- unmeasurable (box straddles zero / singularity): 3/42
- no valid rewrite found: 5/42
- had a missing-rule candidate: 11/42 (26.2%)

## Per-benchmark
(abs) = rel error undefined over the box; compared on absolute error instead

| benchmark | candidates | valid | best rel (ulp) | vs reference |
|---|--:|--:|--:|---|
| cancel_sqrt_shift3 | 10 | 4 | 1.8 | improved |
| cancel_sqrt_sum | 10 | 4 | 1.8 | improved |
| complex_square_real | 10 | 4 | - | improved (abs) |
| conte_near_pole | 10 | 4 | 60.3 | improved |
| conte_x_minus_sqrt | 10 | 7 | - | improved (abs) |
| delta4 | 10 | 3 | - | improved (abs) |
| excel_x0 | 10 | 5 | 126.8 | improved (abs) |
| expand_square | 10 | 6 | - | improved (abs) |
| floudas3 | 10 | 1 | - | improved (abs) |
| martel_p6 | 10 | 6 | - | improved (abs) |
| matrixdeterminant | 10 | 6 | - | improved (abs) |
| matrixdeterminant2 | 10 | 4 | - | improved (abs) |
| nmse_example_3_1 | 10 | 4 | 1.6 | improved |
| nmse_problem_3_2_1_negative | 10 | 4 | - | improved (abs) |
| nmse_problem_3_3_3 | 10 | 1 | 313552.0 | improved (abs) |
| nonlin2 | 10 | 3 | 1.3 | improved |
| pbrt_cone_z | 10 | 2 | - | improved (abs) |
| rigidbody1 | 10 | 1 | - | improved (abs) |
| som_setup_w | 10 | 2 | - | improved (abs) |
| sum | 10 | 3 | 0.8 | improved |
| test05_nonlin1_r4 | 10 | 6 | 0.8 | improved |
| asymptote_c | 10 | 1 | - | unmeasurable |
| beta_a | 10 | 3 | - | worse (abs) |
| beta_b | 10 | 1 | - | worse (abs) |
| cancel_sqrt_2var | 10 | 1 | 181.7 | no-change |
| fastmath_dist4 | 10 | 5 | - | worse (abs) |
| floudas1 | 10 | 0 | - | no-valid |
| himmilbeau | 10 | 2 | - | no-change (abs) |
| kahan_p9 | 10 | 5 | - | no-change (abs) |
| kepler0 | 10 | 0 | - | no-valid |
| kepler1 | 10 | 0 | - | no-valid |
| nmse_example_3_6 | 10 | 3 | 265.3 | worse (abs) |
| nmse_p42_negative | 10 | 6 | 2.5 | no-change (abs) |
| nmse_p42_positive | 10 | 4 | 2.5 | no-change (abs) |
| nmse_problem_3_2_1_positive | 10 | 5 | 2.0 | unmeasurable |
| nmse_problem_3_3_1 | 10 | 4 | 119306.5 | worse (abs) |
| rigidbody2 | 10 | 2 | - | worse (abs) |
| sine | 10 | 0 | - | no-valid |
| test03_nonlin2 | 10 | 2 | - | worse (abs) |
| test04_dqmom9 | 10 | 1 | - | unmeasurable |
| triangle | 10 | 0 | - | no-valid |
| triangle1 | 10 | 5 | 5.2 | no-change |
