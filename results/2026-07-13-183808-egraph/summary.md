# Benchmark run summary — 2026-07-13-183808-egraph

Benchmarks with results: **40**   |   candidates evaluated: **780**

## Programs (all candidates the model produced)
- valid (proven equivalent): **780 (100.0%)**
- invalid — missing e-graph rule (numerically equal, unproven): 0 (0.0%)
- invalid — non-equivalent (model error): 0 (0.0%)
- indeterminate (no finite sample point): 0 (0.0%)

## Benchmarks
- produced >=1 valid rewrite: **39/40 (97.5%)**
- accuracy improved over reference: **22/40 (55.0%)**
- valid rewrite but no accuracy gain: 14/40
- best valid rewrite was worse than reference: 2/40
- unmeasurable (box straddles zero / singularity): 1/40
- no valid rewrite found: 1/40
- had a missing-rule candidate: 0/40 (0.0%)

## Per-benchmark
(abs) = rel error undefined over the box; compared on absolute error instead

| benchmark | candidates | valid | best rel (ulp) | vs reference |
|---|--:|--:|--:|---|
| beta_a | 20 | 20 | - | improved (abs) |
| cancel_sqrt_2var | 20 | 20 | 2.2 | improved |
| cancel_sqrt_shift3 | 20 | 20 | 1.8 | improved |
| cancel_sqrt_sum | 20 | 20 | 1.8 | improved |
| complex_square_real | 20 | 20 | - | improved (abs) |
| conte_near_pole | 20 | 20 | 151.0 | improved |
| delta4 | 20 | 20 | - | improved (abs) |
| expand_square | 20 | 20 | - | improved (abs) |
| fastmath_dist4 | 20 | 20 | - | improved (abs) |
| floudas1 | 20 | 20 | - | improved (abs) |
| floudas3 | 20 | 20 | - | improved (abs) |
| himmilbeau | 20 | 20 | - | improved (abs) |
| martel_p6 | 20 | 20 | - | improved (abs) |
| nmse_example_3_1 | 20 | 20 | 1.6 | improved |
| nmse_p42_positive | 20 | 20 | 2.0 | improved (abs) |
| nonlin2 | 20 | 20 | 1.3 | improved |
| pbrt_cone_z | 20 | 20 | - | improved (abs) |
| rigidbody1 | 20 | 20 | - | improved (abs) |
| rigidbody2 | 20 | 20 | - | improved (abs) |
| som_setup_w | 20 | 20 | - | improved (abs) |
| sum | 20 | 20 | 0.8 | improved |
| test05_nonlin1_r4 | 20 | 20 | 0.8 | improved |
| asymptote_c | 20 | 20 | 68.8 | worse |
| beta_b | 20 | 20 | - | no-change (abs) |
| conte_x_minus_sqrt | 20 | 20 | - | no-change (abs) |
| excel_x0 | 20 | 20 | 7687.7 | no-change (abs) |
| kahan_p9 | 20 | 20 | - | no-change (abs) |
| kepler0 | 20 | 20 | 6.2 | worse |
| kepler1 | 0 | 0 | - | no-valid |
| matrixdeterminant | 20 | 20 | - | no-change (abs) |
| matrixdeterminant2 | 20 | 20 | - | no-change (abs) |
| nmse_example_3_6 | 20 | 20 | - | no-change (abs) |
| nmse_p42_negative | 20 | 20 | 17.9 | no-change (abs) |
| nmse_problem_3_2_1_negative | 20 | 20 | - | no-change (abs) |
| nmse_problem_3_2_1_positive | 20 | 20 | - | no-change (abs) |
| nmse_problem_3_3_1 | 20 | 20 | 118862.9 | no-change (abs) |
| nmse_problem_3_3_3 | 20 | 20 | - | no-change (abs) |
| sine | 20 | 20 | - | no-change (abs) |
| test03_nonlin2 | 20 | 20 | - | no-change (abs) |
| test04_dqmom9 | 20 | 20 | - | unmeasurable |
