# Benchmark run summary

Benchmarks with results: **31**   |   candidates evaluated: **620**

## Programs (all candidates the model produced)
- valid (proven equivalent): **148 (23.9%)**
- invalid — missing e-graph rule (numerically equal, unproven): 37 (6.0%)
- invalid — non-equivalent (model error): 326 (52.6%)
- indeterminate (no finite sample point): 109 (17.6%)

## Benchmarks
- produced >=1 valid rewrite: **22/31 (71.0%)**
- accuracy improved over reference: **9/31 (29.0%)**
- valid rewrite but no accuracy gain: 4/31
- best valid rewrite was worse than reference: 1/31
- unmeasurable (box straddles zero / singularity): 8/31
- no valid rewrite found: 9/31
- had a missing-rule candidate: 13/31 (41.9%)

## Per-benchmark
| benchmark | candidates | valid | best rel (ulp) | vs reference |
|---|--:|--:|--:|---|
| floudas3 | 20 | 4 | - | improved |
| i4 | 20 | 12 | 0.8 | improved |
| matrixdeterminant2 | 20 | 7 | - | improved |
| nonlin2 | 20 | 5 | 1.3 | improved |
| sec4_example | 20 | 4 | 1.3 | improved |
| sine | 20 | 2 | - | improved |
| sum | 20 | 11 | 0.8 | improved |
| test01_sum3 | 20 | 15 | 0.8 | improved |
| test05_nonlin1_r4 | 20 | 8 | 0.8 | improved |
| delta | 20 | 2 | - | unmeasurable |
| delta4 | 20 | 0 | - | no-valid |
| floudas1 | 20 | 0 | - | no-valid |
| himmilbeau | 20 | 4 | - | no-change |
| kepler0 | 20 | 1 | - | unmeasurable |
| kepler2 | 20 | 0 | - | no-valid |
| matrixdeterminant | 20 | 2 | - | no-change |
| nmse_example_3_1 | 20 | 15 | - | unmeasurable |
| nmse_example_3_6 | 20 | 13 | - | unmeasurable |
| nmse_p42_negative | 20 | 0 | - | no-valid |
| nmse_p42_positive | 20 | 0 | - | no-valid |
| nmse_problem_3_2_1_negative | 20 | 3 | - | unmeasurable |
| nmse_problem_3_2_1_positive | 20 | 11 | - | unmeasurable |
| nmse_problem_3_3_1 | 20 | 11 | - | unmeasurable |
| nmse_problem_3_3_3 | 20 | 3 | - | unmeasurable |
| rigidbody1 | 20 | 0 | - | no-valid |
| rigidbody2 | 20 | 2 | - | worse |
| test02_sum8 | 20 | 2 | 2.1 | no-change |
| test03_nonlin2 | 20 | 11 | - | no-change |
| test04_dqmom9 | 20 | 0 | - | no-valid |
| triangle | 20 | 0 | - | no-valid |
| triangle1 | 20 | 0 | - | no-valid |
