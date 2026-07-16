# Herbie comparison — 2026-07-15-221435

## Average bits of error (Herbie's metric, same sampled points)

Lower is better; `ours` = best proven program, scored via `:alt`.

| benchmark | reference | ours (best) | herbie | winner |
|---|--:|--:|--:|---|
| asymptote_c | 1.59 | 0.30 | 0.28 | tie |
| beta_a | 2.83 | 2.48 | 2.19 | herbie |
| beta_b | 2.87 | 2.52 | 2.23 | herbie |
| cancel_sqrt_2var | 0.01 | 0.01 | 0.01 | tie |
| cancel_sqrt_shift3 | 2.95 | 0.02 | 0.02 | tie |
| cancel_sqrt_sum | 0.01 | 0.01 | 0.01 | tie |
| complex_square_real | 1.11 | 0.24 | 0.25 | tie |
| conte_near_pole | 8.14 | 0.32 | 0.32 | tie |
| conte_x_minus_sqrt | 59.09 | 0.02 | 0.24 | ours |
| excel_x0 | 7.21 | 0.30 | 0.31 | tie |
| expand_square | 58.42 | 0.00 | 0.01 | tie |
| fastmath_dist4 | 0.75 | 0.44 | 0.53 | tie |
| floudas3 | 0.14 | 0.14 | 0.14 | tie |
| kahan_p9 | 0.02 | 0.02 | 0.02 | tie |
| martel_p6 | 3.65 | 0.00 | 0.02 | tie |
| matrixdeterminant | 0.17 | 0.16 | 0.17 | tie |
| matrixdeterminant2 | 0.16 | 0.16 | 0.17 | tie |
| nmse_example_3_1 | 1.36 | 0.45 | 0.26 | herbie |
| nmse_example_3_6 | 1.44 | 0.73 | 0.49 | herbie |
| nmse_p42_negative | 0.29 | 0.29 | 0.29 | tie |
| nmse_p42_positive | 0.29 | 0.29 | 0.29 | tie |
| nmse_problem_3_2_1_negative | 0.30 | 0.28 | 0.30 | tie |
| nmse_problem_3_2_1_positive | 0.30 | 0.29 | 0.30 | tie |
| nmse_problem_3_3_1 | 1.61 | 0.32 | 0.32 | tie |
| nmse_problem_3_3_3 | 5.66 | 0.36 | 0.36 | tie |
| nonlin2 | 0.40 | 0.29 | 0.33 | tie |
| pbrt_cone_z | 0.03 | 0.02 | 0.02 | tie |
| rigidbody1 | 0.00 | 0.00 | 0.01 | tie |
| rigidbody2 | 0.07 | 0.07 | 0.07 | tie |
| sine | 0.00 | - | 0.00 | - |
| som_setup_w | 3.15 | 0.25 | 0.26 | tie |
| sum | 0.35 | 0.02 | 0.14 | ours |
| test03_nonlin2 | 0.04 | 0.03 | 0.03 | tie |
| test04_dqmom9 | 0.64 | 0.46 | 0.48 | tie |
| test05_nonlin1_r4 | 0.76 | 0.33 | 0.36 | tie |
| triangle | 2.55 | 2.37 | 0.62 | herbie |
| triangle1 | 1.13 | 0.82 | 0.45 | herbie |

## Worst-case error over the box (FPTaylor)

`ours` = the run's worst-case champion (best bound among proven programs; may differ from the average-bits column's program). Winner compares relative ulps when both sides have them, else absolute error.

| benchmark | reference | ours (best) | herbie | winner |
|---|--:|--:|--:|---|
| asymptote_c | 57.7 ulp | 2.1 ulp | 1.3 ulp | herbie |
| beta_a | 2.2e-16 abs | 1.7e-16 abs | 2.0e-16 abs | ours |
| beta_b | 2.5e-16 abs | 1.9e-16 abs | - | - |
| cancel_sqrt_2var | 180.0 ulp | 2.2 ulp | 180.0 ulp | ours |
| cancel_sqrt_shift3 | 311.9 ulp | 1.8 ulp | - | - |
| cancel_sqrt_sum | 179.4 ulp | 1.8 ulp | 179.4 ulp | ours |
| complex_square_real | 6.7e-16 abs | 6.1e-16 abs | 7.2e-16 abs | ours |
| conte_near_pole | 25062.1 ulp | 180.8 ulp | 125.4 ulp | herbie |
| conte_x_minus_sqrt | 1.4e-13 abs | 7.4e-17 abs | - | - |
| excel_x0 | 4.0e-16 abs | 126.8 ulp | 116.1 ulp | herbie |
| expand_square | 1.1e-15 abs | 2.8e-16 abs | 4.4e-16 abs | ours |
| fastmath_dist4 | 6.5e-12 abs | 4.5e-12 abs | 4.7e-12 abs | ours |
| floudas3 | 1.2e-14 abs | 7.5e-15 abs | 1.2e-14 abs | ours |
| kahan_p9 | 4.0e-12 abs | 4.0e-12 abs | 4.0e-12 abs | tie |
| martel_p6 | 5.8e-15 abs | 7.2e-16 abs | 1.3e-15 abs | ours |
| matrixdeterminant | 1.6e-12 abs | 1.5e-12 abs | - | - |
| matrixdeterminant2 | 1.6e-12 abs | 1.5e-12 abs | - | - |
| nmse_example_3_1 | 193.8 ulp | 1.6 ulp | 5.1 ulp | ours |
| nmse_example_3_6 | 8.4e-14 abs | 119.2 ulp | - | - |
| nmse_p42_negative | 2.3e-14 abs | 9.0e-15 abs | 3.4e-14 abs | ours |
| nmse_p42_positive | 2.3e-14 abs | 8.8e-15 abs | 3.4e-14 abs | ours |
| nmse_problem_3_2_1_negative | 4.6e-14 abs | 2.0 ulp | 4.6e-14 abs | tie |
| nmse_problem_3_2_1_positive | 4.6e-14 abs | 2.0 ulp | 4.6e-14 abs | tie |
| nmse_problem_3_3_1 | 1.9e-15 abs | 83474.9 ulp | 83474.9 ulp | tie |
| nmse_problem_3_3_3 | 4.1e-16 abs | 285193.5 ulp | - | - |
| nonlin2 | 29958.8 ulp | 1.3 ulp | 1.4 ulp | ours |
| pbrt_cone_z | 2.2e-16 abs | 1.7e-16 abs | 1.7e-16 abs | ours |
| rigidbody1 | 2.1e-13 abs | 1.2e-13 abs | 2.5e-13 abs | ours |
| rigidbody2 | 2.3e-11 abs | 1.9e-11 abs | - | - |
| sine | - | - | 4.2e-16 abs | - |
| som_setup_w | 6.0e-16 abs | 2.2e-16 abs | - | - |
| sum | 2.3 ulp | 0.8 ulp | - | - |
| test03_nonlin2 | 3.5e-16 abs | 3.4e-16 abs | - | - |
| test04_dqmom9 | - | 2.5e-06 abs | - | - |
| test05_nonlin1_r4 | 15597717409.8 ulp | 0.8 ulp | 601032842.2 ulp | ours |
| triangle | 17.6 ulp | 17.5 ulp | - | - |
| triangle1 | 5.2 ulp | 5.2 ulp | - | - |
