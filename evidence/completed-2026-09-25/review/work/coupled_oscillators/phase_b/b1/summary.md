# Phase B exploratory panel

Mean ± sample SD, three paired seeds; physical state MSE.

| Regime | Model | Frame | Split | h1 | h4 | h8 | h16 | Pulse response h16 |
|---|---|---|---|---:|---:|---:|---:|---:|
| LINEAR | linear | PHYSICAL | interpolation | 9.784e-31 ± 7.22e-31 | 1.478e-29 ± 1.11e-29 | 5.33e-29 ± 4.39e-29 | 1.608e-28 ± 1.7e-28 | 1.382e-30 ± 1.07e-30 |
| LINEAR | linear | PHYSICAL | amplitude_shift | 2.178e-30 ± 1.63e-30 | 3.291e-29 ± 2.51e-29 | 1.188e-28 ± 9.91e-29 | 3.592e-28 ± 3.84e-28 | 1.826e-30 ± 1.12e-30 |
| LINEAR | opf | LATENT_NATIVE | interpolation | 0.0002422 ± 5.72e-05 | 0.002758 ± 0.000426 | 0.007612 ± 0.000884 | 0.01815 ± 0.0016 | 0.0009917 ± 0.000202 |
| LINEAR | opf | LATENT_NATIVE | amplitude_shift | 0.002706 ± 0.000494 | 0.02416 ± 0.00329 | 0.05546 ± 0.00703 | 0.1032 ± 0.0105 | 0.001801 ± 0.000425 |
| LINEAR | opf | LATENT_ROTATED | interpolation | 0.000281 ± 8.78e-05 | 0.003109 ± 0.000839 | 0.008407 ± 0.00217 | 0.01996 ± 0.0035 | 0.001062 ± 0.000243 |
| LINEAR | opf | LATENT_ROTATED | amplitude_shift | 0.003135 ± 0.000679 | 0.02692 ± 0.00506 | 0.06056 ± 0.0126 | 0.1115 ± 0.0229 | 0.001841 ± 0.000396 |
| LINEAR | opf_fixed | LATENT_NATIVE | interpolation | 0.0002637 ± 5.7e-05 | 0.003002 ± 0.000441 | 0.008208 ± 0.00094 | 0.01903 ± 0.00107 | 0.001027 ± 0.000118 |
| LINEAR | opf_fixed | LATENT_NATIVE | amplitude_shift | 0.002912 ± 0.000481 | 0.02585 ± 0.00317 | 0.05861 ± 0.00731 | 0.1058 ± 0.0123 | 0.001838 ± 0.000466 |
| LINEAR | opf_fixed | LATENT_ROTATED | interpolation | 0.0003058 ± 0.00011 | 0.003364 ± 0.0011 | 0.009026 ± 0.00296 | 0.02089 ± 0.00492 | 0.001144 ± 0.000281 |
| LINEAR | opf_fixed | LATENT_ROTATED | amplitude_shift | 0.003288 ± 0.000753 | 0.02819 ± 0.00595 | 0.06315 ± 0.0153 | 0.1146 ± 0.0265 | 0.001969 ± 0.000547 |
| LINEAR | polynomial_cubic | PHYSICAL | interpolation | 6.337e-30 ± 2.79e-30 | 9.409e-29 ± 4.04e-29 | 3.02e-28 ± 1.14e-28 | 6.398e-28 ± 1.47e-28 | 1.91e-29 ± 9.45e-30 |
| LINEAR | polynomial_cubic | PHYSICAL | amplitude_shift | 3.696e-29 ± 2.3e-29 | 5.534e-28 ± 3.49e-28 | 1.775e-27 ± 1.05e-27 | 3.685e-27 ± 1.47e-27 | 5.714e-29 ± 2.87e-29 |
| LINEAR | standard | LATENT_NATIVE | interpolation | 0.0002637 ± 5.7e-05 | 0.003002 ± 0.000441 | 0.008208 ± 0.00094 | 0.01903 ± 0.00107 | 0.001027 ± 0.000118 |
| LINEAR | standard | LATENT_NATIVE | amplitude_shift | 0.002912 ± 0.000481 | 0.02585 ± 0.00317 | 0.05861 ± 0.00731 | 0.1058 ± 0.0123 | 0.001838 ± 0.000466 |
| LINEAR | standard | LATENT_ROTATED | interpolation | 0.0002963 ± 7.63e-05 | 0.003353 ± 0.000599 | 0.009245 ± 0.00115 | 0.02217 ± 0.00193 | 0.001368 ± 0.000366 |
| LINEAR | standard | LATENT_ROTATED | amplitude_shift | 0.003315 ± 0.000595 | 0.02792 ± 0.00263 | 0.06221 ± 0.00459 | 0.1129 ± 0.00867 | 0.002314 ± 0.000542 |
| MODERATE_NONLINEAR | linear | PHYSICAL | interpolation | 0.0002209 ± 5.93e-05 | 0.002744 ± 0.000714 | 0.006542 ± 0.00145 | 0.01219 ± 0.00249 | 0.0008774 ± 0.000459 |
| MODERATE_NONLINEAR | linear | PHYSICAL | amplitude_shift | 0.003612 ± 0.000889 | 0.0446 ± 0.0105 | 0.1064 ± 0.0211 | 0.2161 ± 0.0432 | 0.002596 ± 0.000317 |
| MODERATE_NONLINEAR | opf | LATENT_NATIVE | interpolation | 0.0005152 ± 9.91e-05 | 0.006138 ± 0.00108 | 0.01648 ± 0.00277 | 0.03522 ± 0.00362 | 0.001679 ± 0.000697 |
| MODERATE_NONLINEAR | opf | LATENT_NATIVE | amplitude_shift | 0.007418 ± 0.00148 | 0.07644 ± 0.017 | 0.1799 ± 0.0375 | 0.3208 ± 0.0422 | 0.004266 ± 0.000973 |
| MODERATE_NONLINEAR | opf | LATENT_ROTATED | interpolation | 0.0005493 ± 0.000132 | 0.006443 ± 0.00164 | 0.01722 ± 0.0049 | 0.03739 ± 0.00935 | 0.001787 ± 0.00055 |
| MODERATE_NONLINEAR | opf | LATENT_ROTATED | amplitude_shift | 0.007826 ± 0.00161 | 0.07967 ± 0.0195 | 0.1874 ± 0.0457 | 0.3313 ± 0.0475 | 0.004336 ± 0.00131 |
| MODERATE_NONLINEAR | opf_fixed | LATENT_NATIVE | interpolation | 0.000538 ± 0.000102 | 0.006437 ± 0.00115 | 0.01729 ± 0.00308 | 0.03659 ± 0.00392 | 0.001729 ± 0.000676 |
| MODERATE_NONLINEAR | opf_fixed | LATENT_NATIVE | amplitude_shift | 0.00767 ± 0.00142 | 0.07879 ± 0.0175 | 0.1846 ± 0.0417 | 0.3243 ± 0.0515 | 0.004559 ± 0.00134 |
| MODERATE_NONLINEAR | opf_fixed | LATENT_ROTATED | interpolation | 0.0005828 ± 0.000146 | 0.006825 ± 0.00186 | 0.01815 ± 0.00559 | 0.03839 ± 0.00979 | 0.001849 ± 0.000546 |
| MODERATE_NONLINEAR | opf_fixed | LATENT_ROTATED | amplitude_shift | 0.008083 ± 0.00169 | 0.0821 ± 0.0212 | 0.1914 ± 0.0524 | 0.3311 ± 0.0619 | 0.004594 ± 0.00171 |
| MODERATE_NONLINEAR | polynomial_cubic | PHYSICAL | interpolation | 1.667e-09 ± 3.2e-10 | 1.864e-08 ± 2.42e-09 | 4.001e-08 ± 6.07e-09 | 7.401e-08 ± 1.49e-08 | 6.228e-09 ± 5.73e-09 |
| MODERATE_NONLINEAR | polynomial_cubic | PHYSICAL | amplitude_shift | 2.556e-07 ± 8.84e-08 | 2.068e-06 ± 5.13e-07 | 2.833e-06 ± 5.3e-07 | 9.409e-06 ± 2.4e-06 | 5.908e-08 ± 3.12e-08 |
| MODERATE_NONLINEAR | standard | LATENT_NATIVE | interpolation | 0.000538 ± 0.000102 | 0.006437 ± 0.00115 | 0.01729 ± 0.00308 | 0.03659 ± 0.00392 | 0.001729 ± 0.000676 |
| MODERATE_NONLINEAR | standard | LATENT_NATIVE | amplitude_shift | 0.00767 ± 0.00142 | 0.07879 ± 0.0175 | 0.1846 ± 0.0417 | 0.3243 ± 0.0515 | 0.004559 ± 0.00134 |
| MODERATE_NONLINEAR | standard | LATENT_ROTATED | interpolation | 0.0005617 ± 0.000113 | 0.006585 ± 0.00102 | 0.01777 ± 0.0021 | 0.03764 ± 0.00201 | 0.001966 ± 0.000553 |
| MODERATE_NONLINEAR | standard | LATENT_ROTATED | amplitude_shift | 0.008076 ± 0.00114 | 0.08083 ± 0.0134 | 0.191 ± 0.0307 | 0.338 ± 0.0357 | 0.004324 ± 0.000829 |
| WEAK_NONLINEAR | linear | PHYSICAL | interpolation | 1.219e-05 ± 4.88e-06 | 0.0001627 ± 6.7e-05 | 0.0004334 ± 0.000178 | 0.0008264 ± 0.000295 | 7.882e-05 ± 2.26e-05 |
| WEAK_NONLINEAR | linear | PHYSICAL | amplitude_shift | 0.0001959 ± 6.71e-05 | 0.00259 ± 0.0009 | 0.006988 ± 0.00234 | 0.01439 ± 0.00433 | 0.0004042 ± 0.000151 |
| WEAK_NONLINEAR | opf | LATENT_NATIVE | interpolation | 0.0002642 ± 5.63e-05 | 0.003048 ± 0.000437 | 0.008458 ± 0.000894 | 0.0198 ± 0.00126 | 0.0009499 ± 0.000231 |
| WEAK_NONLINEAR | opf | LATENT_NATIVE | amplitude_shift | 0.0032 ± 0.000567 | 0.02939 ± 0.00462 | 0.06874 ± 0.0109 | 0.1252 ± 0.0127 | 0.001951 ± 0.000573 |
| WEAK_NONLINEAR | opf | LATENT_ROTATED | interpolation | 0.0003019 ± 8.22e-05 | 0.003386 ± 0.000826 | 0.009268 ± 0.00228 | 0.02163 ± 0.00423 | 0.00103 ± 0.000242 |
| WEAK_NONLINEAR | opf | LATENT_ROTATED | amplitude_shift | 0.003557 ± 0.000587 | 0.03187 ± 0.00524 | 0.07371 ± 0.0151 | 0.1341 ± 0.0279 | 0.002045 ± 0.000633 |
| WEAK_NONLINEAR | opf_fixed | LATENT_NATIVE | interpolation | 0.0002877 ± 5.34e-05 | 0.003325 ± 0.000414 | 0.009166 ± 0.000852 | 0.02092 ± 0.000623 | 0.0009835 ± 0.000166 |
| WEAK_NONLINEAR | opf_fixed | LATENT_NATIVE | amplitude_shift | 0.003416 ± 0.000512 | 0.03123 ± 0.00427 | 0.07216 ± 0.0114 | 0.1271 ± 0.0169 | 0.002013 ± 0.000558 |
| WEAK_NONLINEAR | opf_fixed | LATENT_ROTATED | interpolation | 0.0003277 ± 0.000103 | 0.003656 ± 0.00106 | 0.009949 ± 0.00296 | 0.0227 ± 0.00506 | 0.001115 ± 0.000264 |
| WEAK_NONLINEAR | opf_fixed | LATENT_ROTATED | amplitude_shift | 0.003703 ± 0.000692 | 0.03325 ± 0.00636 | 0.07678 ± 0.0186 | 0.1363 ± 0.0319 | 0.002281 ± 0.000788 |
| WEAK_NONLINEAR | polynomial_cubic | PHYSICAL | interpolation | 5.02e-12 ± 2.22e-12 | 6.61e-11 ± 2.89e-11 | 1.742e-10 ± 7.96e-11 | 3.406e-10 ± 1.95e-10 | 2.235e-11 ± 1.26e-11 |
| WEAK_NONLINEAR | polynomial_cubic | PHYSICAL | amplitude_shift | 5.729e-10 ± 3.86e-10 | 6.498e-09 ± 4.74e-09 | 1.313e-08 ± 1.01e-08 | 2.108e-08 ± 1.5e-08 | 7.66e-10 ± 6.52e-10 |
| WEAK_NONLINEAR | standard | LATENT_NATIVE | interpolation | 0.0002877 ± 5.34e-05 | 0.003325 ± 0.000414 | 0.009166 ± 0.000852 | 0.02092 ± 0.000623 | 0.0009835 ± 0.000166 |
| WEAK_NONLINEAR | standard | LATENT_NATIVE | amplitude_shift | 0.003416 ± 0.000512 | 0.03123 ± 0.00427 | 0.07216 ± 0.0114 | 0.1271 ± 0.0169 | 0.002013 ± 0.000558 |
| WEAK_NONLINEAR | standard | LATENT_ROTATED | interpolation | 0.0003164 ± 7.66e-05 | 0.003617 ± 0.000608 | 0.01005 ± 0.00121 | 0.02371 ± 0.00157 | 0.001346 ± 0.000362 |
| WEAK_NONLINEAR | standard | LATENT_ROTATED | amplitude_shift | 0.003765 ± 0.00057 | 0.03285 ± 0.00298 | 0.07569 ± 0.00731 | 0.137 ± 0.0169 | 0.002432 ± 0.000453 |
