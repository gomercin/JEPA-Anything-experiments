# Frozen encoder: sgd_duration (exploratory)

Mean ± sample SD across three paired, exposed seeds; no model selection.

| Step | Optimizer | Model | Frame / sensor | One-step MSE | h16 MSE | Response h16 |
|---:|---|---|---|---:|---:|---:|
| 250 | SGD | opf | LATENT_NATIVE / RAW | 0.1492 ± 0.0175 | 0.5194 ± 0.0373 | 0.005504 ± 3.66e-05 |
| 250 | SGD | standard | LATENT_NATIVE / RAW | 0.1405 ± 0.0168 | 0.5179 ± 0.0427 | 0.00547 ± 0.000102 |
| 1000 | SGD | opf | LATENT_NATIVE / RAW | 0.01807 ± 0.0057 | 26.87 ± 40.5 | 10.46 ± 17.9 |
| 1000 | SGD | standard | LATENT_NATIVE / RAW | 0.01147 ± 0.00215 | 64.49 ± 101 | 19.06 ± 32.5 |
| 2000 | SGD | opf | LATENT_NATIVE / RAW | 0.006457 ± 0.00103 | 13.36 ± 18.7 | 2.748 ± 4.59 |
| 2000 | SGD | standard | LATENT_NATIVE / RAW | 0.005718 ± 0.000852 | 10.73 ± 14.6 | 2.034 ± 3.37 |
| 5000 | SGD | opf | LATENT_NATIVE / RAW | 0.002135 ± 0.000453 | 0.5034 ± 0.165 | 0.02215 ± 0.0119 |
| 5000 | SGD | standard | LATENT_NATIVE / RAW | 0.002025 ± 0.000449 | 0.4536 ± 0.14 | 0.02006 ± 0.0132 |
