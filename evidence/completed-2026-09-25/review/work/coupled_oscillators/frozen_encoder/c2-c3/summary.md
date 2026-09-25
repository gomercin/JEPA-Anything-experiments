# Frozen encoder: rotation (exploratory)

Mean ± sample SD across three paired, exposed seeds; no model selection.

| Step | Optimizer | Model | Frame / sensor | One-step MSE | h16 MSE | Response h16 |
|---:|---|---|---|---:|---:|---:|
| 250 | Adam | opf | LATENT_NATIVE / RAW | 0.001225 ± 0.000155 | 0.1637 ± 0.0373 | 0.009756 ± 0.00528 |
| 250 | Adam | opf | LATENT_ROTATED / RAW | 0.0015 ± 0.000469 | 0.1988 ± 0.0685 | 0.008422 ± 0.00234 |
| 250 | Adam | standard | LATENT_NATIVE / RAW | 0.00142 ± 0.000201 | 0.2152 ± 0.053 | 0.01323 ± 0.00975 |
| 250 | Adam | standard | LATENT_ROTATED / RAW | 0.001699 ± 0.000248 | 0.3755 ± 0.177 | 0.01727 ± 0.0127 |
| 250 | SGD | opf | LATENT_NATIVE / RAW | 0.1492 ± 0.0175 | 0.5194 ± 0.0373 | 0.005504 ± 3.66e-05 |
| 250 | SGD | opf | LATENT_ROTATED / RAW | 0.1492 ± 0.0175 | 0.5194 ± 0.0373 | 0.005504 ± 3.66e-05 |
| 250 | SGD | standard | LATENT_NATIVE / RAW | 0.1405 ± 0.0168 | 0.5179 ± 0.0427 | 0.00547 ± 0.000102 |
| 250 | SGD | standard | LATENT_ROTATED / RAW | 0.1405 ± 0.0168 | 0.5179 ± 0.0427 | 0.00547 ± 0.000102 |
| 1000 | Adam | opf | LATENT_NATIVE / RAW | 0.0002422 ± 5.72e-05 | 0.01815 ± 0.0016 | 0.0009917 ± 0.000202 |
| 1000 | Adam | opf | LATENT_ROTATED / RAW | 0.000281 ± 8.78e-05 | 0.01996 ± 0.0035 | 0.001062 ± 0.000243 |
| 1000 | Adam | standard | LATENT_NATIVE / RAW | 0.0002637 ± 5.7e-05 | 0.01903 ± 0.00107 | 0.001027 ± 0.000118 |
| 1000 | Adam | standard | LATENT_ROTATED / RAW | 0.0002963 ± 7.63e-05 | 0.02217 ± 0.00193 | 0.001368 ± 0.000366 |
| 1000 | SGD | opf | LATENT_NATIVE / RAW | 0.01807 ± 0.0057 | 26.87 ± 40.5 | 10.46 ± 17.9 |
| 1000 | SGD | opf | LATENT_ROTATED / RAW | 0.01807 ± 0.0057 | 26.87 ± 40.5 | 10.46 ± 17.9 |
| 1000 | SGD | standard | LATENT_NATIVE / RAW | 0.01147 ± 0.00215 | 64.49 ± 101 | 19.06 ± 32.5 |
| 1000 | SGD | standard | LATENT_ROTATED / RAW | 0.01147 ± 0.00215 | 64.49 ± 101 | 19.06 ± 32.5 |
