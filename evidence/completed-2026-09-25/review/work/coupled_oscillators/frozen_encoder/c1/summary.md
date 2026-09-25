# Frozen encoder: physical (exploratory)

Mean ± sample SD across three paired, exposed seeds; no model selection.

| Step | Optimizer | Model | Frame / sensor | One-step MSE | h16 MSE | Response h16 |
|---:|---|---|---|---:|---:|---:|
| 250 | Adam | opf | LATENT_NATIVE / MIXED | 0.001225 ± 0.000155 | 0.1637 ± 0.0373 | 0.009756 ± 0.00528 |
| 250 | Adam | opf | LATENT_NATIVE / RAW | 0.001225 ± 0.000155 | 0.1637 ± 0.0373 | 0.009756 ± 0.00528 |
| 250 | Adam | standard | LATENT_NATIVE / MIXED | 0.00142 ± 0.000201 | 0.2152 ± 0.053 | 0.01323 ± 0.00975 |
| 250 | Adam | standard | LATENT_NATIVE / RAW | 0.00142 ± 0.000201 | 0.2152 ± 0.053 | 0.01323 ± 0.00975 |
| 1000 | Adam | opf | LATENT_NATIVE / MIXED | 0.0002422 ± 5.72e-05 | 0.01815 ± 0.0016 | 0.0009917 ± 0.000202 |
| 1000 | Adam | opf | LATENT_NATIVE / RAW | 0.0002422 ± 5.72e-05 | 0.01815 ± 0.0016 | 0.0009917 ± 0.000202 |
| 1000 | Adam | standard | LATENT_NATIVE / MIXED | 0.0002637 ± 5.7e-05 | 0.01903 ± 0.00107 | 0.001027 ± 0.000118 |
| 1000 | Adam | standard | LATENT_NATIVE / RAW | 0.0002637 ± 5.7e-05 | 0.01903 ± 0.00107 | 0.001027 ± 0.000118 |
