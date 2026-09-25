# Frozen encoder: rotation_adam (exploratory)

Mean ± sample SD across three paired, exposed seeds; no model selection.

| Step | Optimizer | Model | Frame / sensor | One-step MSE | h16 MSE | Response h16 |
|---:|---|---|---|---:|---:|---:|
| 250 | Adam | opf | LATENT_NATIVE / RAW | 0.001225 ± 0.000155 | 0.1637 ± 0.0373 | 0.009756 ± 0.00528 |
| 250 | Adam | opf | LATENT_ROTATED / RAW | 0.001177 ± 0.000175 | 0.1383 ± 0.0165 | 0.007507 ± 0.000556 |
| 250 | Adam | standard | LATENT_NATIVE / RAW | 0.00142 ± 0.000201 | 0.2152 ± 0.053 | 0.01323 ± 0.00975 |
| 250 | Adam | standard | LATENT_ROTATED / RAW | 0.001862 ± 0.000506 | 0.436 ± 0.15 | 0.03054 ± 0.0201 |
| 1000 | Adam | opf | LATENT_NATIVE / RAW | 0.0002422 ± 5.72e-05 | 0.01815 ± 0.0016 | 0.0009917 ± 0.000202 |
| 1000 | Adam | opf | LATENT_ROTATED / RAW | 0.0002615 ± 8.81e-05 | 0.01839 ± 0.00219 | 0.0009608 ± 0.000227 |
| 1000 | Adam | standard | LATENT_NATIVE / RAW | 0.0002637 ± 5.7e-05 | 0.01903 ± 0.00107 | 0.001027 ± 0.000118 |
| 1000 | Adam | standard | LATENT_ROTATED / RAW | 0.0003036 ± 8.65e-05 | 0.02343 ± 0.00171 | 0.001282 ± 5.29e-05 |
