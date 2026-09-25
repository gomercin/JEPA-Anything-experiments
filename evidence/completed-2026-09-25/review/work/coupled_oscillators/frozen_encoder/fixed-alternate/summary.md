# Frozen encoder: fixed_alternate (exploratory)

Mean ± sample SD across three paired, exposed seeds; no model selection.

| Step | Optimizer | Model | Frame / sensor | One-step MSE | h16 MSE | Response h16 |
|---:|---|---|---|---:|---:|---:|
| 250 | Adam | opf_fixed | LATENT_NATIVE / RAW | 0.00142 ± 0.000201 | 0.2152 ± 0.053 | 0.01323 ± 0.00975 |
| 250 | Adam | opf_fixed | LATENT_ROTATED / RAW | 0.001502 ± 0.000156 | 0.2364 ± 0.0796 | 0.01377 ± 0.00557 |
| 1000 | Adam | opf_fixed | LATENT_NATIVE / RAW | 0.0002637 ± 5.7e-05 | 0.01903 ± 0.00107 | 0.001027 ± 0.000118 |
| 1000 | Adam | opf_fixed | LATENT_ROTATED / RAW | 0.0002849 ± 9.78e-05 | 0.0195 ± 0.00262 | 0.001055 ± 0.000254 |
