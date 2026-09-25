# Frozen encoder: gram (exploratory)

Mean ± sample SD across three paired, exposed seeds; no model selection.

| Step | Optimizer | Model | Frame / sensor | One-step MSE | h16 MSE | Response h16 |
|---:|---|---|---|---:|---:|---:|
| 250 | Adam | opf_fixed | LATENT_NATIVE / RAW | 0.00142 ± 0.000201 | 0.2152 ± 0.053 | 0.01323 ± 0.00975 |
| 250 | Adam | opf_fixed | LATENT_ROTATED / RAW | 0.001841 ± 0.000527 | 0.2859 ± 0.0611 | 0.01647 ± 0.00814 |
| 250 | Adam | opf_no_gram | LATENT_NATIVE / RAW | 0.001693 ± 0.000251 | 0.3544 ± 0.161 | 0.01922 ± 0.00972 |
| 250 | Adam | opf_no_gram | LATENT_ROTATED / RAW | 0.002288 ± 0.000861 | 0.5491 ± 0.251 | 0.02979 ± 0.0186 |
| 1000 | Adam | opf_fixed | LATENT_NATIVE / RAW | 0.0002637 ± 5.7e-05 | 0.01903 ± 0.00107 | 0.001027 ± 0.000118 |
| 1000 | Adam | opf_fixed | LATENT_ROTATED / RAW | 0.0003058 ± 0.00011 | 0.02089 ± 0.00492 | 0.001144 ± 0.000281 |
| 1000 | Adam | opf_no_gram | LATENT_NATIVE / RAW | 0.0002591 ± 7.41e-05 | 0.01942 ± 0.00307 | 0.001279 ± 0.000287 |
| 1000 | Adam | opf_no_gram | LATENT_ROTATED / RAW | 0.0002711 ± 8.64e-05 | 0.01979 ± 0.00445 | 0.001322 ± 0.000179 |
