# Coupled oscillators: Phase A

Mean ± sample SD over three paired seeds.
MSE is in observable units, averaged over states and trajectory origins.

| Condition | Model | h=1 | h=4 | h=8 | h=16 | Pulse h=16 | Response h=16 |
|---|---|---:|---:|---:|---:|---:|---:|
| MIXED | linear_transition | 2.13e-30 ± 1.5e-30 | 3.26e-29 ± 2.2e-29 | 1.16e-28 ± 7.9e-29 | 3.26e-28 ± 2.3e-28 | 3.25e-28 ± 2.3e-28 | 2.55e-30 ± 1.9e-30 |
| MIXED | opf | 0.0136 ± 0.0085 | 0.0767 ± 0.025 | 0.18 ± 0.054 | 0.876 ± 0.55 | 1.53 ± 1.3 | 0.028 ± 0.019 |
| MIXED | standard_jepa | 0.0227 ± 0.016 | 0.0913 ± 0.031 | 0.204 ± 0.07 | 1.17 ± 0.93 | 2.28 ± 2.4 | 0.0291 ± 0.018 |
| MIXED | unconstrained_multihead | 0.0227 ± 0.016 | 0.0913 ± 0.031 | 0.204 ± 0.07 | 1.17 ± 0.93 | 2.28 ± 2.4 | 0.0291 ± 0.018 |
| RAW | linear_transition | 9.78e-31 ± 7.2e-31 | 1.48e-29 ± 1.1e-29 | 5.33e-29 ± 4.4e-29 | 1.61e-28 ± 1.7e-28 | 1.47e-28 ± 1.5e-28 | 1.38e-30 ± 1.1e-30 |
| RAW | opf | 0.0418 ± 0.057 | 0.0997 ± 0.07 | 0.225 ± 0.1 | 0.968 ± 0.46 | 1.66 ± 1.3 | 0.0473 ± 0.0078 |
| RAW | standard_jepa | 0.0462 ± 0.058 | 0.106 ± 0.063 | 0.245 ± 0.072 | 1.09 ± 0.18 | 1.49 ± 0.34 | 0.0721 ± 0.031 |
| RAW | unconstrained_multihead | 0.0462 ± 0.058 | 0.106 ± 0.063 | 0.245 ± 0.072 | 1.09 ± 0.18 | 1.49 ± 0.34 | 0.0721 ± 0.031 |

OPF audit outcomes (failures are retained diagnostics, not experiment failures):

| Condition | Seed | Basis pass | Condition number | Round-trip NMSE | Transpose NMSE | Max cross correlation | Inactive coordinates | Mode overlap |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| RAW | 11 | False | 1.0081310576408555 | 3.51e-31 | 1.36e-05 | 0.761 | 0 | 0.409 |
| MIXED | 11 | False | 1.006876468752532 | 3.9e-31 | 1.75e-05 | 0.647 | 0 | 0.438 |
| RAW | 22 | False | 1.0043210370651148 | 3.6e-31 | 9.87e-06 | 0.599 | 0 | 0.425 |
| MIXED | 22 | False | 1.0056826526773768 | 4.17e-31 | 1.51e-05 | 0.617 | 0 | 0.417 |
| RAW | 33 | False | 1.0033060366803241 | 6.52e-31 | 3.78e-06 | 0.746 | 0 | 0.379 |
| MIXED | 33 | False | 1.0058154512217212 | 1.61e-30 | 4.75e-06 | 0.69 | 0 | 0.435 |

- Instrument/integration check; quick training is not a converged benchmark.
- A useful representation != a privileged physical ontology.
- Orthogonal learned factors != statistically independent causes.
- Good prediction after an intervention != causal discovery.
- MIXED coordinates != changed dynamics.
- PARTIAL observation != mere coordinate change; PARTIAL is not implemented.
- Lower training loss is not evidence of architectural advantage.
- Phase B nonlinear extension is intentionally deferred and not executed.

See results.json for every seed, parameter counts, protocol, matrices and diagnostics.
