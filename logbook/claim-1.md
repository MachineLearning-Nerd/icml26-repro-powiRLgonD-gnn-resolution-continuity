# Claim 1 — GCN cross-resolution error

**Exact live claim:** Standard GNNs are not continuous across graph resolutions: GCN's MAE increases from 63.6±1.3 kcal/mol in same-resolution QM7 regression to 138.1±2.4 under cross-resolution evaluation (Table 1).

**Outcome: toy (1-point-scale evidence only; not a QM7 verification).**

## Local clean-room direct fixture

The pinned source location and deviation from the unavailable QM7 protocol are recorded in `evidence/claim1_attempt1_source_locations.md`. We ran a one-layer renormalized-adjacency GCN plus a linear readout trained only at a 16-node fine ring resolution, then evaluated the exact same underlying synthetic signals at an 8-node quotient resolution. Five fixed local-CPU seeds use 300 training and 200 held-out graphs each.

| Metric | Mean over 5 seeds |
|---|---:|
| Fine same-resolution MAE | 0.4979 |
| Coarse cross-resolution MAE | 0.6301 |
| Permuted coarse destructive-control MAE | 0.6313 |

Cross-resolution MAE exceeded same-resolution MAE for all five seeds. The embedding shift and raw rows are retained in `outputs/claim1_resolution_gcn_toy/results.csv`; configuration and runtime metadata are in `config.json`; hashes verify with `(cd outputs/claim1_resolution_gcn_toy && sha256sum -c SHA256SUMS)`.

**Limitations:** this is a reduced synthetic fixture. It neither reproduces QM7, the claimed kcal/mol values, nor proves universal GCN discontinuity. It is explicitly labeled toy evidence.
