# PSG: PhysioNet Challenge 2026 experiments

Independent technical routes for reproducing historical Large 6600 results.
Main contains only the project index and publication exclusions. Each experiment
branch starts independently from main and provides its own code, environment,
run instructions, and verified aggregate results.

## Shared LOSO evaluation protocol

Use the fixed 6,600 records: I0002 (319), I0006 (1,142), S0001 (5,139).
Run three leave-one-site-out folds, holding out each site once. Preserve the
existing validated Challenge age-conditioned AUROC scorer semantics. Report
AC-AUROC, AUROC, and AUPRC separately for each held-out site.
Small-site AC-AUROC is the arithmetic mean over I0002 and I0006; macro metrics
are unweighted arithmetic means over all three sites.

Verified routes publish `results/loso_metrics.csv` (including record, positive,
and eligible age-pair counts) and `results/summary.csv`. Patient-level data and
predictions remain outside the public repository.

## Experiment branch index

Branches are prepared and verified sequentially; this index does not imply
that every branch has already been published.

| Branch | Primary route |
| --- | --- |
| `exp/matcha-m2` | `MATCHA_M2_HISTORICAL_INNER1TO4` |
| `exp/siren-fiveview` | `CORE_NO_CAISR20` |
| `exp/caisr20-baseline` | `CAISR20_HISTORICAL_SEMANTICS_CORE_PROTOCOL` |
| `exp/fusion-siren-caisr20` | `DEDUP_FUSION` |
| `exp/fusion-siren-caisr20-matcha` | `AB_EQUAL_50_50` |
| `exp/feature-matcha-hierarchical` | Historical feature-extractor + Matcha hierarchical experiment; identity pending verification |
