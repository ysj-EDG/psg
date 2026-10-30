# AB_EQUAL_50_50

This route is the final historical fusion experiment in the PhysioNet Challenge 2026 follow-up.

The two parents are independently gated: the DEDUP_FUSION final score and the MATCHA_M2_HISTORICAL_INNER1TO4 final score. Matcha uses the arithmetic mean of the three native raw-logit seeds `20260806`, `20260807`, and `20260808`. For each holdout, each parent is converted to average-tie rank and the two rank vectors are averaged 50/50. There is no final reranking.

Run with private inputs supplied explicitly:

```bash
PYTHONPATH=. python run_matcha_fusion.py \
  --manifest /path/to/loso_manifest.csv \
  --dedup-root /path/to/dedup_fusion_output \
  --matcha-root /path/to/matcha_m2_fused \
  --output-root /path/to/private_output \
  --output-dir results
```

Only aggregate LOSO metrics are committed. Private predictions and record-level inputs remain outside the repository.
