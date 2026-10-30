# CAISR20 historical-semantics core protocol

This branch implements only `CAISR20_HISTORICAL_SEMANTICS_CORE_PROTOCOL` from
`ysj-EDG/physionet-challenge-2026-baseline`, source branch
`research/featurelab-extractors-v2`, commit
`626ec1e9f178ae5922db09fab79955a6182e0f67`.

It uses the fixed 20 historical CAISR20 `x_static` columns, historical
`typed_v1` preprocessing followed by selected-row median fill and mean/std
scaling, balanced elastic-net SAGA logistic regression, decision-function
scores, and average-tie rank means over seeds 42--56. The holdout mapping is
`I0002 -> S0001`, `I0006 -> S0001`, `S0001 -> I0006`.

Run fresh reproduction with external inputs:

```bash
python run_caisr20.py \
  --manifest "$DATA2_MANIFEST" \
  --summary-cache "$SUMMARY_CACHE" \
  --rules "$TYPED_RULES" \
  --pool-root "$POOL_ROOT" \
  --sample-root "$SAMPLE_ROOT" \
  --output-root "$PRIVATE_OUTPUT_ROOT" \
  --output-dir results
```

The public result files contain only aggregate site metrics and summaries.
Private route units, record-level scores, patient identifiers, caches,
checkpoints, audit files, and machine-local paths are intentionally excluded.
