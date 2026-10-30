# DEDUP_FUSION

This branch is independently based on `origin/main` and implements only the
fixed direct fusion route `DEDUP_FUSION` from historical source commit
`626ec1e9f178ae5922db09fab79955a6182e0f67`.

Parent A is `CORE_NO_CAISR20`; Parent B is
`CAISR20_HISTORICAL_SEMANTICS_CORE_PROTOCOL`. The CLI validates shared record
indices, site, labels, ages, ordering, counts, and exact parent metrics before
computing `0.5 * CORE_NO_CAISR20_FINAL + 0.5 * CAISR20_FINAL`. The result is
never reranked, calibrated, or weight-searched.

```bash
python run_fusion.py \
  --manifest "$DATA2_MANIFEST" \
  --core-root "$CORE_FINAL_ROOT" \
  --caisr-root "$CAISR_REPLAY_ROOT" \
  --output-root "$PRIVATE_FUSION_ROOT" \
  --output-dir results
```

Only aggregate results are public; parent prediction units and fusion outputs
remain external private inputs/outputs.
