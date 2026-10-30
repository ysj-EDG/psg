# SIREN five-view CORE_NO_CAISR20

This branch contains one route only:

```text
route = CORE_NO_CAISR20
source = ysj-EDG/physionet-challenge-2026-baseline
source branch = research/featurelab-extractors-v2
source commit = 626ec1e9f178ae5922db09fab79955a6182e0f67
```

This is **not** the early G5 baseline and **not** CORE with CAISR20.  The
public aggregate parity target is Macro AC-AUROC `0.6791222427959095`.

## Feature contract

The route uses the historical G1B five-view base and a FeatureLab CORE V5
extension:

| View | Composition | Width |
|---|---|---:|
| V1 | Wake EEG108 + Demo5 | 113 |
| V2 | REM EEG108 + REM Spatial16 + Demo5 | 129 |
| V3 | N2/N3/NREM EEG324 + N2/N3 Spatial20 + Demo5 | 349 |
| V4 | All-stage EEG108 + Demo5 | 113 |
| V5 | HRV22 + EMG48 + Resp28 + Demo5 + Coherence90 + StageEMG72 | 265 |

V5 is formed from the canonical 285-column CORE matrix and removes the exact
20 canonical CAISR20 Feature IDs.  It contains no CAISR20 features.  The
route therefore has 969 structural positions and 949 unique Feature IDs.

## Data chain

```text
raw/processed PSG
  -> timegrid_v2 extraction
  -> per-epoch physiological features
  -> night-level aggregation
  -> G1B / FeatureLab five-view matrices
  -> frozen Rank256 transform
  -> age-label sampling
  -> LR / HGB / LambdaRank
  -> 15-seed predictions
  -> within-holdout average-tie percentile ranks
  -> unweighted mean of 225 ranked columns
  -> official AC-AUROC / AUROC / AUPRC
```

The raw extraction implementation is in `per_epoch_features/` and is kept
separate from parity.  Fresh parity must use the frozen Large6600 Rank256
inputs for this historical result; replacing them with newly extracted EDF
values is a different preprocessing-reproduction experiment.

Training-site mapping is fixed: `I0002 -> S0001`, `I0006 -> S0001`, and
`S0001 -> I0006`.  Seeds are 42 through 56.  Every seed has five views and
three models (`lr`, `hgb`, `lambdarank`), yielding 225 ranked columns.

Sampling is `age_label_70pct_pcg64_v1`, stratified by age group and label
within each training site.  LambdaRank queries use `floor(age / 5)` and keep
only queries containing both labels.

LR uses l2, `C=1`, `lbfgs`, `max_iter=2000`, `tol=1e-4`. HGB uses learning
rate `0.05`, 250 iterations, 15 leaves, depth 3, minimum leaf 30, l2 1,
and no early stopping. LambdaRank uses 250 estimators, learning rate 0.05,
15 leaves, depth 3, minimum child 30, `reg_lambda=1`, and
`lambdarank_norm=True`. Seeds 43--56 change only estimator `random_state`.

For LR, missing values are filled by training-sample column medians; an
all-missing column uses neutral Rank256 `128/255`. HGB and LambdaRank retain
NaN values.

## Commands

Install the pinned runtime from `requirements.txt`, then import the raw
feature chain with `prepare_features.py`. To train and infer the new V5
branch from frozen Rank256 units:

```bash
python train_loso.py \
  --rank-root "$RANK_ROOT" \
  --metadata "$METADATA" \
  --g2-root "$G2_ROOT" \
  --manifest features/feature_manifest.csv \
  --caisr-manifest features/caisr20_manifest.csv \
  --output-root "$OUTPUT_ROOT" \
  --old-root "$OLD_ROOT"
```

The scorer entry point is `score.py`; `siren/ensemble.py` implements the
average-tie percentile rank mean and writes only aggregate CSVs.  The checked
aggregate outputs are `results/loso_metrics.csv` and `results/summary.csv`.

## Public artifacts policy

This branch intentionally excludes audit/report/spec files, tests, smoke and
verification scripts, logs, private predictions, record-level scores,
checkpoints, feature caches, Rank256 caches, EDF/NPZ data, and machine-local
absolute paths.
