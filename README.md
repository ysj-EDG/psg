# `exp/matcha-m2`: historical Matcha M2

## Route

`MATCHA_M2_HISTORICAL_INNER1TO4` is the primary route. It recovers the
historical three-member M2 Raw sequence ensemble and its inner-1-to-4 epoch
selection followed by outer refitting. The primary result is the verified
`Macro AC-AUROC = 0.7185027051240563`.

The historical source is `ysj-EDG/physionet-challenge-2026-baseline` history
commit `c71c731ecc7d32b5ef54371735a1b07bd2d7cc43`, with the packaged E1
checkpoint identified by SHA256
`e9f135c973c0540970f34acbde89096c028eb3a95a0a601d85f21e8f83d36ac0`.
The M2 V5 date-residual source is `BaoXinQi/psg4ci` commit
`f05c3b12b2a2fcfdc7c1518521c0610d11c11021`.

## Data and model

The fixed Large 6600 cohort contains I0002 (319 records), I0006 (1,142), and
S0001 (5,139). Inputs are frozen historical 192-dimensional E1 embeddings,
ten windows per local block, and a full-night Transformer. The primary model
uses the native mean of raw logits from seeds `20260806`, `20260807`, and
`20260808`; the inner protocol selects among epochs 1--4 and refits the chosen
epoch on each outer-training fold. Pairwise training uses age gap 2 years,
same-site pairs, and weight 0.15.

Three LOSO folds hold out I0002, I0006, and S0001. AC-AUROC uses the existing
age-conditioned Challenge scorer semantics. `results/loso_metrics.csv` contains
site metrics; `results/summary.csv` contains the small-site and macro summaries.

## Primary verified result

| Held-out site | Records | Positives | AC-AUROC | AUROC | AUPRC |
| --- | ---: | ---: | ---: | ---: | ---: |
| I0002 | 319 | 52 | 0.7481442205726405 | 0.7851483722270238 | 0.5209606504394242 |
| I0006 | 1,142 | 112 | 0.7393224315186642 | 0.7901005547850207 | 0.2945779196802375 |
| S0001 | 5,139 | 334 | 0.6680414632808643 | 0.7439418769121485 | 0.1867674115690459 |

Small-site AC-AUROC is `0.7437333260456523`; macro AC-AUROC is
`0.7185027051240563`; macro AUROC is `0.7730636013080644`; macro AUPRC is
`0.3341019938962359`.

## Run

Set these paths to local copies of the prepared manifest and historical E1
embedding directory; do not put their contents in GitHub:

```bash
export PSG_DATA_ROOT=/path/to/prepared
export PSG_M2_EMBEDDING_ROOT=/path/to/e1_historical_decoded

python run_loso.py \
  --manifest "$PSG_DATA_ROOT/training_manifest.parquet" \
  --embedding-manifest "$PSG_M2_EMBEDDING_ROOT/embedding_manifest.parquet" \
  --private-output private/m2_fresh \
  --results-dir results
```

The command fails closed if the cohort, E1 hash, or record alignment differs.
It exports public CSVs only after all five primary AC targets match exactly.

## Secondary M2 V5 date residual

The historical M2 V5 CreationTime residual is retained as a separate stage;
it adds the fold-local fitted date residual directly to the primary M2 ensemble
logit. It does not change the identity of the primary route. Its aggregate
outputs are under `results/residual_date/` and its private record-level outputs
remain under `private/`.

```bash
python run_residual.py \
  --manifest "$PSG_DATA_ROOT/training_manifest.parquet" \
  --primary-private-output private/m2_fresh \
  --private-output private/m2_v5_date_residual \
  --results-dir results/residual_date
```

The secondary macro AC-AUROC is `0.7725904425320311`; it is reported for the
residual ablation only.

## Environment and limitations

Use Python 3.10 with the historical PyTorch CUDA 12.1 environment described by
`requirements.txt`. The public branch contains no EDF files, caches, model
checkpoints, patient-level predictions, private data, or absolute data paths.
The packaged result has been rerun on the fixed Large 6600 inputs with the
three historical seeds; the source E1 embeddings remain frozen.
