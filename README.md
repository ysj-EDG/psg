# `exp/matcha-m2`: historical Matcha M2

This branch contains the complete, data-free M2 route:

`raw PSG/EDF -> channel mapping -> filtering/resampling -> quality masks ->
robust normalization -> canonical HDF5 -> E1 training/freeze/export -> M2
hierarchical LOSO -> three-seed mean-logit scoring -> optional V5 date residual`.

Only code, the historical channel table, and aggregate result CSVs are
committed. Raw EDF files, prepared caches, embeddings, checkpoints, and
record-level predictions stay outside Git.

## Provenance and route identity

The M2 inference/model source is
`ysj-EDG/physionet-challenge-2026-baseline` commit
`c71c731ecc7d32b5ef54371735a1b07bd2d7cc43`. Its recovered source includes
`channel_mapper.py`, `channel_table.csv`, the canonical preprocessing
primitives, `raw_sequence_runtime.py`, and the sequence model.

The historical frozen E1 artifact is identified by SHA256
`e9f135c973c0540970f34acbde89096c028eb3a95a0a601d85f21e8f83d36ac0`.
That binary is intentionally not committed. A source-level E1 retraining
entry is included as `train_e1.py` plus
`train_domain_robust_encoder_full.py`, recovered from the later public
Matcha full-training commit `7f2eae2fe4f170c87a21943b24ea4e217eabf67f`.
This provenance is deliberately distinct: a newly trained E1 checkpoint is
not claimed to have the historical checkpoint SHA unless it is independently
hashed and matches.

The raw cache contract is the historical M2 contract: 30-second
epoch-major HDF5 tensors, modality-specific sampling rates
(EEG/EOG/ECG/EMG 128 Hz, respiration 32 Hz, SpO2 1 Hz), float16 normalized
signals, per-channel 5-second hard-valid masks, modality masks, normalization
metadata, and aligned human/CAISR annotation arrays. E1 consumes only eligible
30-second windows with a neural modality and at least one additional valid
modality.

## 1. Prepare raw EDF

The formal entry is `prepare_data.py`; it inventories raw EDFs, derives the
historical canonical source plan, aligns optional human/CAISR EDF annotations,
and runs the historical cache builder. It writes
`training_manifest.parquet` and `canonical/records/<record_id>.h5` below
the private output root.

The record metadata file is the official training metadata needed by E1
(`record_id`, `patient_id`, `SiteID`, `Age`, `Sex`, `Race`,
`channel_config`, and either `training_include/evaluation_include` or
`pilot_split`):

```bash
export RAW_DATA=/path/to/raw_physionet
export PREPARED=/path/to/private/m2_prepared
export RECORD_METADATA=/path/to/private/training_record_metadata.csv

python prepare_data.py \
  --data-root "$RAW_DATA" \
  --output-root "$PREPARED" \
  --record-metadata "$RECORD_METADATA" \
  --workers 4
```

The output-root's intermediate inventory and alignment tables are runtime
inputs, not repository artifacts. The raw path may contain site subdirectories
and annotation files named `<record>_caisr_annotations.edf` and
`<record>_expert_annotations.edf`.

The signal implementation is in `preprocessing_primitives.py`. It maps
channels through `channel_mapper.py/channel_table.csv`, forms bipolar
components where required, applies the historical modality bandpasses
(EEG/EOG 0.3--35 Hz, EMG 10--45 Hz, ECG 0.5--40 Hz, respiration
0.05--5 Hz), resamples to the contract rates, computes 5-second hard-valid
masks, robust median/MAD normalization, extreme-activity masks, and writes
the canonical HDF5. `annotation_alignment_source.py` supplies the formal
epoch alignment and target semantics used by E1.

## 2. Train and freeze E1

The full E1 trainer uses the historical source-level protocol: seed
`20260804`, EMA teacher with decay `0.996`, 8 epochs by default, 128
windows/record, 64 natural windows, batch size 16, learning rate
`6e-4`, weight decay `1e-4`, token loss 1.0, covariance loss 0.05,
variance loss 1.25, and a 192-dimensional embedding head.

```bash
export E1_OUTPUT=/path/to/private/m2_e1

python train_e1.py \
  --manifest "$PREPARED/training_manifest.parquet" \
  --annotation-cache-dir "$PREPARED/canonical" \
  --candidate-index "$E1_OUTPUT/candidate_index.pkl.gz" \
  --output-dir "$E1_OUTPUT" \
  --variant e1 \
  --seed 20260804 \
  --epochs 8 \
  --windows-per-record 128 \
  --natural-windows 64 \
  --batch-size 16 \
  --num-workers 4 \
  --learning-rate 6e-4 \
  --weight-decay 1e-4 \
  --ema-decay 0.996 \
  --token-loss-weight 1.0 \
  --covariance-loss-weight 0.05 \
  --variance-loss-weight 1.25 \
  --device cuda \
  --amp \
  --overwrite
```

The trainer keeps student and EMA-teacher states in the private output. Freeze
the selected teacher/model state according to the local training run before
export; no checkpoint is committed.

## 3. Export frozen 192-D E1 embeddings

```bash
export EMBEDDINGS=/path/to/private/m2_embeddings

python export_embeddings.py \
  --checkpoint "$E1_OUTPUT/checkpoint_last.pt" \
  --prepared-root "$PREPARED" \
  --output-dir "$EMBEDDINGS" \
  --device cuda
```

This produces a private `embedding_manifest.parquet` and private HDF5 shard
files. The exporter checks the checkpoint identity and uses the same
canonical HDF5 reader as E1.

## 4. Train/evaluate M2 LOSO

For the historical M2 route, keep the fixed Large 6,600-record cohort and
these parameters unchanged:

- held-out sites: I0002, I0006, S0001;
- seeds: 20260806, 20260807, 20260808;
- inner 1--4 epoch selection followed by outer refitting;
- same-site age-pair loss with weight 0.15;
- ten 30-second windows per five-minute local token;
- two-layer local Transformer and three-layer full-night Transformer;
- native mean of the three raw logits.

```bash
python run_loso.py \
  --manifest "$PREPARED/training_manifest.parquet" \
  --embedding-manifest "$EMBEDDINGS/embedding_manifest.parquet" \
  --private-output private/m2_fresh \
  --results-dir results
```

The runner fails closed on cohort, site counts, record alignment, E1 identity,
CUDA, or fixed-parameter mismatches. It writes public aggregate CSVs only
after the primary target values match exactly.

## 5. Official metrics and optional M2 V5 residual

The public outputs are:

- `results/loso_metrics.csv`: I0002/I0006/S0001 AC-AUROC, AUROC, AUPRC;
- `results/summary.csv`: small-site AC, macro AC, macro AUROC, macro AUPRC;
- `results/residual_date/`: the separate M2 V5 CreationTime residual stage.

The verified primary result is:

| Held-out site | AC-AUROC | AUROC | AUPRC |
| --- | ---: | ---: | ---: |
| I0002 | 0.7481442205726405 | 0.7851483722270238 | 0.5209606504394242 |
| I0006 | 0.7393224315186642 | 0.7901005547850207 | 0.2945779196802375 |
| S0001 | 0.6680414632808643 | 0.7439418769121485 | 0.1867674115690459 |

Small-site AC-AUROC is `0.7437333260456523`; macro AC-AUROC is
`0.7185027051240563`; macro AUROC is `0.7730636013080644`; macro AUPRC is
`0.3341019938962359`.

The optional V5 stage is not mixed into primary M2:

```bash
python run_residual.py \
  --manifest "$PREPARED/training_manifest.parquet" \
  --primary-private-output private/m2_fresh \
  --private-output private/m2_v5_date_residual \
  --results-dir results/residual_date
```

Its verified macro AC-AUROC is `0.7725904425320311`. No V14/V18
date+CAISR/follow-up residual is included.

## Fast path with an existing historical frozen E1

If the historical E1 checkpoint and decoded embedding shards already exist,
skip sections 1--3 and run section 4 directly. This is only the reproducible
fast path; the raw-EDF and source-level E1 route above is the primary
technical documentation.

Use Python 3.10 and the historical CUDA/PyTorch environment described in
`requirements.txt`. The branch contains no EDF, prepared cache, embedding
cache, checkpoint, audit output, run log, patient-level prediction, or
absolute private data path.
