# CAISR20 特征与全量可用性审计

## 1. 目的与范围

CAISR20 是从 canonical HDF5 的 `annotations/caisr/` 读取的固定顺序、night-level `(20,)` 特征集合。本分支只读取已经对齐的 HDF5 annotation、timebase、metadata 和 manifest provenance；不读取 EDF 或 waveform，不写 HDF5，不执行滤波、重采样、归一化、QC 或模型训练。

本目录中的 `caisr_audit.py` 直接打开 manifest 指向的 canonical HDF5，复用 `caisr.py` 的 `extract` 计算逻辑后立即累加有效率统计，输出 `caisr.txt`；不读取原始 CAISR/human EDF，不保存逐条中间结果，也不会写入 `local_resume_output`。

## 2. 固定特征和输出契约

特征顺序固定为：

1. `caisr_sleep_tst_sec`
2. `caisr_sleep_se`
3. `caisr_sleep_sol_sec`
4. `caisr_sleep_rem_latency_sec`
5. `caisr_sleep_waso_sec`
6. `caisr_sleep_n1_pct`
7. `caisr_sleep_n2_pct`
8. `caisr_sleep_n3_pct`
9. `caisr_sleep_rem_pct`
10. `caisr_sleep_transition_rate`
11. `caisr_sleep_short_bout_ratio`
12. `caisr_sleep_mean_stage_entropy`
13. `caisr_arousal_arousal_index`
14. `caisr_arousal_arousal_burden_ratio`
15. `caisr_arousal_mean_arousal_duration`
16. `caisr_respiratory_respiratory_event_index`
17. `caisr_respiratory_respiratory_burden_ratio`
18. `caisr_respiratory_mean_resp_duration`
19. `caisr_limb_limb_movement_index`
20. `caisr_limb_plmi`

`FeatureBlock` 固定为 `float32`、rank-1 `(20,)`。有效值必须 finite；无效值为 NaN，并由 `valid_mask` 和 `invalid_reason` 同时记录原因。诊断信息放在 `BranchResult.metadata`，不扩展特征向量。

## 3. 标注来源与缺失语义

- manifest 中 `algorithmic_annotations` 表示自动 CAISR，`human_annotations` 表示人工标注，空路径表示 unavailable。
- 自动/人工/unavailable 由 manifest provenance 与 HDF5 `metadata/caisr/channel_names` 共同判定，不能从全零事件数组推断来源。
- canonical stage code 为 `1=N3`、`2=N2`、`3=N1`、`4=REM`、`5=Wake`；`-1` 表示 invalid/unknown。`0`、`9` 不是 canonical 有效 stage。
- 无效 stage 保留原 epoch 位置，并打断相邻 transition、sleep bout 和事件统计；不会被当作 Wake。
- 人工 annotation 没有五列 stage probability 时，entropy 保持 NaN/invalid，不填 0，也不构造 one-hot 概率。
- 事件来源可用但没有事件时，index/burden 为 0，平均事件时长为 NaN；来源 unavailable 时对应特征为 NaN。

## 4. CAISR20 计算规则

- TST 为有效 N3/N2/N1/REM epoch 数乘 30 秒。
- SE 使用 TST 除以完整 PSG 记录时长；SOL 为首个 sleep epoch 的 PSG-relative 秒数；无 sleep 或无 REM 时相应特征为 NaN；WASO 为首个 sleep 后的 Wake 时长。
- 睡眠阶段百分比以 TST 为分母；transition rate 为合法相邻有效 stage transition 数除以 TST 小时。
- short bout ratio 按连续同一 sleep stage bout 计算，长度严格小于 3 个 epoch 的 bout 占比；invalid、Wake 或 stage 变化均断开 bout。
- mean stage entropy 只使用五项 probability 全 finite 且 `stage_probability_valid=True` 的行，计算 `-sum(p*log(p))/log(5)`；不做 `nan_to_num`、clipping 或 renormalization。
- arousal 使用 2 Hz 二值 annotation；respiratory/limb 使用 1 Hz categorical annotation。respiratory 保留 `1..5` subtype，limb 使用 canonical `0/1/2`。

## 5. 已完成的 6600 条预审

此前预审结果为：

| 站点 | 自动 CAISR | 人工标注 | unavailable | 总数 |
|---|---:|---:|---:|---:|
| I0002 | 317 | 2 | 0 | 319 |
| I0006 | 1138 | 4 | 0 | 1142 |
| S0001 | 5075 | 57 | 7 | 5139 |
| 合计 | 6530 | 63 | 7 | 6600 |

预审发现：6570 条记录至少有一个有效 stage；30 条没有有效 sleep stage；188 条在 sleep 后没有 REM；70 条没有共同 finite 且 valid 的五类 stage probability epoch。validity mask 与 finite 行没有发现不一致。此前预审明细保留在 `caisr20_precheck_flags.tsv`。

## 6. 与旧实现的差异

旧 `feature_extractor_algorithmic.py` 默认会压缩 invalid stage，并对 unavailable 输入使用零回退；canonical CAISR20 保留 PSG-relative 位置并显式记录 NaN/invalid。旧 entropy 路径会 `nan_to_num`、clipping 和行归一化；canonical 只接受真实有效概率行。旧实现的 20 个选中特征位置来自 `data2_caisr20_results/feature_manifest.csv`；当前实现不复制训练、标准化、评分或模型拼接逻辑。

单条旧 NPZ/HDF5 对照中，20 个位置有 19 个数值一致；唯一预期差异是 canonical entropy 按 `log(5)` 归一化，而旧输出保留未归一化熵。

## 7. 使用方法

单条特征提取：

```bash
PYTHONNOUSERSITE=1 \
/database/home/gaohaojie/.conda/envs/sleepfm_env/bin/python \
-m feature_extraction.branches.caisr \
--manifest cache_manifest.tsv \
--record-id sub-I0002150026011_ses-2 \
--show-diagnostics
```

6600 条 annotation-only 有效率审计：

```bash
PYTHONNOUSERSITE=1 \
/database/home/gaohaojie/.conda/envs/sleepfm_env/bin/python \
feature_extraction/check/caisr/caisr_audit.py \
--manifest cache_manifest.tsv \
--output feature_extraction/check/caisr/caisr.txt
```

审计输出至少包含：总扫描数、成功提取数、错误记录、20 个特征的 valid/invalid 数和比例、每个站点的 invalid 数、invalid reason 分布及示例 record_id。
