"""PSG 多模态 canonical source 解析与 provenance 规划模块。

本模块负责调用 channel alias 标准化结果，把 EDF 原始标签组织到 EEG、EOG、ECG、
EMG、RESP 和 SpO2 的 canonical slots，并按照既有优先级选择 direct source 或
成对导联推导 source。它同时生成 SourceSpec 和 SourcePlan，供 waveform cache
写入 selected source、candidate source、reference、采样率和 sample count 等 provenance。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Iterable, Sequence

from channel_mapper import ChannelMapper


CHANNEL_TABLE = Path(__file__).resolve().with_name("channel_table.csv")

CANONICAL_BY_MODALITY = {
    "eeg": (
        "EEG_F3",
        "EEG_F4",
        "EEG_C3",
        "EEG_C4",
        "EEG_O1",
        "EEG_O2",
    ),
    "eog": ("EOG_E1", "EOG_E2"),
    "ecg": ("ECG",),
    "emg": ("CHIN_EMG", "LEG_EMG_LEFT", "LEG_EMG_RIGHT"),
    "resp": (
        "NASAL_PRESSURE",
        "THERMAL_AIRFLOW",
        "AIRFLOW_UNSPECIFIED",
        "PAP_FLOW",
        "PAP_PRESSURE",
        "THORACIC_EFFORT",
        "ABDOMINAL_EFFORT",
    ),
    "spo2": ("SPO2",),
}


@dataclass(frozen=True)
class SourceSpec:
    """描述一个 canonical slot 的源通道、推导方法和参考关系。"""

    indices: tuple[int, ...]
    labels: tuple[str, ...]
    method: str
    reference: str = ""


@dataclass(frozen=True)
class SourcePlan:
    """记录一个 canonical slot 的 header-level source provenance 和解析异常。"""

    canonical_name: str
    candidate_source_labels: tuple[str, ...]
    selected_source_labels: tuple[str, ...]
    channel_present: bool
    channel_source: str
    derivation_method: str
    reference_system: str
    native_sampling_rate_hz: float
    native_sample_count: int
    physical_dimension: str
    selected_source_indices: tuple[int, ...] = ()
    anomalies: tuple[str, ...] = ()


_CANDIDATE_CANONICALS_BY_SLOT = {
    "EEG_F3": ("F3-M2", "F3", "M2"),
    "EEG_F4": ("F4-M1", "F4", "M1"),
    "EEG_C3": ("C3-M2", "C3", "M2"),
    "EEG_C4": ("C4-M1", "C4", "M1"),
    "EEG_O1": ("O1-M2", "O1", "M2"),
    "EEG_O2": ("O2-M1", "O2", "M1"),
    "EOG_E1": ("LOC", "E1", "M2"),
    "EOG_E2": ("ROC", "E2", "M1"),
    "CHIN_EMG": ("CHIN1", "CHIN2", "CHIN"),
    "LEG_EMG_LEFT": ("LAT", "LLEG+", "LLEG-"),
    "LEG_EMG_RIGHT": ("RAT", "RLEG+", "RLEG-"),
    "ECG": ("EKG",),
    "NASAL_PRESSURE": ("NASAL_PRESSURE",),
    "THERMAL_AIRFLOW": ("AIRFLOW",),
    "AIRFLOW_UNSPECIFIED": ("AIRFLOW",),
    "PAP_FLOW": ("CFLOW",),
    "PAP_PRESSURE": ("CPAP_PRESSURE",),
    "THORACIC_EFFORT": ("CHEST",),
    "ABDOMINAL_EFFORT": ("ABDOMINAL",),
    "SPO2": ("SaO2",),
}


def _first(values: Iterable[int]) -> int | None:
    return next(iter(values), None)


def _mapped_indices(mapper: ChannelMapper, labels: list[str]) -> dict[str, list[int]]:
    output: dict[str, list[int]] = {}
    for index, label in enumerate(labels):
        info = mapper.map_channel(label)
        if info.matched and info.canonical_name:
            output.setdefault(info.canonical_name, []).append(index)
    return output


def _direct(mapped: dict[str, list[int]], canonical: str, labels: list[str]) -> SourceSpec | None:
    index = _first(mapped.get(canonical, []))
    if index is None:
        return None
    return SourceSpec((index,), (labels[index],), "direct")


def _direct_filtered(
    mapped: dict[str, list[int]],
    canonical: str,
    labels: list[str],
    mapper: ChannelMapper,
    rejected_names: set[str],
) -> SourceSpec | None:
    index = _first(
        candidate
        for candidate in mapped.get(canonical, [])
        if mapper.normalize_channel_name(labels[candidate]) not in rejected_names
    )
    if index is None:
        return None
    return SourceSpec((index,), (labels[index],), "direct")


def _direct_preferred(
    mapped: dict[str, list[int]],
    canonical: str,
    labels: list[str],
    mapper: ChannelMapper,
    preferred_names: list[str],
) -> SourceSpec | None:
    candidates = mapped.get(canonical, [])
    if not candidates:
        return None
    rank = {name: index for index, name in enumerate(preferred_names)}
    selected = min(
        candidates,
        key=lambda index: (
            rank.get(mapper.normalize_channel_name(labels[index]), len(rank)),
            index,
        ),
    )
    return SourceSpec((selected,), (labels[selected],), "direct")


def _direct_matching(
    mapped: dict[str, list[int]],
    canonical: str,
    labels: list[str],
    mapper: ChannelMapper,
    accepted_names: set[str],
) -> SourceSpec | None:
    index = _first(
        candidate
        for candidate in mapped.get(canonical, [])
        if mapper.normalize_channel_name(labels[candidate]) in accepted_names
    )
    if index is None:
        return None
    return SourceSpec((index,), (labels[index],), "direct")


def _direct_named(
    labels: list[str],
    mapper: ChannelMapper,
    accepted_names: set[str],
) -> SourceSpec | None:
    index = _first(
        candidate
        for candidate, label in enumerate(labels)
        if mapper.normalize_channel_name(label) in accepted_names
    )
    if index is None:
        return None
    return SourceSpec((index,), (labels[index],), "direct")


def _direct_named_preferred(
    labels: list[str],
    mapper: ChannelMapper,
    preferred_names: list[str],
) -> SourceSpec | None:
    rank = {name: index for index, name in enumerate(preferred_names)}
    candidates = [
        index
        for index, label in enumerate(labels)
        if mapper.normalize_channel_name(label) in rank
    ]
    if not candidates:
        return None
    selected = min(
        candidates,
        key=lambda index: (rank[mapper.normalize_channel_name(labels[index])], index),
    )
    return SourceSpec((selected,), (labels[selected],), "direct")


def _component(
    mapped: dict[str, list[int]],
    positive: str,
    negative: str,
    labels: list[str],
    mapper: ChannelMapper,
) -> SourceSpec | None:
    positive_candidates = mapped.get(positive, [])
    negative_candidates = mapped.get(negative, [])
    for positive_index in positive_candidates:
        normalized = mapper.normalize_channel_name(labels[positive_index])
        accepted = {
            positive.lower(),
            f"eeg {positive.lower()}",
            f"eog {positive.lower()}",
        }
        if normalized not in accepted:
            continue
        for negative_index in negative_candidates:
            negative_normalized = mapper.normalize_channel_name(labels[negative_index])
            negative_aliases = {
                negative.lower(),
                f"eeg {negative.lower()}",
            }
            if negative == "M1":
                negative_aliases.add("a1")
            elif negative == "M2":
                negative_aliases.add("a2")
            if negative_normalized not in negative_aliases:
                continue
            return SourceSpec(
                (positive_index, negative_index),
                (labels[positive_index], labels[negative_index]),
                "component_minus_reference",
                negative,
            )
    return None


def _paired_by_names(
    mapped: dict[str, list[int]],
    canonical: str,
    labels: list[str],
    mapper: ChannelMapper,
    pairs: list[tuple[set[str], set[str]]],
) -> SourceSpec | None:
    candidates = mapped.get(canonical, [])
    normalized = {index: mapper.normalize_channel_name(labels[index]) for index in candidates}
    for positive_names, negative_names in pairs:
        positive = _first(index for index in candidates if normalized[index] in positive_names)
        negative = _first(index for index in candidates if normalized[index] in negative_names)
        if positive is not None and negative is not None:
            return SourceSpec(
                (positive, negative),
                (labels[positive], labels[negative]),
                "positive_minus_negative",
            )
    return None


def _paired_components(
    mapped: dict[str, list[int]],
    positive_canonical: str,
    negative_canonical: str,
    labels: list[str],
    mapper: ChannelMapper,
    positive_names: set[str],
    negative_names: set[str],
) -> SourceSpec | None:
    positive = _first(
        index
        for index in mapped.get(positive_canonical, [])
        if mapper.normalize_channel_name(labels[index]) in positive_names
    )
    negative = _first(
        index
        for index in mapped.get(negative_canonical, [])
        if mapper.normalize_channel_name(labels[index]) in negative_names
    )
    if positive is None or negative is None:
        return None
    return SourceSpec(
        (positive, negative),
        (labels[positive], labels[negative]),
        "positive_minus_negative",
    )


def build_source_specs(labels: list[str]) -> dict[str, SourceSpec | None]:
    """按照已经确认的 alias、优先级和 bipolar 规则生成 canonical source specifications。"""

    mapper = ChannelMapper(CHANNEL_TABLE)
    mapped = _mapped_indices(mapper, labels)
    specs: dict[str, SourceSpec | None] = {}

    eeg_targets = {
        "EEG_F3": ("F3-M2", "F3", "M2"),
        "EEG_F4": ("F4-M1", "F4", "M1"),
        "EEG_C3": ("C3-M2", "C3", "M2"),
        "EEG_C4": ("C4-M1", "C4", "M1"),
        "EEG_O1": ("O1-M2", "O1", "M2"),
        "EEG_O2": ("O2-M1", "O2", "M1"),
    }
    for target, (direct_name, positive, negative) in eeg_targets.items():
        specs[target] = _direct(mapped, direct_name, labels) or _component(
            mapped, positive, negative, labels, mapper
        )

    left_component = _paired_components(
        mapped,
        "LOC",
        "M2",
        labels,
        mapper,
        {"loc", "eog1", "eog l", "eog-l", "leog"},
        {"m2", "a2", "eeg m2"},
    ) or _component(mapped, "E1", "M2", labels, mapper)
    right_component = _paired_components(
        mapped,
        "ROC",
        "M1",
        labels,
        mapper,
        {"roc", "eog2", "eog r", "eog-r", "reog"},
        {"m1", "a1", "eeg m1"},
    ) or _component(mapped, "E2", "M1", labels, mapper)
    specs["EOG_E1"] = left_component or _direct(mapped, "E1", labels) or _direct_filtered(
        mapped,
        "LOC",
        labels,
        mapper,
        {"loc", "eog1", "eog l", "eog-l", "leog"},
    )
    specs["EOG_E2"] = right_component or _direct(mapped, "E2", labels) or _direct_filtered(
        mapped,
        "ROC",
        labels,
        mapper,
        {"roc", "eog2", "eog r", "eog-r", "reog"},
    )

    specs["CHIN_EMG"] = _paired_components(
        mapped,
        "CHIN1",
        "CHIN2",
        labels,
        mapper,
        {"chin 1", "chin1", "chin l", "chin-l", "chinl", "china", "emg1", "l chin"},
        {"chin 2", "chin2", "chin r", "chin-r", "chinr", "chinb", "emg2", "r chin"},
    ) or _direct(mapped, "CHIN", labels)
    specs["LEG_EMG_LEFT"] = _direct(mapped, "LAT", labels) or _paired_components(
        mapped,
        "LLEG+",
        "LLEG-",
        labels,
        mapper,
        {"l-leg1", "lat 1", "lat-u", "lat1", "lleg plus"},
        {"l-leg2", "lat 2", "lat-l", "lat2", "lleg-"},
    )
    specs["LEG_EMG_RIGHT"] = _direct_filtered(
        mapped, "RAT", labels, mapper, {"l rat"}
    ) or _paired_components(
        mapped,
        "RLEG+",
        "RLEG-",
        labels,
        mapper,
        {"r-leg1", "rat 1", "rat-u", "rat1", "rleg plus"},
        {"r-leg2", "rat 2", "rat-l", "rat2", "rleg-"},
    )

    ecg_candidates = mapped.get("EKG", [])
    ecg_pair = _paired_by_names(
        mapped,
        "EKG",
        labels,
        mapper,
        [
            ({"ecg1", "ekg-l", "ecg l", "ecg-l"}, {"ecg2", "ekg-r", "ecg r", "ecg-r"}),
            ({"ecg-la"}, {"ecg-ra"}),
            ({"ekg1"}, {"ekg2"}),
        ],
    )
    generic_ecg = _first(
        index
        for index in ecg_candidates
        if mapper.normalize_channel_name(labels[index]) in {"ecg", "ekg", "ecg ii", "ecgii"}
    )
    if generic_ecg is not None:
        specs["ECG"] = SourceSpec((generic_ecg,), (labels[generic_ecg],), "direct")
    elif ecg_pair is not None:
        specs["ECG"] = ecg_pair
    else:
        specs["ECG"] = _direct(mapped, "EKG", labels)

    specs["NASAL_PRESSURE"] = _direct_preferred(
        mapped,
        "NASAL_PRESSURE",
        labels,
        mapper,
        ["ptaf", "npt", "nasal pressure", "pressure", "nasal", "cannula", "nptaf"],
    )
    specs["THERMAL_AIRFLOW"] = _direct_named_preferred(
        labels,
        mapper,
        ["thermistor", "therm", "thermal airflow", "thermal", "oral thermistor", "oral therm", "thermistor 2"],
    )
    specs["AIRFLOW_UNSPECIFIED"] = _direct_matching(
        mapped, "AIRFLOW", labels, mapper, {"airflow", "air flow", "flow"}
    )
    specs["PAP_FLOW"] = _direct_preferred(
        mapped, "CFLOW", labels, mapper, ["cflow", "c-flow", "c flow", "cpap flow"]
    )
    specs["PAP_PRESSURE"] = _direct_preferred(
        mapped, "CPAP_PRESSURE", labels, mapper, ["cpres", "cpap pressure", "c press", "cpap press", "cpress"]
    )
    specs["THORACIC_EFFORT"] = _direct(mapped, "CHEST", labels)
    specs["ABDOMINAL_EFFORT"] = _direct(mapped, "ABDOMINAL", labels)
    specs["SPO2"] = _direct(mapped, "SaO2", labels)
    return specs



def _header_values(
    labels: Sequence[str],
    values: Sequence[object] | None,
    *,
    default: object,
) -> tuple[object, ...]:
    if values is None:
        return tuple(default for _ in labels)
    if len(values) != len(labels):
        raise ValueError("header metadata length does not match labels")
    return tuple(values)


def build_source_plans(
    labels: Sequence[str],
    native_sampling_rates_hz: Sequence[float] | None = None,
    native_sample_counts: Sequence[int] | None = None,
    physical_dimensions: Sequence[str] | None = None,
) -> dict[str, SourcePlan]:
    """根据 EDF channel header 构建 candidate、selected 和 native metadata provenance。"""

    labels = tuple(str(label).strip() for label in labels)
    rates = tuple(
        float(value)
        for value in _header_values(labels, native_sampling_rates_hz, default=float("nan"))
    )
    counts = tuple(
        int(value)
        for value in _header_values(labels, native_sample_counts, default=-1)
    )
    dimensions = tuple(
        str(value).strip()
        for value in _header_values(labels, physical_dimensions, default="")
    )
    mapper = ChannelMapper(CHANNEL_TABLE)
    infos = tuple(mapper.map_channel(label) for label in labels)
    specs = build_source_specs(list(labels))
    plans: dict[str, SourcePlan] = {}

    for canonical_name, candidate_canonicals in _CANDIDATE_CANONICALS_BY_SLOT.items():
        allowed = set(candidate_canonicals)
        candidate_entries = tuple(
            (index, label, info)
            for index, (label, info) in enumerate(zip(labels, infos))
            if info.matched and info.canonical_name in allowed
        )
        candidates = tuple(label for _, label, _ in candidate_entries)
        spec = specs.get(canonical_name)
        selected_labels = tuple() if spec is None else tuple(spec.labels)
        anomalies: list[str] = []
        if spec is None:
            anomalies.append(
                "source_unresolved" if candidates else "no_mapped_candidates"
            )

        selected_indices = tuple() if spec is None else tuple(spec.indices)
        selected_rates = tuple(rates[index] for index in selected_indices)
        selected_counts = tuple(counts[index] for index in selected_indices)
        selected_units = tuple(dimensions[index] for index in selected_indices)
        native_rate = float("nan")
        native_count = -1
        physical_dimension = ""
        if selected_indices:
            finite_rates = [value for value in selected_rates if math.isfinite(value)]
            if len(finite_rates) != len(selected_rates) or len(set(selected_rates)) != 1:
                anomalies.append("selected_sources_fs_inconsistent")
            else:
                native_rate = selected_rates[0]
            if len(set(selected_counts)) != 1:
                anomalies.append("selected_sources_sample_count_inconsistent")
            else:
                native_count = selected_counts[0]
            if len(set(selected_units)) != 1:
                anomalies.append("selected_sources_physical_dimension_inconsistent")
            else:
                physical_dimension = selected_units[0]

        direct_canonical = candidate_canonicals[0]
        direct_candidates = tuple(
            info for _, _, info in candidate_entries
            if info.canonical_name == direct_canonical
        )
        if direct_candidates:
            best_priority = min(info.priority for info in direct_candidates)
            if sum(info.priority == best_priority for info in direct_candidates) > 1:
                anomalies.append("multiple_same_priority_direct_candidates")

        if spec is None:
            channel_source = "unavailable"
        elif spec.method == "direct":
            channel_source = "direct"
        else:
            channel_source = "derived"
        plans[canonical_name] = SourcePlan(
            canonical_name=canonical_name,
            candidate_source_labels=candidates,
            selected_source_labels=selected_labels,
            channel_present=spec is not None,
            channel_source=channel_source,
            derivation_method="unavailable" if spec is None else spec.method,
            reference_system="" if spec is None else spec.reference,
            native_sampling_rate_hz=native_rate,
            native_sample_count=native_count,
            physical_dimension=physical_dimension,
            selected_source_indices=selected_indices,
            anomalies=tuple(dict.fromkeys(anomalies)),
        )
    return plans


def find_resp_source_conflicts(
    source_plans: dict[str, SourcePlan],
) -> tuple[tuple[int, str, str], ...]:
    """找出同一 raw EDF channel 被分配给多个 RESP canonical slots 的情况。"""

    selected_by_raw_index: dict[int, list[str]] = {}
    for canonical_name in CANONICAL_BY_MODALITY["resp"]:
        plan = source_plans[canonical_name]
        for raw_index in plan.selected_source_indices:
            selected_by_raw_index.setdefault(raw_index, []).append(canonical_name)

    conflicts: list[tuple[int, str, str]] = []
    for raw_index, slots in sorted(selected_by_raw_index.items()):
        unique_slots = tuple(dict.fromkeys(slots))
        for left_index, left in enumerate(unique_slots):
            for right in unique_slots[left_index + 1 :]:
                conflicts.append((raw_index, left, right))
    return tuple(conflicts)
