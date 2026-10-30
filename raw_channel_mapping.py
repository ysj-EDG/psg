"""Historical M2 raw-EDF source-spec mapping.

The functions below are extracted from source_m2/online_features.py at
ysj-EDG/physionet-challenge-2026-baseline commit c71c731. The wrapper adds
an explicit table_path argument so prepare_data.py can remain relocatable.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from channel_mapper import ChannelMapper

@dataclass(frozen=True)
class SourceSpec:
    indices: tuple[int, ...]
    labels: tuple[str, ...]
    method: str
    reference: str = ''


def _first(values: Iterable[int]) -> int | None:
    return next(iter(values), None)


def _mapped_indices(mapper: ChannelMapper, labels: list[str]) -> dict[str, list[int]]:
    output: dict[str, list[int]] = {}
    for (index, label) in enumerate(labels):
        info = mapper.map_channel(label)
        if info.matched and info.canonical_name:
            output.setdefault(info.canonical_name, []).append(index)
    return output


def _direct(mapped: dict[str, list[int]], canonical: str, labels: list[str]) -> SourceSpec | None:
    index = _first(mapped.get(canonical, []))
    if index is None:
        return None
    return SourceSpec((index,), (labels[index],), 'direct')


def _direct_filtered(mapped: dict[str, list[int]], canonical: str, labels: list[str], mapper: ChannelMapper, rejected_names: set[str]) -> SourceSpec | None:
    index = _first((candidate for candidate in mapped.get(canonical, []) if mapper.normalize_channel_name(labels[candidate]) not in rejected_names))
    if index is None:
        return None
    return SourceSpec((index,), (labels[index],), 'direct')


def _direct_preferred(mapped: dict[str, list[int]], canonical: str, labels: list[str], mapper: ChannelMapper, preferred_names: list[str]) -> SourceSpec | None:
    candidates = mapped.get(canonical, [])
    if not candidates:
        return None
    rank = {name: index for (index, name) in enumerate(preferred_names)}
    selected = min(candidates, key=lambda index: (rank.get(mapper.normalize_channel_name(labels[index]), len(rank)), index))
    return SourceSpec((selected,), (labels[selected],), 'direct')


def _direct_matching(mapped: dict[str, list[int]], canonical: str, labels: list[str], mapper: ChannelMapper, accepted_names: set[str]) -> SourceSpec | None:
    index = _first((candidate for candidate in mapped.get(canonical, []) if mapper.normalize_channel_name(labels[candidate]) in accepted_names))
    if index is None:
        return None
    return SourceSpec((index,), (labels[index],), 'direct')


def _component(mapped: dict[str, list[int]], positive: str, negative: str, labels: list[str], mapper: ChannelMapper) -> SourceSpec | None:
    positive_candidates = mapped.get(positive, [])
    negative_candidates = mapped.get(negative, [])
    for positive_index in positive_candidates:
        normalized = mapper.normalize_channel_name(labels[positive_index])
        accepted = {positive.lower(), f'eeg {positive.lower()}', f'eog {positive.lower()}'}
        if normalized not in accepted:
            continue
        for negative_index in negative_candidates:
            negative_normalized = mapper.normalize_channel_name(labels[negative_index])
            negative_aliases = {negative.lower(), f'eeg {negative.lower()}'}
            if negative == 'M1':
                negative_aliases.add('a1')
            elif negative == 'M2':
                negative_aliases.add('a2')
            if negative_normalized not in negative_aliases:
                continue
            return SourceSpec((positive_index, negative_index), (labels[positive_index], labels[negative_index]), 'component_minus_reference', negative)
    return None


def _paired_by_names(mapped: dict[str, list[int]], canonical: str, labels: list[str], mapper: ChannelMapper, pairs: list[tuple[set[str], set[str]]]) -> SourceSpec | None:
    candidates = mapped.get(canonical, [])
    normalized = {index: mapper.normalize_channel_name(labels[index]) for index in candidates}
    for (positive_names, negative_names) in pairs:
        positive = _first((index for index in candidates if normalized[index] in positive_names))
        negative = _first((index for index in candidates if normalized[index] in negative_names))
        if positive is not None and negative is not None:
            return SourceSpec((positive, negative), (labels[positive], labels[negative]), 'positive_minus_negative')
    return None


def _paired_components(mapped: dict[str, list[int]], positive_canonical: str, negative_canonical: str, labels: list[str], mapper: ChannelMapper, positive_names: set[str], negative_names: set[str]) -> SourceSpec | None:
    positive = _first((index for index in mapped.get(positive_canonical, []) if mapper.normalize_channel_name(labels[index]) in positive_names))
    negative = _first((index for index in mapped.get(negative_canonical, []) if mapper.normalize_channel_name(labels[index]) in negative_names))
    if positive is None or negative is None:
        return None
    return SourceSpec((positive, negative), (labels[positive], labels[negative]), 'positive_minus_negative')


def build_source_specs(labels: list[str], table_path: Path | str) -> dict[str, SourceSpec | None]:
    mapper = ChannelMapper(table_path)
    mapped = _mapped_indices(mapper, labels)
    specs: dict[str, SourceSpec | None] = {}
    eeg_targets = {'EEG_F3': ('F3-M2', 'F3', 'M2'), 'EEG_F4': ('F4-M1', 'F4', 'M1'), 'EEG_C3': ('C3-M2', 'C3', 'M2'), 'EEG_C4': ('C4-M1', 'C4', 'M1'), 'EEG_O1': ('O1-M2', 'O1', 'M2'), 'EEG_O2': ('O2-M1', 'O2', 'M1')}
    for (target, (direct_name, positive, negative)) in eeg_targets.items():
        specs[target] = _direct(mapped, direct_name, labels) or _component(mapped, positive, negative, labels, mapper)
    left_component = _paired_components(mapped, 'LOC', 'M2', labels, mapper, {'loc', 'eog1', 'eog l', 'eog-l', 'leog'}, {'m2', 'a2', 'eeg m2'}) or _component(mapped, 'E1', 'M2', labels, mapper)
    right_component = _paired_components(mapped, 'ROC', 'M1', labels, mapper, {'roc', 'eog2', 'eog r', 'eog-r', 'reog'}, {'m1', 'a1', 'eeg m1'}) or _component(mapped, 'E2', 'M1', labels, mapper)
    specs['EOG_E1'] = left_component or _direct(mapped, 'E1', labels) or _direct_filtered(mapped, 'LOC', labels, mapper, {'loc', 'eog1', 'eog l', 'eog-l', 'leog'})
    specs['EOG_E2'] = right_component or _direct(mapped, 'E2', labels) or _direct_filtered(mapped, 'ROC', labels, mapper, {'roc', 'eog2', 'eog r', 'eog-r', 'reog'})
    specs['CHIN_EMG'] = _paired_components(mapped, 'CHIN1', 'CHIN2', labels, mapper, {'chin 1', 'chin1', 'chin l', 'chin-l', 'chinl', 'china', 'emg1', 'l chin'}, {'chin 2', 'chin2', 'chin r', 'chin-r', 'chinr', 'chinb', 'emg2', 'r chin'}) or _direct(mapped, 'CHIN', labels)
    specs['LEG_EMG_LEFT'] = _direct(mapped, 'LAT', labels) or _paired_components(mapped, 'LLEG+', 'LLEG-', labels, mapper, {'l-leg1', 'lat 1', 'lat-u', 'lat1', 'lleg plus'}, {'l-leg2', 'lat 2', 'lat-l', 'lat2', 'lleg-'})
    specs['LEG_EMG_RIGHT'] = _direct_filtered(mapped, 'RAT', labels, mapper, {'l rat'}) or _paired_components(mapped, 'RLEG+', 'RLEG-', labels, mapper, {'r-leg1', 'rat 1', 'rat-u', 'rat1', 'rleg plus'}, {'r-leg2', 'rat 2', 'rat-l', 'rat2', 'rleg-'})
    ecg_candidates = mapped.get('EKG', [])
    ecg_pair = _paired_by_names(mapped, 'EKG', labels, mapper, [({'ecg1', 'ekg-l', 'ecg l', 'ecg-l'}, {'ecg2', 'ekg-r', 'ecg r', 'ecg-r'}), ({'ekg1'}, {'ekg2'})])
    generic_ecg = _first((index for index in ecg_candidates if mapper.normalize_channel_name(labels[index]) in {'ecg', 'ekg', 'ecg ii', 'ecgii'}))
    if generic_ecg is not None:
        specs['ECG'] = SourceSpec((generic_ecg,), (labels[generic_ecg],), 'direct')
    elif ecg_pair is not None:
        specs['ECG'] = ecg_pair
    else:
        specs['ECG'] = _direct(mapped, 'EKG', labels)
    specs['NASAL_PRESSURE'] = _direct_preferred(mapped, 'NASAL_PRESSURE', labels, mapper, ['ptaf', 'npt', 'nasal pressure', 'pressure', 'nasal', 'cannula', 'nptaf'])
    specs['THERMAL_AIRFLOW'] = _direct_matching(mapped, 'AIRFLOW', labels, mapper, {'thermistor', 'therm', 'thermistor 2', 'thermal airflow', 'thermal', 'oral thermistor', 'oral therm'})
    specs['AIRFLOW_UNSPECIFIED'] = _direct_matching(mapped, 'AIRFLOW', labels, mapper, {'airflow', 'air flow', 'flow'})
    specs['PAP_FLOW'] = _direct_preferred(mapped, 'CFLOW', labels, mapper, ['cflow', 'c-flow', 'c flow', 'cpap flow'])
    specs['PAP_PRESSURE'] = _direct_preferred(mapped, 'CPAP_PRESSURE', labels, mapper, ['cpres', 'cpap pressure', 'c press', 'cpap press', 'cpress'])
    specs['THORACIC_EFFORT'] = _direct(mapped, 'CHEST', labels)
    specs['ABDOMINAL_EFFORT'] = _direct(mapped, 'ABDOMINAL', labels)
    specs['SPO2'] = _direct(mapped, 'SaO2', labels)
    return specs
