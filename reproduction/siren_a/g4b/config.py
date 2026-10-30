"""Fixed seed schedule for seeds 43--56; data paths are caller supplied."""
from typing import Dict

from reproduction.siren_a.g4.config import *

NEW_SEEDS = tuple(range(43, 57))
G4A_SEED = 42
SCHEMA = "g4b_seeds43_56_v1"
PRIVATE_ROOT = Path("private")
G4A_ROOT = Path("private")
G4A_RUN_IDENTITY_SHA256 = ""
G3B_RUN_IDENTITY_SHA256 = ""
G3B_COMPLETE_SHA256 = ""
THREAD_ENV = {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}


def new_model_identities():
    return tuple((site, seed, view, model) for seed in NEW_SEEDS for site in TRAIN_SITES for view in VIEWS for model in MODEL_TYPES)


def new_prediction_identities():
    return tuple((holdout, protocol, view, model, seed) for seed in NEW_SEEDS for holdout in HOLDOUTS for protocol in PROTOCOLS for view in VIEWS for model in MODEL_TYPES)


def full_model_identities():
    return tuple((site, seed, view, model) for seed in ALL_SEEDS for site in TRAIN_SITES for view in VIEWS for model in MODEL_TYPES)


def full_prediction_identities():
    return tuple((holdout, protocol, view, model, seed) for seed in ALL_SEEDS for holdout in HOLDOUTS for protocol in PROTOCOLS for view in VIEWS for model in MODEL_TYPES)


def parameters_for_seed(model_type: str, seed: int, truncation_level: int) -> Dict[str, object]:
    if not isinstance(seed, int) or seed not in NEW_SEEDS:
        raise ValueError("G4-B seed must be an integer in 43..56")
    if model_type == "lr":
        parameters = dict(LR_PARAMS)
    elif model_type == "hgb":
        parameters = dict(HGB_PARAMS)
    elif model_type == "lambdarank":
        parameters = dict(LGBM_PARAMS)
        parameters["lambdarank_truncation_level"] = int(truncation_level)
    else:
        raise ValueError("unknown model type")
    parameters["random_state"] = seed
    return parameters
