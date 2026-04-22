"""Notebook-first utilities for CyTOF archetype embedding experiments."""

from .analysis import (
    archetype_marker_rankings,
    cosine_similarity_matrix,
    dominant_assignments,
    kl_history_columns,
    load_run_outputs,
    nearest_neighbors_from_similarity,
    per_marker_reconstruction_stats,
    posterior_mean_weights,
    residual_norms,
    residual_summary,
    summarize_by_group,
    weight_entropy,
)
from .data import (
    DataBundle,
    MarkerScaler,
    extract_matrix,
    fit_scaler,
    preprocess_array,
    split_train_val_indices,
)
from .model import ArchetypeEmbeddingModel, ProbabilisticArchetypeModel
from .train import build_sweep_configs, run_sweep, train_one_run
from .utils import collect_software_versions, make_run_id, resolve_device, set_seed, validate_run_config

__all__ = [
    "ArchetypeEmbeddingModel",
    "DataBundle",
    "MarkerScaler",
    "ProbabilisticArchetypeModel",
    "archetype_marker_rankings",
    "build_sweep_configs",
    "collect_software_versions",
    "cosine_similarity_matrix",
    "dominant_assignments",
    "extract_matrix",
    "fit_scaler",
    "kl_history_columns",
    "load_run_outputs",
    "make_run_id",
    "nearest_neighbors_from_similarity",
    "per_marker_reconstruction_stats",
    "posterior_mean_weights",
    "preprocess_array",
    "resolve_device",
    "residual_norms",
    "residual_summary",
    "run_sweep",
    "set_seed",
    "split_train_val_indices",
    "summarize_by_group",
    "train_one_run",
    "validate_run_config",
    "weight_entropy",
]
