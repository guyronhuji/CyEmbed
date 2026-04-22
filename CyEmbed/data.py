from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


def _to_numpy(x: Any) -> np.ndarray:
    """Convert dense/sparse-like matrix to numpy array."""
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray())
    return np.asarray(x)


@dataclass
class DataBundle:
    """Container for matrix and aligned metadata."""

    X: np.ndarray
    marker_names: list[str]
    cell_ids: list[str]
    sample_ids: np.ndarray | None = None
    cluster_ids: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float32)
        if self.X.ndim != 2:
            raise ValueError("X must be a 2D matrix.")
        n_cells, n_markers = self.X.shape
        if len(self.marker_names) != n_markers:
            raise ValueError("marker_names length must match number of markers.")
        if len(self.cell_ids) != n_cells:
            raise ValueError("cell_ids length must match number of cells.")
        if self.sample_ids is not None and len(self.sample_ids) != n_cells:
            raise ValueError("sample_ids length must match number of cells.")
        if self.cluster_ids is not None and len(self.cluster_ids) != n_cells:
            raise ValueError("cluster_ids length must match number of cells.")


@dataclass
class MarkerScaler:
    """Per-marker scaling state used for explicit preprocessing."""

    mode: str = "none"
    eps: float = 1e-8
    center_: np.ndarray | None = None
    scale_: np.ndarray | None = None

    def fit(self, x: np.ndarray) -> "MarkerScaler":
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("x must be 2D for scaling.")
        if self.mode == "none":
            self.center_ = np.zeros(x.shape[1], dtype=np.float32)
            self.scale_ = np.ones(x.shape[1], dtype=np.float32)
            return self

        if self.mode == "zscore":
            center = np.mean(x, axis=0)
            scale = np.std(x, axis=0)
        elif self.mode == "robust_zscore":
            center = np.median(x, axis=0)
            mad = np.median(np.abs(x - center[None, :]), axis=0)
            scale = 1.4826 * mad
        else:
            raise ValueError(f"Unsupported scaling mode: {self.mode}")

        scale = np.maximum(scale, self.eps)
        self.center_ = center.astype(np.float32)
        self.scale_ = scale.astype(np.float32)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Scaler is not fit.")
        return ((np.asarray(x, dtype=np.float32) - self.center_) / self.scale_).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.center_ is None or self.scale_ is None:
            raise RuntimeError("Scaler is not fit.")
        return (np.asarray(x, dtype=np.float32) * self.scale_ + self.center_).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "eps": float(self.eps),
            "center": None if self.center_ is None else self.center_.tolist(),
            "scale": None if self.scale_ is None else self.scale_.tolist(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MarkerScaler":
        scaler = cls(mode=payload["mode"], eps=float(payload.get("eps", 1e-8)))
        if payload.get("center") is not None:
            scaler.center_ = np.asarray(payload["center"], dtype=np.float32)
        if payload.get("scale") is not None:
            scaler.scale_ = np.asarray(payload["scale"], dtype=np.float32)
        return scaler


def extract_matrix(
    *,
    adata: Any | None = None,
    source: str = "X",
    layer: str | None = None,
    obsm_key: str | None = None,
    dataframe: pd.DataFrame | None = None,
    matrix: np.ndarray | None = None,
    marker_names: Sequence[str] | None = None,
    cell_ids: Sequence[str] | None = None,
    sample_ids: Sequence[Any] | None = None,
    cluster_ids: Sequence[Any] | None = None,
    sample_col: str | None = None,
    cluster_col: str | None = None,
) -> DataBundle:
    """Extract cell-by-marker matrix from AnnData, DataFrame, or ndarray."""
    if adata is not None:
        if source == "X":
            x = _to_numpy(adata.X)
        elif source == "layer":
            if not layer:
                raise ValueError("layer must be provided when source='layer'.")
            x = _to_numpy(adata.layers[layer])
        elif source == "obsm":
            if not obsm_key:
                raise ValueError("obsm_key must be provided when source='obsm'.")
            x = _to_numpy(adata.obsm[obsm_key])
        else:
            raise ValueError("source must be one of {'X', 'layer', 'obsm'}.")

        n_markers = x.shape[1]
        if source == "obsm":
            resolved_markers = (
                list(marker_names) if marker_names is not None else [f"feature_{i}" for i in range(n_markers)]
            )
        else:
            resolved_markers = [str(v) for v in getattr(adata, "var_names", [f"marker_{i}" for i in range(n_markers)])]
        resolved_cells = [str(v) for v in getattr(adata, "obs_names", [f"cell_{i}" for i in range(x.shape[0])])]

        resolved_samples = None
        if sample_col is not None and sample_col in adata.obs:
            resolved_samples = adata.obs[sample_col].to_numpy()

        resolved_clusters = None
        if cluster_col is not None and cluster_col in adata.obs:
            resolved_clusters = adata.obs[cluster_col].to_numpy()

        return DataBundle(
            X=x.astype(np.float32),
            marker_names=resolved_markers,
            cell_ids=resolved_cells,
            sample_ids=None if resolved_samples is None else np.asarray(resolved_samples),
            cluster_ids=None if resolved_clusters is None else np.asarray(resolved_clusters),
        )

    if dataframe is not None:
        x = dataframe.to_numpy(dtype=np.float32)
        resolved_markers = [str(c) for c in dataframe.columns]
        resolved_cells = [str(idx) for idx in dataframe.index]
        return DataBundle(
            X=x,
            marker_names=resolved_markers,
            cell_ids=resolved_cells,
            sample_ids=None if sample_ids is None else np.asarray(sample_ids),
            cluster_ids=None if cluster_ids is None else np.asarray(cluster_ids),
        )

    if matrix is not None:
        x = np.asarray(matrix, dtype=np.float32)
        n_cells, n_markers = x.shape
        resolved_markers = [str(v) for v in marker_names] if marker_names is not None else [f"marker_{i}" for i in range(n_markers)]
        resolved_cells = [str(v) for v in cell_ids] if cell_ids is not None else [f"cell_{i}" for i in range(n_cells)]
        return DataBundle(
            X=x,
            marker_names=resolved_markers,
            cell_ids=resolved_cells,
            sample_ids=None if sample_ids is None else np.asarray(sample_ids),
            cluster_ids=None if cluster_ids is None else np.asarray(cluster_ids),
        )

    raise ValueError("Provide one of: adata, dataframe, or matrix.")


def balanced_downsample_indices(
    labels: Sequence[Any],
    max_per_group: int,
    random_state: int,
) -> np.ndarray:
    """Balanced random sampling across groups."""
    if max_per_group <= 0:
        raise ValueError("max_per_group must be > 0.")
    labels_arr = np.asarray(labels, dtype=object)
    group_codes, _ = pd.factorize(labels_arr, sort=False)
    rng = np.random.default_rng(random_state)
    chosen: list[np.ndarray] = []
    for code in pd.unique(group_codes):
        idx = np.where(group_codes == code)[0]
        n_take = min(max_per_group, idx.size)
        chosen.append(rng.choice(idx, size=n_take, replace=False))
    if not chosen:
        return np.array([], dtype=np.int64)
    out = np.concatenate(chosen)
    rng.shuffle(out)
    return out


def fit_scaler(
    x: np.ndarray,
    *,
    mode: str = "none",
    sample_ids: Sequence[Any] | None = None,
    balanced_max_per_sample: int | None = None,
    random_state: int = 0,
) -> tuple[MarkerScaler, np.ndarray]:
    """Fit marker scaler with optional balanced subset by sample IDs."""
    x = np.asarray(x, dtype=np.float32)
    fit_idx = np.arange(x.shape[0], dtype=np.int64)
    if sample_ids is not None and balanced_max_per_sample is not None:
        fit_idx = balanced_downsample_indices(sample_ids, balanced_max_per_sample, random_state)
    scaler = MarkerScaler(mode=mode).fit(x[fit_idx])
    return scaler, fit_idx


def preprocess_array(x: np.ndarray, scaler: MarkerScaler | None) -> np.ndarray:
    if scaler is None:
        return np.asarray(x, dtype=np.float32)
    return scaler.transform(x)


def split_train_val_indices(
    n_cells: int,
    val_fraction: float = 0.2,
    seed: int = 0,
    stratify_labels: Sequence[Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Create train/validation split indices."""
    if not (0 < val_fraction < 1):
        raise ValueError("val_fraction must be between 0 and 1.")
    rng = np.random.default_rng(seed)

    if stratify_labels is None:
        all_idx = np.arange(n_cells)
        rng.shuffle(all_idx)
        n_val = max(1, int(round(n_cells * val_fraction)))
        val_idx = np.sort(all_idx[:n_val])
        train_idx = np.sort(all_idx[n_val:])
        return train_idx, val_idx

    labels = np.asarray(stratify_labels, dtype=object)
    if labels.shape[0] != n_cells:
        raise ValueError("stratify_labels length must equal n_cells.")

    group_codes, _ = pd.factorize(labels, sort=False)
    val_chunks: list[np.ndarray] = []
    for code in pd.unique(group_codes):
        idx = np.where(group_codes == code)[0]
        rng.shuffle(idx)
        n_val_group = max(1, int(round(idx.size * val_fraction))) if idx.size > 1 else 0
        val_chunks.append(idx[:n_val_group])
    val_idx = np.sort(np.concatenate([arr for arr in val_chunks if arr.size > 0]))
    train_mask = np.ones(n_cells, dtype=bool)
    train_mask[val_idx] = False
    train_idx = np.where(train_mask)[0]
    return train_idx.astype(np.int64), val_idx.astype(np.int64)
