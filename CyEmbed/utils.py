from __future__ import annotations

import json
import random
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Set Python / NumPy / Torch RNG seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device: str | None = None) -> torch.device:
    """Resolve torch device with priority CUDA -> MPS -> CPU."""
    def _best_available() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    dev = "auto" if device is None else device.lower()
    if dev in {"auto", "best"}:
        return _best_available()
    if dev == "cuda":
        return torch.device("cuda") if torch.cuda.is_available() else _best_available()
    if dev == "mps":
        return torch.device("mps") if torch.backends.mps.is_available() else _best_available()
    if dev == "cpu":
        return torch.device("cpu")
    return torch.device(dev)


def make_run_id(prefix: str = "run") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


class NumpyJSONEncoder(json.JSONEncoder):
    """JSON encoder that supports NumPy scalar and array types."""

    def default(self, o: Any) -> Any:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        return super().default(o)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, cls=NumpyJSONEncoder)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def flatten_dict(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested dictionaries for tabular summaries."""
    out: dict[str, Any] = {}
    for key, value in values.items():
        flat_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten_dict(value, prefix=flat_key))
        else:
            out[flat_key] = value
    return out


def collect_software_versions() -> dict[str, str]:
    """Collect core package versions for reproducibility logs."""
    versions = {"python": sys.version.split()[0]}
    packages = ["numpy", "pandas", "torch", "matplotlib", "anndata", "scanpy"]
    for pkg in packages:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = "not-installed"
        except Exception:
            versions[pkg] = "unknown"
    return versions


def validate_run_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize train/sweep config fields."""
    cfg = dict(config)
    model_type = str(cfg.get("model_type", "deterministic")).lower()
    if model_type not in {"deterministic", "probabilistic"}:
        raise ValueError("model_type must be 'deterministic' or 'probabilistic'.")
    cfg["model_type"] = model_type

    decoder_type = str(cfg.get("decoder_type", "factorized")).lower()
    if decoder_type not in {"factorized", "direct"}:
        raise ValueError("decoder_type must be 'factorized' or 'direct'.")
    cfg["decoder_type"] = decoder_type

    tau = float(cfg.get("tau", 1.0))
    if tau <= 0:
        raise ValueError("tau must be > 0.")
    cfg["tau"] = tau

    logit_normalizer = str(cfg.get("logit_normalizer", "softmax")).lower()
    if logit_normalizer not in {"softmax", "entmax"}:
        raise ValueError("logit_normalizer must be 'softmax' or 'entmax'.")
    cfg["logit_normalizer"] = logit_normalizer

    entmax_alpha = float(cfg.get("entmax_alpha", 1.5))
    if not 1.0 <= entmax_alpha <= 2.0:
        raise ValueError("entmax_alpha must be in [1.0, 2.0].")
    cfg["entmax_alpha"] = entmax_alpha
    cfg["simplex_impl_version"] = 3 if logit_normalizer == "entmax" else 1

    if model_type == "probabilistic":
        cfg["use_residual_latent"] = bool(cfg.get("use_residual_latent", False))
        cfg["residual_dim"] = int(cfg.get("residual_dim", cfg.get("d", 8)))
        cfg["beta_w"] = float(cfg.get("beta_w", 1e-3))
        cfg["beta_r"] = float(cfg.get("beta_r", 1e-3))
        cfg["kl_warmup_epochs"] = int(cfg.get("kl_warmup_epochs", 0))
        cfg["prob_eval_mode"] = str(cfg.get("prob_eval_mode", "mean")).lower()
        cfg["prob_eval_samples"] = int(cfg.get("prob_eval_samples", 1))
        if cfg["residual_dim"] <= 0:
            raise ValueError("residual_dim must be > 0 when probabilistic model is used.")
        if cfg["prob_eval_mode"] not in {"mean", "sample", "mc"}:
            raise ValueError("prob_eval_mode must be one of {'mean', 'sample', 'mc'}.")
        if cfg["prob_eval_samples"] <= 0:
            raise ValueError("prob_eval_samples must be >= 1.")
    else:
        cfg["use_residual_latent"] = False

    return cfg
