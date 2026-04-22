from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import product
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    def tqdm(iterable=None, *args, **kwargs):  # type: ignore[no-redef]
        if iterable is None:
            total = kwargs.get("total", 0)
            return range(int(total))
        return iterable

from .losses import reconstruction_loss, total_loss, total_variational_loss
from .model import ArchetypeEmbeddingModel, ProbabilisticArchetypeModel
from .utils import (
    collect_software_versions,
    ensure_dir,
    load_json,
    make_run_id,
    resolve_device,
    save_json,
    set_seed,
    validate_run_config,
)


@dataclass
class RunResult:
    run_id: str
    run_dir: Path
    summary: dict[str, Any]
    history: pd.DataFrame


def build_sweep_configs(param_grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Expand a sweep grid into one config per run."""
    keys = list(param_grid.keys())
    values = [param_grid[key] for key in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


_IDENTITY_IGNORE_KEYS = {
    "run_name",
    "progress_sweep",
    "progress_epoch",
    "print_every",
    "skip_existing_runs",
    "resolved_device",
    "software_versions",
}


def _config_identity_payload(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: _to_jsonable(v) for k, v in cfg.items() if k not in _IDENTITY_IGNORE_KEYS}


def _config_fingerprint(cfg: dict[str, Any]) -> str:
    payload_json = json.dumps(_config_identity_payload(cfg), sort_keys=True, separators=(",", ":"))
    fp = hashlib.sha1(payload_json.encode("utf-8")).hexdigest()[:10]
    return fp


def _stable_run_id(run_cfg: dict[str, Any]) -> str:
    fp = _config_fingerprint(run_cfg)
    model_type = str(run_cfg.get("model_type", "deterministic"))
    return f"run_{model_type}_{fp}"


def _find_matching_run_dir(out_dir: Path, run_cfg: dict[str, Any]) -> Path | None:
    """Find an existing run directory with equivalent hyperparameters."""
    target_fp = _config_fingerprint(run_cfg)
    preferred = out_dir / _stable_run_id(run_cfg)
    if preferred.exists() and preferred.is_dir():
        return preferred

    for run_dir in sorted([d for d in out_dir.iterdir() if d.is_dir()]):
        config_path = run_dir / "config.json"
        if not config_path.exists():
            continue
        try:
            cfg_existing = load_json(config_path)
        except Exception:
            continue
        if _config_fingerprint(cfg_existing) == target_fp:
            return run_dir
    return None


def _load_saved_flat_summary(run_dir: Path) -> dict[str, Any] | None:
    config_path = run_dir / "config.json"
    summary_path = run_dir / "summary_metrics.json"
    if not (config_path.exists() and summary_path.exists()):
        return None
    try:
        cfg = load_json(config_path)
        sm = load_json(summary_path)
    except Exception:
        return None

    val = sm.get("val", {})
    out: dict[str, Any] = {
        "run_id": sm.get("run_id", run_dir.name),
        "run_dir": str(run_dir.resolve()),
        "model_type": str(cfg.get("model_type", "deterministic")),
        "decoder_type": str(cfg.get("decoder_type", "factorized")),
        "logit_normalizer": str(cfg.get("logit_normalizer", "softmax")),
        "entmax_alpha": float(cfg.get("entmax_alpha", 1.5)),
        "use_residual_latent": bool(cfg.get("use_residual_latent", False)),
        "beta_w": float(cfg.get("beta_w", 0.0)),
        "beta_r": float(cfg.get("beta_r", 0.0)),
        "residual_dim": int(cfg.get("residual_dim", cfg.get("d", 0))),
        "K": int(cfg.get("K", 0)),
        "d": int(cfg.get("d", 0)),
        "hidden_dims": "-".join(str(v) for v in cfg.get("hidden_dims", [])),
        "lr": float(cfg.get("lr", np.nan)),
        "batch_size": int(cfg.get("batch_size", 0)),
        "recon_loss_type": str(cfg.get("recon_loss_type", "mse")),
        "lambda_entropy": float(cfg.get("lambda_entropy", 0.0)),
        "lambda_sep": float(cfg.get("lambda_sep", 0.0)),
        "lambda_balance": float(cfg.get("lambda_balance", 0.0)),
        "tau": float(cfg.get("tau", 1.0)),
        "patience": int(cfg.get("patience", 0)),
        "best_epoch": int(sm.get("best_epoch", 0)),
        "stopped_early": bool(sm.get("stopped_early", False)),
        "val_recon": float(sm.get("best_val_recon", sm.get("final_val_recon", np.nan))),
        "train_loss": float(sm.get("final_train_loss", np.nan)),
        "mean_marker_corr_val": float(val.get("mean_marker_corr", np.nan)),
        "mean_entropy_val": float(val.get("mean_weight_entropy", np.nan)),
        "usage_std_val": float(val.get("usage_std", np.nan)),
        "dead_archetypes_val": int(val.get("dead_archetypes_lt_1pct", 0)),
        "dominant_frac_gt_0_5_val": float(val.get("dominant_frac_gt_0_5", np.nan)),
        "dominant_frac_gt_0_8_val": float(val.get("dominant_frac_gt_0_8", np.nan)),
    }
    if "final_kl_w" in sm:
        out["final_kl_w"] = float(sm["final_kl_w"])
    if "final_kl_r" in sm:
        out["final_kl_r"] = float(sm["final_kl_r"])
    return out


def _batch_loader(x: np.ndarray, batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    tensor = torch.from_numpy(np.asarray(x, dtype=np.float32))
    ds = TensorDataset(tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=False)


def _safe_pearson(x: np.ndarray, y: np.ndarray, eps: float = 1e-8) -> float:
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denom = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    if denom < eps:
        return 0.0
    return float(np.sum(x_centered * y_centered) / denom)


def _resolve_beta_value(base_beta: float, warmup_epochs: int, epoch: int) -> float:
    if warmup_epochs <= 0:
        return float(base_beta)
    frac = min(1.0, float(epoch) / float(warmup_epochs))
    return float(base_beta) * frac


def _build_model(num_markers: int, run_config: dict[str, Any]) -> nn.Module:
    model_type = str(run_config["model_type"])
    if model_type == "probabilistic":
        return ProbabilisticArchetypeModel(
            num_markers=num_markers,
            num_archetypes=int(run_config["K"]),
            latent_dim=int(run_config["d"]),
            hidden_dims=tuple(run_config["hidden_dims"]),
            tau=float(run_config["tau"]),
            logit_normalizer=str(run_config.get("logit_normalizer", "softmax")),
            entmax_alpha=float(run_config.get("entmax_alpha", 1.5)),
            decoder_type=str(run_config["decoder_type"]),
            use_residual_latent=bool(run_config.get("use_residual_latent", False)),
            residual_dim=int(run_config.get("residual_dim", run_config["d"])),
            dropout=float(run_config.get("dropout", 0.0)),
            logvar_min=float(run_config.get("logvar_min", -10.0)),
            logvar_max=float(run_config.get("logvar_max", 10.0)),
            logvar_init_bias=float(run_config.get("logvar_init_bias", -3.0)),
        )
    return ArchetypeEmbeddingModel(
        num_markers=num_markers,
        num_archetypes=int(run_config["K"]),
        latent_dim=int(run_config["d"]),
        hidden_dims=tuple(run_config["hidden_dims"]),
        tau=float(run_config["tau"]),
        logit_normalizer=str(run_config.get("logit_normalizer", "softmax")),
        entmax_alpha=float(run_config.get("entmax_alpha", 1.5)),
        decoder_type=str(run_config["decoder_type"]),
        dropout=float(run_config.get("dropout", 0.0)),
    )


def _predict_batches(
    model: nn.Module,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
    *,
    model_type: str,
    tau_override: float | None = None,
    prob_eval_mode: str = "mean",
    prob_eval_samples: int = 1,
) -> dict[str, np.ndarray]:
    model.eval()
    blocks: dict[str, list[np.ndarray]] = {}
    a_hat_cache: np.ndarray | None = None

    def _append(key: str, tensor: torch.Tensor | None) -> None:
        if tensor is None:
            return
        blocks.setdefault(key, []).append(tensor.detach().cpu().numpy())

    with torch.no_grad():
        loader = _batch_loader(x, batch_size=batch_size, shuffle=False, num_workers=0)
        for (batch,) in loader:
            batch = batch.to(device)

            if model_type == "deterministic":
                out = model(batch, tau_override=tau_override)  # type: ignore[misc]
                _append("X_hat", out["X_hat"])
                _append("W", out["W"])
                _append("U", out["U"])
                if a_hat_cache is None and out.get("A_hat") is not None:
                    a_hat_cache = out["A_hat"].detach().cpu().numpy()
                continue

            mode = prob_eval_mode.lower()
            if mode == "mean":
                out = model(batch, tau_override=tau_override, sample=False, use_posterior_mean=True)  # type: ignore[misc]
                _append("X_hat", out["X_hat"])
                _append("W", out["W"])
                _append("W_mean", out["W_mean"])
                _append("U", out["U"])
                _append("mu_w", out["mu_w"])
                _append("logvar_w", out["logvar_w"])
                _append("mu_r", out.get("mu_r"))
                _append("logvar_r", out.get("logvar_r"))
            elif mode == "sample":
                out = model(batch, tau_override=tau_override, sample=True, use_posterior_mean=False)  # type: ignore[misc]
                _append("X_hat", out["X_hat"])
                _append("W", out["W"])
                _append("W_mean", out["W_mean"])
                _append("U", out["U"])
                _append("mu_w", out["mu_w"])
                _append("logvar_w", out["logvar_w"])
                _append("R", out.get("r_sample"))
                _append("mu_r", out.get("mu_r"))
                _append("logvar_r", out.get("logvar_r"))
            elif mode == "mc":
                x_hat_samples: list[torch.Tensor] = []
                w_samples: list[torch.Tensor] = []
                u_samples: list[torch.Tensor] = []
                r_samples: list[torch.Tensor] = []
                base_out = None
                for _ in range(max(1, int(prob_eval_samples))):
                    mc_out = model(batch, tau_override=tau_override, sample=True, use_posterior_mean=False)  # type: ignore[misc]
                    base_out = mc_out
                    x_hat_samples.append(mc_out["X_hat"])
                    w_samples.append(mc_out["W"])
                    u_samples.append(mc_out["U"])
                    if mc_out.get("r_sample") is not None:
                        r_samples.append(mc_out["r_sample"])
                x_hat_mc = torch.stack(x_hat_samples, dim=0).mean(dim=0)
                w_mc = torch.stack(w_samples, dim=0).mean(dim=0)
                u_mc = torch.stack(u_samples, dim=0).mean(dim=0)
                _append("X_hat", x_hat_mc)
                _append("W", w_mc)
                _append("U", u_mc)
                if r_samples:
                    _append("R", torch.stack(r_samples, dim=0).mean(dim=0))
                if base_out is not None:
                    _append("W_mean", base_out.get("W_mean"))
                    _append("mu_w", base_out.get("mu_w"))
                    _append("logvar_w", base_out.get("logvar_w"))
                    _append("mu_r", base_out.get("mu_r"))
                    _append("logvar_r", base_out.get("logvar_r"))
            else:
                raise ValueError("prob_eval_mode must be one of {'mean', 'sample', 'mc'}.")

            if a_hat_cache is None:
                out_for_a = model(batch[:1], tau_override=tau_override, sample=False, use_posterior_mean=True)  # type: ignore[misc]
                if out_for_a.get("A_hat") is not None:
                    a_hat_cache = out_for_a["A_hat"].detach().cpu().numpy()

    arrays: dict[str, np.ndarray] = {}
    for key, value_list in blocks.items():
        arrays[key] = np.concatenate(value_list, axis=0)
    if a_hat_cache is not None:
        arrays["A_hat"] = a_hat_cache
    return arrays


def _reconstruction_loss_numpy(
    x_hat: np.ndarray,
    x: np.ndarray,
    *,
    loss_type: str,
    huber_delta: float,
) -> float:
    diff = x_hat - x
    if loss_type == "mse":
        return float(np.mean(diff**2))
    if loss_type == "huber":
        abs_diff = np.abs(diff)
        quadratic = np.minimum(abs_diff, huber_delta)
        linear = abs_diff - quadratic
        return float(np.mean(0.5 * quadratic**2 + huber_delta * linear))
    raise ValueError(f"Unsupported loss_type: {loss_type}")


def _validation_recon_loss(
    model: nn.Module,
    x_val: np.ndarray,
    device: torch.device,
    batch_size: int,
    recon_loss_type: str,
    huber_delta: float,
    *,
    model_type: str,
    tau_override: float | None = None,
    prob_eval_mode: str = "mean",
    prob_eval_samples: int = 1,
) -> float:
    val_eval = _predict_batches(
        model,
        x_val,
        device=device,
        batch_size=batch_size,
        model_type=model_type,
        tau_override=tau_override,
        prob_eval_mode=prob_eval_mode,
        prob_eval_samples=prob_eval_samples,
    )
    return _reconstruction_loss_numpy(
        val_eval["X_hat"],
        np.asarray(x_val, dtype=np.float32),
        loss_type=recon_loss_type,
        huber_delta=huber_delta,
    )


def _metric_summary(x: np.ndarray, x_hat: np.ndarray, w: np.ndarray) -> dict[str, Any]:
    residual = x_hat - x
    mse = float(np.mean(residual**2))
    mae = float(np.mean(np.abs(residual)))
    marker_corr = np.array([_safe_pearson(x[:, m], x_hat[:, m]) for m in range(x.shape[1])], dtype=np.float32)
    entropy = -np.sum(w * np.log(w + 1e-8), axis=1)
    w_bar = w.mean(axis=0)
    dominant = w.max(axis=1)
    summary = {
        "recon_mse": mse,
        "recon_mae": mae,
        "mean_marker_corr": float(np.mean(marker_corr)),
        "median_marker_corr": float(np.median(marker_corr)),
        "mean_weight_entropy": float(np.mean(entropy)),
        "usage_min": float(np.min(w_bar)),
        "usage_max": float(np.max(w_bar)),
        "usage_std": float(np.std(w_bar)),
        "usage_entropy": float(-np.sum(w_bar * np.log(w_bar + 1e-8))),
        "dominant_frac_gt_0_5": float(np.mean(dominant > 0.5)),
        "dominant_frac_gt_0_8": float(np.mean(dominant > 0.8)),
        "dead_archetypes_lt_1pct": int(np.sum(w_bar < 0.01)),
    }
    return {"summary": summary, "per_marker_corr": marker_corr, "entropy": entropy, "usage": w_bar}


def _save_outputs(
    run_dir: Path,
    config: dict[str, Any],
    history_df: pd.DataFrame,
    summary_metrics: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    marker_names: list[str],
    cell_ids: list[str],
    sample_ids: np.ndarray | None,
    cluster_ids: np.ndarray | None,
    split_indices: dict[str, np.ndarray],
    scaler_state: dict[str, Any] | None,
    model_state_dict: dict[str, torch.Tensor],
) -> None:
    ensure_dir(run_dir)
    torch.save(model_state_dict, run_dir / "model_state.pt")
    save_json(run_dir / "config.json", config)
    save_json(run_dir / "summary_metrics.json", summary_metrics)
    history_df.to_csv(run_dir / "history.csv", index=False)

    for name, value in arrays.items():
        if isinstance(value, np.ndarray):
            np.save(run_dir / f"{name}.npy", value)

    pd.DataFrame({"marker_name": marker_names}).to_csv(run_dir / "marker_names.csv", index=False)
    pd.DataFrame({"cell_id": cell_ids}).to_csv(run_dir / "cell_ids.csv", index=False)
    if sample_ids is not None:
        pd.DataFrame({"sample_id": sample_ids}).to_csv(run_dir / "sample_ids.csv", index=False)
    if cluster_ids is not None:
        pd.DataFrame({"cluster_id": cluster_ids}).to_csv(run_dir / "cluster_ids.csv", index=False)

    pd.DataFrame({"train_idx": split_indices["train_idx"]}).to_csv(run_dir / "train_idx.csv", index=False)
    pd.DataFrame({"val_idx": split_indices["val_idx"]}).to_csv(run_dir / "val_idx.csv", index=False)
    if scaler_state is not None:
        save_json(run_dir / "scaler.json", scaler_state)

    if "val" in summary_metrics and "per_marker_corr" in summary_metrics["val"]:
        pd.DataFrame(
            {"marker_name": marker_names, "pearson_r": summary_metrics["val"]["per_marker_corr"]}
        ).to_csv(run_dir / "per_marker_corr.csv", index=False)


def train_one_run(
    *,
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_full: np.ndarray,
    marker_names: list[str],
    cell_ids: list[str],
    run_config: dict[str, Any],
    output_root: str | Path,
    run_id: str | None = None,
    sample_ids: np.ndarray | None = None,
    cluster_ids: np.ndarray | None = None,
    train_idx: np.ndarray | None = None,
    val_idx: np.ndarray | None = None,
    scaler_state: dict[str, Any] | None = None,
) -> RunResult:
    """Train one configuration and export arrays/metrics for notebook analysis."""
    run_config = validate_run_config(run_config)
    seed = int(run_config.get("seed", 0))
    set_seed(seed, deterministic=bool(run_config.get("deterministic", True)))
    device = resolve_device(run_config.get("device", "auto"))
    model_type = str(run_config.get("model_type", "deterministic"))

    run_id = run_id or make_run_id(f"run_{model_type}")
    run_dir = ensure_dir(Path(output_root) / run_id)
    device_msg = f"[{run_id}] Using device: {device}"
    if hasattr(tqdm, "write"):
        tqdm.write(device_msg)
    else:
        print(device_msg)

    model = _build_model(num_markers=x_full.shape[1], run_config=run_config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(run_config["lr"]),
        weight_decay=float(run_config.get("weight_decay", 0.0)),
    )

    epochs = int(run_config["epochs"])
    batch_size = int(run_config["batch_size"])
    recon_loss_type = str(run_config["recon_loss_type"])
    huber_delta = float(run_config.get("huber_delta", 1.0))
    lambda_entropy = float(run_config.get("lambda_entropy", 0.0))
    lambda_sep = float(run_config.get("lambda_sep", 0.0))
    lambda_balance = float(run_config.get("lambda_balance", 0.0))
    separation_mode = str(run_config.get("separation_mode", "cosine_sq"))
    balance_mode = str(run_config.get("balance_mode", "l2_uniform"))
    rbf_gamma = float(run_config.get("rbf_gamma", 1.0))
    tau_override = float(run_config["tau"])
    grad_clip_norm = run_config.get("grad_clip_norm")
    grad_clip_norm = None if grad_clip_norm is None else float(grad_clip_norm)
    print_every = int(run_config.get("print_every", 10))
    progress_epoch = bool(run_config.get("progress_epoch", True))
    early_stopping = bool(run_config.get("early_stopping", True))
    patience = max(1, int(run_config.get("patience", 20)))
    min_delta = float(run_config.get("min_delta", 0.0))
    restore_best_weights = bool(run_config.get("restore_best_weights", True))
    kl_warmup_epochs = int(run_config.get("kl_warmup_epochs", 0))
    beta_w = float(run_config.get("beta_w", 1e-3))
    beta_r = float(run_config.get("beta_r", 1e-3))

    loader = _batch_loader(x_train, batch_size=batch_size, shuffle=True, num_workers=0)
    history_rows: list[dict[str, Any]] = []
    best_val_recon = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    stopped_early = False
    best_state_dict: dict[str, torch.Tensor] | None = None

    prob_eval_mode = str(run_config.get("prob_eval_mode", "mean"))
    prob_eval_samples = int(run_config.get("prob_eval_samples", 1))

    epoch_iter = tqdm(
        range(1, epochs + 1),
        total=epochs,
        desc=f"{run_id} epochs",
        leave=False,
        disable=not progress_epoch,
    )
    for epoch in epoch_iter:
        model.train()
        epoch_accum: dict[str, float] = {"loss": 0.0}
        n_seen = 0
        beta_w_eff = _resolve_beta_value(beta_w, kl_warmup_epochs, epoch)
        beta_r_eff = _resolve_beta_value(beta_r, kl_warmup_epochs, epoch)

        for (batch,) in loader:
            batch = batch.to(device)
            if model_type == "deterministic":
                out = model(batch, tau_override=tau_override)  # type: ignore[misc]
                archetype_matrix = model.archetype_separation_tensor()  # type: ignore[operator]
                loss, parts = total_loss(
                    out["X_hat"],
                    batch,
                    out["W"],
                    archetype_matrix=archetype_matrix,
                    recon_loss_type=recon_loss_type,
                    huber_delta=huber_delta,
                    lambda_entropy=lambda_entropy,
                    lambda_sep=lambda_sep,
                    lambda_balance=lambda_balance,
                    separation_mode=separation_mode,
                    balance_mode=balance_mode,
                    rbf_gamma=rbf_gamma,
                )
            else:
                out = model(batch, tau_override=tau_override, sample=True, use_posterior_mean=False)  # type: ignore[misc]
                archetype_matrix = model.archetype_separation_tensor()  # type: ignore[operator]
                loss, parts = total_variational_loss(
                    out["X_hat"],
                    batch,
                    out["W"],
                    archetype_matrix=archetype_matrix,
                    kl_w=out["kl_w"],
                    kl_r=out.get("kl_r"),
                    beta_w=beta_w_eff,
                    beta_r=beta_r_eff if bool(run_config.get("use_residual_latent", False)) else 0.0,
                    recon_loss_type=recon_loss_type,
                    huber_delta=huber_delta,
                    lambda_entropy=lambda_entropy,
                    lambda_sep=lambda_sep,
                    lambda_balance=lambda_balance,
                    separation_mode=separation_mode,
                    balance_mode=balance_mode,
                    rbf_gamma=rbf_gamma,
                )

            tensors_to_check = {
                "loss": loss,
                "X_hat": out["X_hat"],
                "W": out["W"],
            }
            if "U" in out and out["U"] is not None:
                tensors_to_check["U"] = out["U"]
            if "mu_w" in out and out["mu_w"] is not None:
                tensors_to_check["mu_w"] = out["mu_w"]
            if "logvar_w" in out and out["logvar_w"] is not None:
                tensors_to_check["logvar_w"] = out["logvar_w"]
            if "kl_w" in out and out["kl_w"] is not None:
                tensors_to_check["kl_w"] = out["kl_w"]
            if "kl_r" in out and out["kl_r"] is not None:
                tensors_to_check["kl_r"] = out["kl_r"]
            nonfinite = [name for name, tensor in tensors_to_check.items() if not torch.isfinite(tensor).all()]
            if nonfinite:
                raise FloatingPointError(
                    f"Non-finite tensors encountered during training: {', '.join(nonfinite)} "
                    f"(run_id={run_id}, epoch={epoch}, device={device}, "
                    f"logit_normalizer={run_config.get('logit_normalizer')}, "
                    f"entmax_alpha={run_config.get('entmax_alpha')}, tau={tau_override})"
                )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)

            bad_grad_names: list[str] = []
            for name, param in model.named_parameters():
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    bad_grad_names.append(name)
            if bad_grad_names:
                raise FloatingPointError(
                    f"Non-finite gradients encountered during training: {', '.join(bad_grad_names)} "
                    f"(run_id={run_id}, epoch={epoch}, device={device}, "
                    f"logit_normalizer={run_config.get('logit_normalizer')}, "
                    f"entmax_alpha={run_config.get('entmax_alpha')}, tau={tau_override})"
                )
            optimizer.step()

            bad_param_names: list[str] = []
            for name, param in model.named_parameters():
                if not torch.isfinite(param).all():
                    bad_param_names.append(name)
            if bad_param_names:
                raise FloatingPointError(
                    f"Non-finite model parameters encountered after optimizer step: {', '.join(bad_param_names)} "
                    f"(run_id={run_id}, epoch={epoch}, device={device}, "
                    f"logit_normalizer={run_config.get('logit_normalizer')}, "
                    f"entmax_alpha={run_config.get('entmax_alpha')}, tau={tau_override})"
                )

            bs = int(batch.shape[0])
            n_seen += bs
            epoch_accum["loss"] += float(loss.item()) * bs
            for key, value in parts.items():
                epoch_accum[key] = epoch_accum.get(key, 0.0) + float(value.item()) * bs

        epoch_means = {k: v / max(n_seen, 1) for k, v in epoch_accum.items()}
        if model_type == "probabilistic":
            epoch_means["beta_w_eff"] = beta_w_eff
            epoch_means["beta_r_eff"] = beta_r_eff if bool(run_config.get("use_residual_latent", False)) else 0.0

        val_recon = _validation_recon_loss(
            model,
            x_val=x_val,
            device=device,
            batch_size=max(512, batch_size),
            recon_loss_type=recon_loss_type,
            huber_delta=huber_delta,
            model_type=model_type,
            tau_override=tau_override,
            prob_eval_mode=prob_eval_mode,
            prob_eval_samples=prob_eval_samples,
        )
        row = {"epoch": epoch, **epoch_means, "val_recon": val_recon}
        history_rows.append(row)

        improved = val_recon < (best_val_recon - min_delta)
        if improved:
            best_val_recon = float(val_recon)
            best_epoch = int(epoch)
            epochs_no_improve = 0
            if restore_best_weights:
                best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if progress_epoch and hasattr(epoch_iter, "set_postfix"):
            epoch_iter.set_postfix(
                loss=f"{epoch_means.get('loss', np.nan):.4f}",
                val_recon=f"{val_recon:.4f}",
                best=f"{best_val_recon:.4f}",
            )

        if (not progress_epoch) and (epoch == 1 or epoch % print_every == 0 or epoch == epochs):
            print(
                f"[{run_id}] epoch={epoch:04d} "
                f"loss={epoch_means.get('loss', np.nan):.5f} "
                f"val_recon={val_recon:.5f}"
            )

        if early_stopping and epochs_no_improve >= patience:
            stopped_early = True
            stop_msg = (
                f"[{run_id}] Early stopping at epoch {epoch} "
                f"(patience={patience}, best_epoch={best_epoch}, best_val_recon={best_val_recon:.5f})"
            )
            if hasattr(tqdm, "write"):
                tqdm.write(stop_msg)
            else:
                print(stop_msg)
            break

    history_df = pd.DataFrame(history_rows)
    if restore_best_weights and best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    all_eval = _predict_batches(
        model,
        x_full,
        device=device,
        batch_size=max(1024, batch_size),
        model_type=model_type,
        tau_override=tau_override,
        prob_eval_mode=prob_eval_mode,
        prob_eval_samples=prob_eval_samples,
    )
    val_eval = _predict_batches(
        model,
        x_val,
        device=device,
        batch_size=max(1024, batch_size),
        model_type=model_type,
        tau_override=tau_override,
        prob_eval_mode="mean" if model_type == "probabilistic" else prob_eval_mode,
        prob_eval_samples=prob_eval_samples,
    )
    train_eval = _predict_batches(
        model,
        x_train,
        device=device,
        batch_size=max(1024, batch_size),
        model_type=model_type,
        tau_override=tau_override,
        prob_eval_mode="mean" if model_type == "probabilistic" else prob_eval_mode,
        prob_eval_samples=prob_eval_samples,
    )
    val_w_for_metrics = val_eval["W_mean"] if "W_mean" in val_eval else val_eval["W"]
    train_w_for_metrics = train_eval["W_mean"] if "W_mean" in train_eval else train_eval["W"]
    val_metrics = _metric_summary(x_val, val_eval["X_hat"], val_w_for_metrics)
    train_metrics = _metric_summary(x_train, train_eval["X_hat"], train_w_for_metrics)

    arrays: dict[str, np.ndarray] = {
        "X_hat": all_eval["X_hat"],
        "W": all_eval["W"],
        "U": all_eval["U"],
        "X_observed": np.asarray(x_full, dtype=np.float32),
        "A_hat": all_eval["A_hat"],
        "residuals": all_eval["X_hat"] - np.asarray(x_full, dtype=np.float32),
    }
    if "W_mean" in all_eval:
        arrays["W_mean"] = all_eval["W_mean"]
    if "mu_w" in all_eval:
        arrays["mu_w"] = all_eval["mu_w"]
    if "logvar_w" in all_eval:
        arrays["logvar_w"] = all_eval["logvar_w"]
    if "R" in all_eval:
        arrays["R"] = all_eval["R"]
        arrays["R_norm"] = np.linalg.norm(all_eval["R"], axis=1)
    if "mu_r" in all_eval:
        arrays["mu_r"] = all_eval["mu_r"]
    if "logvar_r" in all_eval:
        arrays["logvar_r"] = all_eval["logvar_r"]

    if model_type == "probabilistic":
        sampled_eval = _predict_batches(
            model,
            x_full,
            device=device,
            batch_size=max(1024, batch_size),
            model_type=model_type,
            tau_override=tau_override,
            prob_eval_mode="sample",
            prob_eval_samples=1,
        )
        arrays["W_sample"] = sampled_eval["W"]
        arrays["U_sample"] = sampled_eval["U"]
        if "R" in sampled_eval:
            arrays["R_sample"] = sampled_eval["R"]
        arrays["X_hat_sample"] = sampled_eval["X_hat"]

    if hasattr(model, "Z") and getattr(model, "Z") is not None:
        arrays["Z"] = model.Z.detach().cpu().numpy()  # type: ignore[attr-defined]
    if hasattr(model, "E") and getattr(model, "E") is not None:
        arrays["E"] = model.E.detach().cpu().numpy()  # type: ignore[attr-defined]
    if hasattr(model, "b") and getattr(model, "b") is not None:
        arrays["b"] = model.b.detach().cpu().numpy()  # type: ignore[attr-defined]
    if hasattr(model, "A") and getattr(model, "A") is not None:
        arrays["A"] = model.A.detach().cpu().numpy()  # type: ignore[attr-defined]
    if hasattr(model, "P_r") and getattr(model, "P_r") is not None:
        arrays["P_r"] = model.P_r.detach().cpu().numpy()  # type: ignore[attr-defined]
    if hasattr(model, "G") and getattr(model, "G") is not None:
        arrays["G"] = model.G.detach().cpu().numpy()  # type: ignore[attr-defined]

    summary_metrics = {
        "run_id": run_id,
        "model_type": model_type,
        "use_residual_latent": bool(run_config.get("use_residual_latent", False)),
        "decoder_type": str(run_config["decoder_type"]),
        "logit_normalizer": str(run_config["logit_normalizer"]),
        "entmax_alpha": float(run_config["entmax_alpha"]),
        "train": {**train_metrics["summary"], "per_marker_corr": train_metrics["per_marker_corr"].tolist()},
        "val": {**val_metrics["summary"], "per_marker_corr": val_metrics["per_marker_corr"].tolist()},
        "best_epoch": int(best_epoch),
        "best_val_recon": float(best_val_recon),
        "stopped_early": bool(stopped_early),
        "final_epoch": int(history_df["epoch"].max()),
        "final_train_loss": float(history_df["loss"].iloc[-1]),
        "final_val_recon": float(history_df["val_recon"].iloc[-1]),
    }
    if "kl_w" in history_df.columns:
        summary_metrics["final_kl_w"] = float(history_df["kl_w"].iloc[-1])
    if "kl_r" in history_df.columns:
        summary_metrics["final_kl_r"] = float(history_df["kl_r"].iloc[-1])

    _save_outputs(
        run_dir=run_dir,
        config={
            **run_config,
            "resolved_device": str(device),
            "software_versions": collect_software_versions(),
        },
        history_df=history_df,
        summary_metrics=summary_metrics,
        arrays=arrays,
        marker_names=marker_names,
        cell_ids=cell_ids,
        sample_ids=sample_ids,
        cluster_ids=cluster_ids,
        split_indices={
            "train_idx": np.arange(x_train.shape[0]) if train_idx is None else train_idx,
            "val_idx": np.arange(x_val.shape[0]) if val_idx is None else val_idx,
        },
        scaler_state=scaler_state,
        model_state_dict=model.state_dict(),
    )

    flat_summary = {
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "model_type": model_type,
        "decoder_type": str(run_config["decoder_type"]),
        "logit_normalizer": str(run_config["logit_normalizer"]),
        "entmax_alpha": float(run_config["entmax_alpha"]),
        "use_residual_latent": bool(run_config.get("use_residual_latent", False)),
        "beta_w": float(run_config.get("beta_w", 0.0)),
        "beta_r": float(run_config.get("beta_r", 0.0)),
        "residual_dim": int(run_config.get("residual_dim", run_config["d"])),
        "K": int(run_config["K"]),
        "d": int(run_config["d"]),
        "hidden_dims": "-".join(str(v) for v in run_config["hidden_dims"]),
        "lr": float(run_config["lr"]),
        "batch_size": int(run_config["batch_size"]),
        "recon_loss_type": recon_loss_type,
        "lambda_entropy": lambda_entropy,
        "lambda_sep": lambda_sep,
        "lambda_balance": lambda_balance,
        "tau": float(run_config["tau"]),
        "patience": int(patience),
        "best_epoch": int(summary_metrics["best_epoch"]),
        "stopped_early": bool(summary_metrics["stopped_early"]),
        "val_recon": float(summary_metrics["best_val_recon"]),
        "train_loss": float(summary_metrics["final_train_loss"]),
        "mean_marker_corr_val": float(summary_metrics["val"]["mean_marker_corr"]),
        "mean_entropy_val": float(summary_metrics["val"]["mean_weight_entropy"]),
        "usage_std_val": float(summary_metrics["val"]["usage_std"]),
        "dead_archetypes_val": int(summary_metrics["val"]["dead_archetypes_lt_1pct"]),
        "dominant_frac_gt_0_5_val": float(summary_metrics["val"]["dominant_frac_gt_0_5"]),
        "dominant_frac_gt_0_8_val": float(summary_metrics["val"]["dominant_frac_gt_0_8"]),
    }
    if "final_kl_w" in summary_metrics:
        flat_summary["final_kl_w"] = float(summary_metrics["final_kl_w"])
    if "final_kl_r" in summary_metrics:
        flat_summary["final_kl_r"] = float(summary_metrics["final_kl_r"])

    return RunResult(run_id=run_id, run_dir=run_dir, summary=flat_summary, history=history_df)


def run_sweep(
    *,
    x: np.ndarray,
    marker_names: list[str],
    cell_ids: list[str],
    output_root: str | Path,
    base_config: dict[str, Any],
    sweep_configs: list[dict[str, Any]],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    sample_ids: np.ndarray | None = None,
    cluster_ids: np.ndarray | None = None,
    scaler_state: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run all hyperparameter configurations sequentially and collect metrics."""
    out_dir = ensure_dir(output_root)
    x = np.asarray(x, dtype=np.float32)
    x_train = x[train_idx]
    x_val = x[val_idx]
    summary_rows: list[dict[str, Any]] = []
    progress_sweep = bool(base_config.get("progress_sweep", True))
    skip_existing_runs = bool(base_config.get("skip_existing_runs", True))

    sweep_iter = tqdm(
        enumerate(sweep_configs, start=1),
        total=len(sweep_configs),
        desc="Sweep",
        leave=True,
        disable=not progress_sweep,
    )
    for run_number, sweep_cfg in sweep_iter:
        run_cfg = validate_run_config({**base_config, **sweep_cfg})
        run_name = run_cfg.get("run_name")
        if run_name:
            run_id = str(run_name)
        else:
            run_id = _stable_run_id(run_cfg=run_cfg)

        matched_existing_dir = _find_matching_run_dir(out_dir, run_cfg) if skip_existing_runs else None
        if matched_existing_dir is not None:
            restored = _load_saved_flat_summary(matched_existing_dir)
            if restored is not None:
                summary_rows.append(restored)
                msg = (
                    "[run_sweep] Skipping existing run with matching hyperparameters: "
                    f"{matched_existing_dir.name}"
                )
                if hasattr(tqdm, "write"):
                    tqdm.write(msg)
                else:
                    print(msg)
                partial_df = (
                    pd.DataFrame(summary_rows).sort_values("val_recon", ascending=True).reset_index(drop=True)
                )
                partial_df.to_csv(out_dir / "sweep_summary.partial.csv", index=False)
                partial_df.to_csv(out_dir / "sweep_summary.csv", index=False)
                continue
            msg = (
                "[run_sweep] Found matching run directory but missing/invalid summary for "
                f"{matched_existing_dir.name}; "
                "skipping to avoid overwrite."
            )
            if hasattr(tqdm, "write"):
                tqdm.write(msg)
            else:
                print(msg)
            continue
        if not progress_sweep:
            print(f"\n=== Sweep run {run_number}/{len(sweep_configs)}: {run_id} ===")

        result = train_one_run(
            x_train=x_train,
            x_val=x_val,
            x_full=x,
            marker_names=marker_names,
            cell_ids=cell_ids,
            run_config=run_cfg,
            output_root=out_dir,
            run_id=run_id,
            sample_ids=sample_ids,
            cluster_ids=cluster_ids,
            train_idx=train_idx,
            val_idx=val_idx,
            scaler_state=scaler_state,
        )
        summary_rows.append(result.summary)
        partial_df = pd.DataFrame(summary_rows).sort_values("val_recon", ascending=True).reset_index(drop=True)
        partial_df.to_csv(out_dir / "sweep_summary.partial.csv", index=False)
        partial_df.to_csv(out_dir / "sweep_summary.csv", index=False)
        if progress_sweep and hasattr(sweep_iter, "set_postfix"):
            sweep_iter.set_postfix(
                run=run_id,
                val_recon=f"{result.summary['val_recon']:.4f}",
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("val_recon", ascending=True).reset_index(drop=True)
    summary_df.to_csv(out_dir / "sweep_summary.csv", index=False)

    save_json(
        out_dir / "sweep_metadata.json",
        {
            "n_runs": len(sweep_configs),
            "base_config": base_config,
            "sweep_configs": sweep_configs,
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
        },
    )
    return summary_df
