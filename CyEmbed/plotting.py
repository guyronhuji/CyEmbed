from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_training_history(history_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_df["epoch"], history_df["loss"], label="train_total")
    axes[0].plot(history_df["epoch"], history_df["recon"], label="train_recon")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["val_recon"], color="tab:orange", label="val_recon")
    axes[1].set_title("Validation Reconstruction")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    plt.tight_layout()


def plot_matrix_heatmap(
    matrix: np.ndarray,
    row_labels: Sequence[str] | None = None,
    col_labels: Sequence[str] | None = None,
    title: str = "",
    cmap: str = "viridis",
    figsize: tuple[int, int] = (10, 5),
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    plot_df = pd.DataFrame(matrix)
    if row_labels is not None:
        plot_df.index = list(row_labels)
    if col_labels is not None:
        plot_df.columns = list(col_labels)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        plot_df,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        ax=ax,
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title(title)
    ax.tick_params(axis="x", labelrotation=90)
    ax.tick_params(axis="y", labelrotation=0)
    plt.tight_layout()


def plot_clustermap(
    matrix: np.ndarray,
    row_labels: Sequence[str] | None = None,
    col_labels: Sequence[str] | None = None,
    title: str = "",
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[int, int] = (8, 8),
    metric: str = "euclidean",
    method: str = "average",
) -> None:
    plot_df = pd.DataFrame(matrix)
    if row_labels is not None:
        plot_df.index = list(row_labels)
    if col_labels is not None:
        plot_df.columns = list(col_labels)

    g = sns.clustermap(
        plot_df,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        metric=metric,
        method=method,
        figsize=figsize,
        xticklabels=True,
        yticklabels=True,
    )
    g.ax_heatmap.set_title(title)
    g.ax_heatmap.tick_params(axis="x", labelrotation=90)
    g.ax_heatmap.tick_params(axis="y", labelrotation=0)
    plt.show()


def plot_observed_vs_reconstructed(
    x: np.ndarray,
    x_hat: np.ndarray,
    marker_names: list[str],
    markers: list[str] | None = None,
    max_points: int = 5000,
    random_state: int = 0,
) -> None:
    if markers is None:
        markers = marker_names[: min(6, len(marker_names))]
    idx_map = {m: i for i, m in enumerate(marker_names)}
    rng = np.random.default_rng(random_state)
    n = x.shape[0]
    take = np.arange(n) if n <= max_points else np.sort(rng.choice(n, size=max_points, replace=False))

    n_cols = min(3, len(markers))
    n_rows = int(np.ceil(len(markers) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for i, marker in enumerate(markers):
        m = idx_map[marker]
        axes[i].scatter(x[take, m], x_hat[take, m], s=4, alpha=0.3)
        mn = min(float(np.min(x[take, m])), float(np.min(x_hat[take, m])))
        mx = max(float(np.max(x[take, m])), float(np.max(x_hat[take, m])))
        axes[i].plot([mn, mx], [mn, mx], linestyle="--", linewidth=1, color="black")
        axes[i].set_title(marker)
        axes[i].set_xlabel("Observed")
        axes[i].set_ylabel("Reconstructed")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()


def plot_weight_histograms(w: np.ndarray, bins: int = 40) -> None:
    k = w.shape[1]
    n_cols = min(4, k)
    n_rows = int(np.ceil(k / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.5 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for i in range(k):
        sns.histplot(w[:, i], bins=bins, ax=axes[i], color="tab:blue", alpha=0.8)
        axes[i].set_title(f"W[{i}]")
        axes[i].set_xlim(0, 1)
    for j in range(k, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()


def plot_umap_overlay(
    umap_xy: np.ndarray,
    values: np.ndarray,
    title: str,
    cmap: str = "viridis",
    s: float = 3.0,
) -> None:
    fig, ax = plt.subplots(figsize=(5, 4.2))
    sc = ax.scatter(umap_xy[:, 0], umap_xy[:, 1], c=values, s=s, cmap=cmap, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()


def plot_umap_categorical(
    umap_xy: np.ndarray,
    labels: np.ndarray,
    title: str,
    s: float = 3.0,
) -> None:
    unique_labels = np.unique(labels)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    cmap = plt.get_cmap("tab20", len(unique_labels))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(umap_xy[mask, 0], umap_xy[mask, 1], s=s, color=cmap(i), label=str(label), linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(markerscale=3, bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    plt.tight_layout()


def plot_embedding_scatter(
    xy: np.ndarray,
    labels: list[str] | None = None,
    title: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(xy[:, 0], xy[:, 1], s=40, alpha=0.9)
    if labels is not None:
        for i, label in enumerate(labels):
            ax.text(xy[i, 0], xy[i, 1], str(label), fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    plt.tight_layout()
