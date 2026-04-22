from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from itertools import permutations

import numpy as np
import pandas as pd
from PIL import Image as PILImage
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


@dataclass
class BatchData:
    batch_key: str
    batch_label: str
    dose_label: str
    analysis_dir: Path
    run_summary: pd.Series
    condition_mean_weights: pd.DataFrame
    condition_entropy_summary: pd.DataFrame
    archetype_delta: pd.DataFrame
    dominant_delta: pd.DataFrame
    archetype_rankings: pd.DataFrame
    components_mean: pd.DataFrame
    reconstruction_by_condition: pd.DataFrame
    sample_mean_weights: pd.DataFrame
    condition_entropy_tests: pd.DataFrame
    sample_condition_counts: pd.DataFrame
    figures_dir: Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
ANALYSIS_ROOT = PROJECT_ROOT / "Analysis"
REPORT_DIR = OUTPUT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FIG_DIR = REPORT_DIR / "figures"
REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = REPORT_DIR / "pdac_gfppos_chemo_archetype_report.pdf"

REPORT_TITLE = "PDAC GFP+ Chemo Archetype Report"
REPORT_SUBTITLE = (
    "Integrated interpretation of Batch 2 (5 doses), Batch 3 (5 doses), "
    "and Batch 4 (6 doses) CyEmbed analyses"
)

BATCH_CONFIG = {
    "batch2": {
        "label": "Batch 2",
        "dose_label": "5 doses",
        "analysis_dir": ANALYSIS_ROOT / "batch2_unt_tr_5dosages_gfppos_only_analysis",
    },
    "batch3": {
        "label": "Batch 3",
        "dose_label": "5 doses",
        "analysis_dir": ANALYSIS_ROOT / "batch3_unt_tr_5dosages_gfppos_only_analysis",
    },
    "batch4": {
        "label": "Batch 4",
        "dose_label": "6 doses",
        "analysis_dir": ANALYSIS_ROOT / "batch4_unt_tr_6dosages_gfppos_only_analysis",
    },
}

# Shared archetype naming is anchored to the refreshed Batch 2 K=7 solution.
SHARED_ARCHETYPE_LABELS: dict[str, dict[str, str]] = {
    "archetype_0": {
        "label": "Basal-like / KRT5-mesenchymal hybrid",
        "rationale": "KRT5, Vimentin, Pan-KRT, H3K36me3, H4K20me3",
    },
    "archetype_1": {
        "label": "Epithelial-adhesion / acetylated CD24-CD44 hybrid",
        "rationale": "H3K9ac, E-cadherin, CD24, CD44, H3K27ac, H2AK119ub",
    },
    "archetype_2": {
        "label": "Classical epithelial / secretory",
        "rationale": "GAL4, GATA6, TSPAN8, MHC-II, EpCAM",
    },
    "archetype_3": {
        "label": "Mesenchymal-plastic / hybrid EMT",
        "rationale": "S100A4, E-cadherin, Vimentin, CD44, H4K20me3",
    },
    "archetype_4": {
        "label": "Proliferative / chromatin-active",
        "rationale": "H2AK119ub, pRb, H3K4me3, H3K4me1, EZH2",
    },
    "archetype_5": {
        "label": "CXCR4-CD109 receptor-high plasticity state",
        "rationale": "CXCR4, CD109, H3K36me2, H3K9me2, H4K16ac, H3K9me3",
    },
    "archetype_6": {
        "label": "Polycomb-repressed / cycling chromatin-high",
        "rationale": "H3K4me2, H3K4me3, H3K27me3, KI67, H3K9me3, pRb",
    },
}

EXECUTIVE_SUMMARY = [
    "The refreshed analyses now resolve seven archetypes per batch, so all cross-batch comparisons and treatment-shift summaries in this report were rebuilt from scratch against the updated Analysis exports.",
    "The seven-state solutions still organize around familiar PDAC programs, but the corrected runs split the old mixed states into a cleaner epithelial-adhesion hybrid state and a distinct CXCR4-CD109 receptor-high plasticity state.",
    "Cross-batch comparison is performed in shared archetype space using one-to-one marker-program matching to the Batch 2 K=7 solution, rather than by raw archetype number.",
    "The strongest treatment-associated changes differ across batches, but the recurring state families remain basal-like/KRT5, epithelial-adhesion hybrid, classical epithelial, mesenchymal-plastic, proliferative/chromatin-active, receptor-high plasticity, and Polycomb/cycling chromatin states.",
]

PDAC_INTERPRETATION = [
    "The classical epithelial state is defined by GAL4, EpCAM, MHC-II, GATA6, and TSPAN8, consistent with a classical/progenitor-like PDAC program.",
    "The corrected seven-state solutions separate two epithelial-adjacent programs: a classical epithelial/secretory state and an epithelial-adhesion hybrid enriched for E-cadherin, CD24, CD44, and acetylation-associated chromatin marks.",
    "Two plastic states recur across the refreshed runs: a mesenchymal-plastic / hybrid EMT state centered on S100A4, CD44, Vimentin, and E-cadherin, and a receptor-high plasticity state marked by CXCR4 and CD109.",
    "A basal-like / KRT5-mesenchymal hybrid state, a proliferative / chromatin-active state, and a Polycomb-repressed / cycling chromatin-high state remain recognizable, but their treatment behavior should now be interpreted in the seven-state context rather than by legacy six-state labels.",
]


def load_batch_data() -> dict[str, BatchData]:
    out: dict[str, BatchData] = {}
    for key, cfg in BATCH_CONFIG.items():
        base = cfg["analysis_dir"]
        runs_df = pd.read_csv(base / "available_runs_snapshot.csv")
        out[key] = BatchData(
            batch_key=key,
            batch_label=cfg["label"],
            dose_label=cfg["dose_label"],
            analysis_dir=base,
            run_summary=runs_df.iloc[0],
            condition_mean_weights=pd.read_csv(base / "condition_mean_weights.csv"),
            condition_entropy_summary=pd.read_csv(base / "condition_entropy_summary.csv"),
            archetype_delta=pd.read_csv(base / "archetype_treated_minus_untreated.csv"),
            dominant_delta=pd.read_csv(base / "dominant_archetype_treated_minus_untreated.csv"),
            archetype_rankings=pd.read_csv(base / "archetype_marker_rankings.csv"),
            components_mean=pd.read_csv(base / "components_mean_matrix.csv", index_col=0),
            reconstruction_by_condition=pd.read_csv(base / "reconstruction_summary_by_condition.csv"),
            sample_mean_weights=pd.read_csv(base / "by_group" / "sample_name_mean_archetype_weights.csv"),
            condition_entropy_tests=pd.read_csv(base / "condition_entropy_welch_tests.csv"),
            sample_condition_counts=(
                pd.read_csv(base / "aligned_metadata_for_run.csv")
                .groupby(["sample_name", "condition_binary"], dropna=False)
                .size()
                .reset_index(name="n_cells")
            ),
            figures_dir=base / "figures",
        )
    return out


def build_styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=26,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=11,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#444444"),
            spaceAfter=16,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            spaceAfter=8,
            textColor=colors.HexColor("#15395b"),
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            spaceBefore=6,
            spaceAfter=6,
            textColor=colors.HexColor("#1d4f73"),
        ),
        "body": ParagraphStyle(
            "Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12,
            leftIndent=12,
            firstLineIndent=-8,
            spaceAfter=4,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=styles["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#555555"),
            spaceAfter=8,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7,
            leading=9,
            spaceAfter=3,
        ),
    }


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def bullet(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(f"• {text}", style)


def dataframe_to_table(
    df: pd.DataFrame,
    styles: dict[str, ParagraphStyle],
    max_rows: int | None = None,
    max_cols: int | None = None,
    col_renames: dict[str, str] | None = None,
) -> Table:
    display_df = df.copy()
    if col_renames:
        display_df = display_df.rename(columns=col_renames)
    if max_cols is not None:
        display_df = display_df.iloc[:, :max_cols]
    if max_rows is not None:
        display_df = display_df.head(max_rows)

    cols = list(display_df.columns)
    data: list[list[Any]] = [[paragraph(str(c), styles["small"]) for c in cols]]
    for _, row in display_df.iterrows():
        rendered_row: list[Any] = []
        for value in row.tolist():
            if isinstance(value, float):
                if abs(value) >= 1000 or (abs(value) > 0 and abs(value) < 1e-3):
                    text = f"{value:.2e}"
                else:
                    text = f"{value:.4f}"
            else:
                text = str(value)
            rendered_row.append(paragraph(text, styles["small"]))
        data.append(rendered_row)

    table = Table(data, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e8f5")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#10293f")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#999999")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7fbff")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    with PILImage.open(path) as img:
        width_px, height_px = img.size
    width = float(width_px)
    height = float(height_px)
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def build_batch_label_maps(
    alignment_map: dict[str, dict[str, tuple[str, float]]],
    anchor_key: str = "batch2",
) -> dict[str, dict[str, dict[str, str]]]:
    out: dict[str, dict[str, dict[str, str]]] = {anchor_key: {}}
    for arch_key, info in SHARED_ARCHETYPE_LABELS.items():
        out[anchor_key][arch_key] = {
            "label": info["label"],
            "rationale": info["rationale"],
        }

    for batch_key, match_map in alignment_map.items():
        if batch_key == anchor_key:
            continue
        out[batch_key] = {}
        for anchor_arch, (matched_arch, corr) in match_map.items():
            info = SHARED_ARCHETYPE_LABELS.get(anchor_arch, {"label": anchor_arch, "rationale": anchor_arch})
            out[batch_key][matched_arch] = {
                "label": info["label"],
                "rationale": f"{info['rationale']} (matched to {anchor_arch}; corr={corr:.3f})",
            }
    return out


def archetype_label_table(
    batch: BatchData,
    styles: dict[str, ParagraphStyle],
    label_map: dict[str, dict[str, str]],
) -> Table:
    rows = []
    for arch_key in batch.components_mean.index.tolist():
        info = label_map.get(str(arch_key), {"label": str(arch_key), "rationale": "Shared label unavailable"})
        rank_df = batch.archetype_rankings
        pos = (
            rank_df[
                (rank_df["archetype"].astype(str) == str(arch_key).split("_")[1])
                & (rank_df["direction"] == "positive")
            ]
            .sort_values("rank")
            .head(6)
        )
        top_markers = ", ".join(pos["marker"].astype(str).tolist())
        rows.append(
            {
                "archetype": arch_key,
                "assigned_state": info["label"],
                "core_logic": info["rationale"],
                "top_positive_markers": top_markers,
            }
        )
    return dataframe_to_table(pd.DataFrame(rows), styles, max_rows=None, max_cols=None)


def build_cross_batch_alignment(batches: dict[str, BatchData]) -> pd.DataFrame:
    frames = {key: batch.components_mean for key, batch in batches.items()}
    common_markers = sorted(set.intersection(*[set(df.columns) for df in frames.values()]))

    rows = []
    batch_keys = list(frames.keys())
    for i, left_key in enumerate(batch_keys):
        for right_key in batch_keys[i + 1 :]:
            left_df = frames[left_key]
            right_df = frames[right_key]
            best_for_pair = []
            for left_arch in left_df.index:
                best_corr = None
                best_arch = None
                left_vec = left_df.loc[left_arch, common_markers].to_numpy(dtype=float)
                for right_arch in right_df.index:
                    right_vec = right_df.loc[right_arch, common_markers].to_numpy(dtype=float)
                    corr = float(np.corrcoef(left_vec, right_vec)[0, 1])
                    if best_corr is None or corr > best_corr:
                        best_corr = corr
                        best_arch = right_arch
                best_for_pair.append(
                    {
                        "comparison": f"{BATCH_CONFIG[left_key]['label']} vs {BATCH_CONFIG[right_key]['label']}",
                        "left_archetype": left_arch,
                        "right_best_match": best_arch,
                        "marker_program_correlation": best_corr,
                    }
                )
            rows.extend(best_for_pair)
    return pd.DataFrame(rows)


def build_one_to_one_alignment_map(
    batches: dict[str, BatchData],
    anchor_key: str = "batch2",
) -> dict[str, dict[str, tuple[str, float]]]:
    frames = {key: batch.components_mean for key, batch in batches.items()}
    common_markers = sorted(set.intersection(*[set(df.columns) for df in frames.values()]))
    anchor = frames[anchor_key].loc[:, common_markers]
    out: dict[str, dict[str, tuple[str, float]]] = {anchor_key: {}}
    for arch in anchor.index.tolist():
        out[anchor_key][str(arch)] = (str(arch), 1.0)

    for other_key, other_df_raw in frames.items():
        if other_key == anchor_key:
            continue
        other_df = other_df_raw.loc[:, common_markers]
        corr = np.corrcoef(anchor.to_numpy(), other_df.to_numpy())[: anchor.shape[0], anchor.shape[0] :]
        best_score: float | None = None
        best_perm: tuple[int, ...] | None = None
        for perm in permutations(range(other_df.shape[0])):
            score = float(sum(corr[i, perm[i]] for i in range(anchor.shape[0])))
            if best_score is None or score > best_score:
                best_score = score
                best_perm = perm
        if best_perm is None:
            raise RuntimeError(f"Could not compute alignment for {other_key}.")
        out[other_key] = {}
        for i, j in enumerate(best_perm):
            out[other_key][str(anchor.index[i])] = (str(other_df.index[j]), float(corr[i, j]))
    return out


def build_shared_archetype_delta_table(
    batches: dict[str, BatchData],
    anchor_key: str = "batch2",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    alignment_map = build_one_to_one_alignment_map(batches, anchor_key=anchor_key)
    anchor_arches = batches[anchor_key].components_mean.index.tolist()
    delta_lookup = {
        key: batch.archetype_delta.set_index("archetype")["treated_minus_untreated"].to_dict()
        for key, batch in batches.items()
    }

    delta_rows: list[dict[str, Any]] = []
    mapping_rows: list[dict[str, Any]] = []
    for arch in anchor_arches:
        arch_str = str(arch)
        label = SHARED_ARCHETYPE_LABELS.get(arch_str, {}).get("label", arch_str)
        delta_row: dict[str, Any] = {
            "shared_archetype": arch_str,
            "shared_label": label,
        }
        mapping_row: dict[str, Any] = {
            "shared_archetype": arch_str,
            "shared_label": label,
        }
        for batch_key, batch in batches.items():
            matched_arch, corr = alignment_map[batch_key][arch_str]
            batch_col = f"{batch.batch_label} ({batch.dose_label})"
            delta_row[batch_col] = float(delta_lookup[batch_key][matched_arch])
            mapping_row[f"{batch.batch_label}_matched"] = matched_arch
            mapping_row[f"{batch.batch_label}_corr"] = float(corr)
            if corr >= 0.7:
                quality = "high"
            elif corr >= 0.5:
                quality = "moderate"
            else:
                quality = "low"
            mapping_row[f"{batch.batch_label}_confidence"] = quality
        delta_rows.append(delta_row)
        mapping_rows.append(mapping_row)
    return pd.DataFrame(delta_rows), pd.DataFrame(mapping_rows)


def low_confidence_alignment_notes(mapping_df: pd.DataFrame) -> list[str]:
    notes: list[str] = []
    for _, row in mapping_df.iterrows():
        label = str(row["shared_label"])
        shared_arch = str(row["shared_archetype"])
        for batch_label in ["Batch 3", "Batch 4"]:
            conf_col = f"{batch_label}_confidence"
            corr_col = f"{batch_label}_corr"
            match_col = f"{batch_label}_matched"
            if conf_col in row.index and str(row[conf_col]) == "low":
                notes.append(
                    f"{batch_label} mapping for {label} ({shared_arch} -> {row[match_col]}) is low-confidence with marker-program correlation {float(row[corr_col]):.3f}."
                )
    return notes


def build_overview_table(batches: dict[str, BatchData]) -> pd.DataFrame:
    rows = []
    for batch in batches.values():
        run = batch.run_summary
        recon = batch.reconstruction_by_condition.copy()
        treated = recon.loc[recon["condition_binary"] == "Treated"].iloc[0]
        untreated = recon.loc[recon["condition_binary"] == "Untreated"].iloc[0]
        ent = batch.condition_entropy_summary
        ent_t = ent.loc[ent["condition_binary"] == "Treated"].iloc[0]
        ent_u = ent.loc[ent["condition_binary"] == "Untreated"].iloc[0]
        rows.append(
            {
                "batch": batch.batch_label,
                "chemo_dose": batch.dose_label,
                "selected_run": run["run_id"],
                "K": int(run["K"]),
                "latent_d": int(run["d"]),
                "val_recon": float(run["val_recon"]),
                "mean_marker_corr_val": float(run["mean_marker_corr_val"]),
                "treated_entropy": float(ent_t["weight_entropy_mean"]),
                "untreated_entropy": float(ent_u["weight_entropy_mean"]),
                "treated_mse": float(treated["mse_mean"]),
                "untreated_mse": float(untreated["mse_mean"]),
            }
        )
    return pd.DataFrame(rows)


def build_design_and_scope_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scope": "Cell population",
                "value": "GFP+ cells only",
            },
            {
                "scope": "Comparison axes",
                "value": "Sample and condition_binary (Treated vs Untreated)",
            },
            {
                "scope": "Excluded markers",
                "value": "Core histones, GFP, CD45, DNA",
            },
            {
                "scope": "Archetype interpretation",
                "value": "By marker program, not raw archetype index",
            },
            {
                "scope": "Batches",
                "value": "Batch 2 and 3: 5 doses; Batch 4: 6 doses",
            },
        ]
    )


def build_cross_batch_named_delta_table(batches: dict[str, BatchData]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for batch_key, batch in batches.items():
        label_map = build_batch_label_maps(build_one_to_one_alignment_map(batches), anchor_key="batch2")[batch_key]
        for _, row in batch.archetype_delta.iterrows():
            arch_name = str(row["archetype"])
            state = label_map.get(arch_name, {}).get("label", arch_name)
            rows.append(
                {
                    "state": state,
                    "batch": f"{batch.batch_label} ({batch.dose_label})",
                    "treated_minus_untreated": float(row["treated_minus_untreated"]),
                }
            )
    wide = pd.DataFrame(rows).pivot(index="state", columns="batch", values="treated_minus_untreated")
    if wide.empty:
        return wide
    order = (
        pd.DataFrame(rows)
        .assign(abs_delta=lambda df: df["treated_minus_untreated"].abs())
        .groupby("state", as_index=False)["abs_delta"]
        .max()
        .sort_values("abs_delta", ascending=False)["state"]
        .tolist()
    )
    return wide.reindex(order)


def build_cross_batch_raw_delta_table(batches: dict[str, BatchData]) -> pd.DataFrame:
    archetypes = [str(v) for v in batches["batch2"].components_mean.index.tolist()]
    rows: list[dict[str, Any]] = []
    for arch in archetypes:
        row: dict[str, Any] = {"archetype": arch}
        for batch_key, batch in batches.items():
            batch_col = f"{batch.batch_label} ({batch.dose_label})"
            match = batch.archetype_delta.loc[batch.archetype_delta["archetype"] == arch, "treated_minus_untreated"]
            row[batch_col] = float(match.iloc[0]) if len(match) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def render_cross_batch_raw_delta_figure(delta_df: pd.DataFrame) -> Path | None:
    if delta_df.empty:
        return None

    fig_path = REPORT_FIG_DIR / "cross_batch_raw_archetype_deltas_heatmap.png"
    plot_df = delta_df.set_index("archetype")
    values = plot_df.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    vmax = max(vmax, 0.01)

    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="#f3f3f3")
    im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(plot_df.shape[1]))
    ax.set_xticklabels(plot_df.columns.tolist(), fontsize=10)
    ax.set_yticks(np.arange(plot_df.shape[0]))
    ax.set_yticklabels(plot_df.index.tolist(), fontsize=10)
    ax.set_title("Compact cross-batch treated-minus-untreated shifts", fontsize=14, pad=12)

    for i in range(plot_df.shape[0]):
        for j in range(plot_df.shape[1]):
            value = values[i, j]
            if np.isfinite(value):
                text_color = "white" if abs(value) > (0.55 * vmax) else "black"
                ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=9, color=text_color)

    ax.set_xticks(np.arange(-0.5, plot_df.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, plot_df.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Mean archetype-weight shift", rotation=90)

    fig.text(
        0.5,
        0.02,
        "Archetype indices are within-batch identifiers. Use the cross-batch alignment table above to map related states across batches.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.98))
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def render_cross_batch_named_delta_figure(delta_wide: pd.DataFrame) -> Path | None:
    if delta_wide.empty:
        return None

    fig_path = REPORT_FIG_DIR / "cross_batch_named_state_deltas_heatmap.png"
    plot_df = delta_wide.copy()
    plot_df = plot_df.iloc[::-1]

    values = plot_df.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    vmax = max(vmax, 0.01)

    fig_height = max(4.2, 0.55 * plot_df.shape[0] + 1.8)
    fig, ax = plt.subplots(figsize=(9.2, fig_height))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="#f3f3f3")
    im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(plot_df.shape[1]))
    ax.set_xticklabels(plot_df.columns.tolist(), rotation=0, fontsize=10)
    ax.set_yticks(np.arange(plot_df.shape[0]))
    ax.set_yticklabels(plot_df.index.tolist(), fontsize=10)
    ax.set_title("Cross-batch treated-minus-untreated archetype shifts", fontsize=14, pad=12)

    for i in range(plot_df.shape[0]):
        for j in range(plot_df.shape[1]):
            value = values[i, j]
            if np.isfinite(value):
                text_color = "white" if abs(value) > (0.55 * vmax) else "black"
                ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=9, color=text_color)
            else:
                ax.text(j, i, "NA", ha="center", va="center", fontsize=8, color="#666666")

    ax.set_xticks(np.arange(-0.5, plot_df.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, plot_df.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Mean archetype-weight shift", rotation=90)

    fig.text(
        0.5,
        0.01,
        "Positive values indicate enrichment in treated GFP+ cells; negative values indicate depletion.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.98))
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def render_shared_archetype_delta_figure(delta_df: pd.DataFrame) -> Path | None:
    if delta_df.empty:
        return None

    fig_path = REPORT_FIG_DIR / "cross_batch_shared_archetype_deltas_heatmap.png"
    plot_df = delta_df.set_index("shared_label")
    plot_df = plot_df[[c for c in plot_df.columns if c not in {"shared_archetype"}]]
    values = plot_df.to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    vmax = max(vmax, 0.01)

    fig_height = max(4.2, 0.58 * plot_df.shape[0] + 1.9)
    fig, ax = plt.subplots(figsize=(8.6, fig_height))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad(color="#f3f3f3")
    im = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(plot_df.shape[1]))
    ax.set_xticklabels(plot_df.columns.tolist(), fontsize=10)
    ax.set_yticks(np.arange(plot_df.shape[0]))
    ax.set_yticklabels(plot_df.index.tolist(), fontsize=10)
    ax.set_title("Cross-batch shared-archetype treated-minus-untreated shifts", fontsize=14, pad=12)

    for i in range(plot_df.shape[0]):
        for j in range(plot_df.shape[1]):
            value = values[i, j]
            if np.isfinite(value):
                text_color = "white" if abs(value) > (0.55 * vmax) else "black"
                ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=9, color=text_color)

    ax.set_xticks(np.arange(-0.5, plot_df.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, plot_df.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Mean archetype-weight shift", rotation=90)

    fig.text(
        0.5,
        0.02,
        "Rows are shared archetype families defined by one-to-one cross-batch alignment to Batch 2 marker programs.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.98))
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def batch_narrative(
    batch: BatchData,
    label_map: dict[str, dict[str, str]],
) -> list[str]:
    diff_df = batch.archetype_delta.sort_values("treated_minus_untreated", ascending=False).reset_index(drop=True)
    top_gain = diff_df.iloc[0]
    top_loss = diff_df.iloc[-1]
    gain_label = label_map.get(str(top_gain["archetype"]), {}).get("label", str(top_gain["archetype"]))
    loss_label = label_map.get(str(top_loss["archetype"]), {}).get("label", str(top_loss["archetype"]))

    ent = batch.condition_entropy_summary.set_index("condition_binary")
    treated_entropy = float(ent.loc["Treated", "weight_entropy_mean"]) if "Treated" in ent.index else float("nan")
    untreated_entropy = float(ent.loc["Untreated", "weight_entropy_mean"]) if "Untreated" in ent.index else float("nan")

    return [
        f"This refreshed report uses the seven-archetype solution for {batch.batch_label}. The strongest treated enrichment is {gain_label} ({float(top_gain['treated_minus_untreated']):+.3f}), while the strongest depletion is {loss_label} ({float(top_loss['treated_minus_untreated']):+.3f}).",
        f"Mean weight entropy is {treated_entropy:.3f} in treated cells versus {untreated_entropy:.3f} in untreated cells, providing a compact summary of how mixed the archetype usage is in each condition.",
        "Raw archetype IDs remain batch-specific, so the identity table below uses the one-to-one marker-program alignment to translate each batch into the shared seven-state PDAC frame.",
    ]


def add_page_number(canvas: Any, doc: Any) -> None:
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawRightString(doc.pagesize[0] - 0.5 * inch, 0.35 * inch, f"Page {doc.page}")


def build_report() -> None:
    batches = load_batch_data()
    styles = build_styles()
    alignment_map = build_one_to_one_alignment_map(batches, anchor_key="batch2")
    batch_label_maps = build_batch_label_maps(alignment_map, anchor_key="batch2")
    shared_delta_df, shared_mapping_df = build_shared_archetype_delta_table(batches, anchor_key="batch2")
    alignment_notes = low_confidence_alignment_notes(shared_mapping_df)
    shared_delta_df.to_csv(REPORT_DIR / "cross_batch_shared_archetype_deltas.csv", index=False)
    shared_mapping_df.to_csv(REPORT_DIR / "cross_batch_shared_archetype_alignment_map.csv", index=False)
    shared_delta_fig = render_shared_archetype_delta_figure(shared_delta_df)

    story: list[Any] = []
    story.append(Spacer(1, 0.3 * inch))
    story.append(paragraph(REPORT_TITLE, styles["title"]))
    story.append(paragraph(REPORT_SUBTITLE, styles["subtitle"]))
    story.append(paragraph(
        "This report summarizes the refreshed GFP+ CyEmbed analyses for Batch 2 and Batch 3 (5 doses of chemotherapy) and Batch 4 (6 doses). "
        "All three batches now use seven-archetype runs, and cross-batch comparisons are rebuilt from the updated Analysis exports.",
        styles["body"],
    ))
    story.append(Spacer(1, 0.12 * inch))
    story.append(paragraph("Executive Summary", styles["h1"]))
    for item in EXECUTIVE_SUMMARY:
        story.append(bullet(item, styles["bullet"]))

    story.append(Spacer(1, 0.12 * inch))
    story.append(paragraph("Study Overview", styles["h1"]))
    story.append(paragraph(
        "All analyses are restricted to the GFP+ compartment. Batch 2 and Batch 3 were treated with 5 chemotherapy doses; Batch 4 was treated with 6 doses. "
        "Within each batch, the report uses the selected best CyEmbed run from notebook 02 outputs and interprets condition shifts using treated-minus-untreated archetype weight changes, dominant-archetype composition, marker programs, and embedding geometry.",
        styles["body"],
    ))
    story.append(dataframe_to_table(build_design_and_scope_table(), styles))
    story.append(Spacer(1, 0.08 * inch))
    story.append(paragraph("Selected Run Overview", styles["h2"]))
    story.append(dataframe_to_table(build_overview_table(batches), styles))

    story.append(Spacer(1, 0.12 * inch))
    story.append(paragraph("Cross-Batch PDAC Interpretation", styles["h1"]))
    for item in PDAC_INTERPRETATION:
        story.append(bullet(item, styles["bullet"]))

    story.append(Spacer(1, 0.08 * inch))
    story.append(paragraph("Cross-Batch Archetype Alignment", styles["h2"]))
    story.append(paragraph(
        "The table below aligns archetypes across batches by correlation of their archetype marker programs (`components_mean_matrix.csv`). "
        "This is the basis for comparing PDAC states across batches even though archetype IDs differ.",
        styles["body"],
    ))
    story.append(dataframe_to_table(build_cross_batch_alignment(batches), styles, max_rows=None))
    story.append(Spacer(1, 0.08 * inch))
    story.append(paragraph("Cross-Batch Treatment Shift Comparison", styles["h2"]))
    story.append(paragraph(
        "To compare treatment effects in a truly collapsed form, the treated-minus-untreated deltas were remapped into seven shared archetype families using the cross-batch marker-program alignment. "
        "Batch 2 was used as the anchor reference, and Batch 3 and Batch 4 archetypes were matched one-to-one to Batch 2 archetypes by maximizing total marker-program correlation across the seven archetypes.",
        styles["body"],
    ))
    story.append(paragraph("Shared-archetype mapping derived from the alignment table", styles["small"]))
    story.append(dataframe_to_table(shared_mapping_df, styles, max_rows=None))
    if alignment_notes:
        story.append(Spacer(1, 0.03 * inch))
        story.append(paragraph("Low-confidence rematching warnings", styles["small"]))
        for note in alignment_notes:
            story.append(bullet(note, styles["bullet"]))
    story.append(Spacer(1, 0.05 * inch))
    story.append(paragraph("Collapsed treated-minus-untreated deltas in shared archetype space", styles["small"]))
    story.append(dataframe_to_table(shared_delta_df, styles, max_rows=None))
    if shared_delta_fig is not None and shared_delta_fig.exists():
        story.append(Spacer(1, 0.05 * inch))
        story.append(scaled_image(shared_delta_fig, max_width=9.8 * inch, max_height=5.4 * inch))
        story.append(paragraph(
            "Heatmap of treatment-associated changes after collapsing all three batches into seven shared archetype families using the refreshed cross-batch alignment map. Low-confidence matched rows should be interpreted cautiously.",
            styles["caption"],
        ))

    for batch_key, batch in batches.items():
        story.append(PageBreak())
        story.append(paragraph(f"{batch.batch_label}: {batch.dose_label}", styles["h1"]))
        story.append(paragraph(
            f"Selected run: `{batch.run_summary['run_id']}` with K={int(batch.run_summary['K'])}, d={int(batch.run_summary['d'])}, "
            f"validation reconstruction loss={float(batch.run_summary['val_recon']):.4f}, and mean validation marker correlation={float(batch.run_summary['mean_marker_corr_val']):.3f}.",
            styles["body"],
        ))
        for item in batch_narrative(batch, batch_label_maps[batch_key]):
            story.append(bullet(item, styles["bullet"]))

        story.append(Spacer(1, 0.06 * inch))
        story.append(paragraph("Archetype Identity Table", styles["h2"]))
        story.append(archetype_label_table(batch, styles, batch_label_maps[batch_key]))

        story.append(Spacer(1, 0.06 * inch))
        story.append(paragraph("Condition and Treatment-Shift Tables", styles["h2"]))
        story.append(paragraph("Cell counts by sample and condition", styles["small"]))
        story.append(dataframe_to_table(batch.sample_condition_counts, styles, max_rows=None))
        story.append(Spacer(1, 0.05 * inch))
        story.append(paragraph("Mean archetype weights by condition", styles["small"]))
        story.append(dataframe_to_table(batch.condition_mean_weights, styles))
        story.append(Spacer(1, 0.05 * inch))
        story.append(paragraph("Mean archetype weights by sample", styles["small"]))
        story.append(dataframe_to_table(batch.sample_mean_weights, styles, max_rows=None))
        story.append(Spacer(1, 0.05 * inch))
        story.append(paragraph("Treated minus untreated archetype shift", styles["small"]))
        story.append(dataframe_to_table(batch.archetype_delta, styles))
        story.append(Spacer(1, 0.05 * inch))
        story.append(paragraph("Dominant-archetype fraction shift", styles["small"]))
        story.append(dataframe_to_table(batch.dominant_delta, styles))
        story.append(Spacer(1, 0.05 * inch))
        story.append(paragraph("Entropy and dominance summary by condition", styles["small"]))
        story.append(dataframe_to_table(batch.condition_entropy_summary, styles))
        story.append(Spacer(1, 0.05 * inch))
        story.append(paragraph("Entropy Welch tests by condition", styles["small"]))
        story.append(dataframe_to_table(batch.condition_entropy_tests, styles, max_rows=None))
        story.append(Spacer(1, 0.05 * inch))
        story.append(paragraph("Reconstruction summary by condition", styles["small"]))
        story.append(dataframe_to_table(batch.reconstruction_by_condition, styles))

        story.append(PageBreak())
        story.append(paragraph(f"{batch.batch_label} Figures", styles["h1"]))
        figure_specs = [
            (
                batch.figures_dir / "selected_training_curve.png",
                f"{batch.batch_label} training curve for the selected run.",
            ),
            (
                batch.figures_dir / "condition_mean_archetype_weights.png",
                f"{batch.batch_label} condition-level mean archetype weights.",
            ),
            (
                batch.figures_dir / "archetype_condition_means_heatmap.png",
                f"{batch.batch_label} heatmap view of mean archetype weights by condition.",
            ),
            (
                batch.figures_dir / "archetype_treated_minus_untreated_barh.png",
                f"{batch.batch_label} treated-minus-untreated archetype differences.",
            ),
            (
                batch.figures_dir / "weight_entropy_by_condition.png",
                f"{batch.batch_label} entropy distribution by condition. Higher entropy reflects more mixed archetype usage per cell.",
            ),
            (
                batch.figures_dir / "dominant_weight_by_condition.png",
                f"{batch.batch_label} dominant archetype weight by condition. Lower dominant weights indicate more mixed or plastic states.",
            ),
            (
                batch.figures_dir / "dominant_archetype_stackedbars_sample_condition.png",
                f"{batch.batch_label} dominant-archetype composition by sample and condition.",
            ),
            (
                batch.figures_dir / "archetype_marker_programs_zscore.png",
                f"{batch.batch_label} archetype marker programs (row-wise z-score).",
            ),
            (
                batch.figures_dir / "marker_embeddings_pca.png",
                f"{batch.batch_label} marker embedding PCA. Marker neighborhoods should separate epithelial, mesenchymal, proliferative, and chromatin programs.",
            ),
            (
                batch.figures_dir / "marker_embeddings_umap.png",
                f"{batch.batch_label} marker embedding UMAP. Marker neighborhoods help validate epithelial, mesenchymal, proliferative, and chromatin programs.",
            ),
            (
                batch.figures_dir / "marker_embedding_cosine_similarity_heatmap.png",
                f"{batch.batch_label} cosine-similarity heatmap of marker embeddings, highlighting co-embedded marker neighborhoods.",
            ),
        ]
        for fig_path, caption in figure_specs:
            if fig_path.exists():
                story.append(scaled_image(fig_path, max_width=9.5 * inch, max_height=5.8 * inch))
                story.append(paragraph(caption, styles["caption"]))

    story.append(PageBreak())
    story.append(paragraph("Integrated Conclusion", styles["h1"]))
    conclusion_points = [
        "Across the refreshed K=7 analyses, the GFP+ PDAC compartment is best described as a therapy-remodeled mixture of seven recurring state families: basal-like/KRT5 hybrid, epithelial-adhesion hybrid, classical epithelial, mesenchymal-plastic, proliferative/chromatin-active, CXCR4-CD109 receptor-high plasticity, and Polycomb/cycling chromatin states.",
        "The extra archetype resolves structure that was partially compressed in the earlier six-state report, so the new cross-batch rematching should be treated as the current reference analysis.",
        "Treatment does not collapse the GFP+ compartment into one resistant endpoint. Instead, the post-treatment pool remains a composite of epithelial survivors, proliferative or chromatin-active cells, and multiple plastic mesenchymal or hybrid states.",
        "The most defensible cross-batch read is therefore residual-state pluralism under chemotherapy, interpreted in shared seven-archetype space rather than by legacy raw archetype numbering.",
    ]
    for item in conclusion_points:
        story.append(bullet(item, styles["bullet"]))

    doc = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=landscape(letter),
        leftMargin=0.45 * inch,
        rightMargin=0.45 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        title=REPORT_TITLE,
        author="Codex",
    )
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


if __name__ == "__main__":
    build_report()
    print(f"Wrote {REPORT_PATH}")
