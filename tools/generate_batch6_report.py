from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


@dataclass
class Batch6Data:
    analysis_dir: Path
    figures_dir: Path
    run_summary: pd.Series
    condition_mean_weights: pd.DataFrame
    condition_entropy_summary: pd.DataFrame
    pairwise_differences: pd.DataFrame
    dominant_condition_fraction: pd.DataFrame
    archetype_rankings: pd.DataFrame
    embedding_dim_top_markers: pd.DataFrame
    reconstruction_by_condition: pd.DataFrame
    sample_condition_counts: pd.DataFrame


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = PROJECT_ROOT / "Analysis" / "batch6_unt_gemr_vitro_vivo_gfppos_only_analysis"
FIG_DIR = ANALYSIS_DIR / "figures"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = REPORT_DIR / "batch6_gemr_gfppos_report.pdf"

REPORT_TITLE = "Batch 6 GFP+ Gemcitabine Resistance Report"
REPORT_SUBTITLE = "Integrated interpretation of untreated, Gem-R in vitro, and Gem-R in vivo CyEmbed analysis"

ARCHETYPE_LABELS = {
    "archetype_0": {
        "label": "Adhesive epithelial-plastic hybrid",
        "rationale": "E-cadherin, CD24, CD44, TSPAN8, S100A4",
    },
    "archetype_1": {
        "label": "Mesenchymal / chromatin-remodeled plastic state",
        "rationale": "MBD, Vimentin, H3K9ac, Pan-KRT, H4K20me3",
    },
    "archetype_2": {
        "label": "Basal-like keratinized plastic state",
        "rationale": "TSPAN8, Vimentin, H3K36me3, KRT5, KRT17, EZH2",
    },
    "archetype_3": {
        "label": "Chromatin-high / epigenetic state",
        "rationale": "H3K4me2, H3K27ac, H3K9me2, H3K27me3, H3K36me2",
    },
    "archetype_4": {
        "label": "Classical epithelial / progenitor-like state",
        "rationale": "GAL4, EpCAM, CD133, MHC-II, TSPAN8, GATA6",
    },
    "archetype_5": {
        "label": "Plastic survivor state",
        "rationale": "S100A4, CD44 with loss of Pan-KRT, MBD, Vimentin, GAL4",
    },
}

EXECUTIVE_SUMMARY = [
    "Batch 6 resolves six archetypes and separates the Gem-resistant cells into distinct in vitro and in vivo residual-state programs rather than a single resistant endpoint.",
    "Both Gem-R states move away from the untreated adhesive epithelial-plastic hybrid state and gain a shared plastic survivor component.",
    "Gem-R in vitro is more strongly enriched for a mesenchymal / chromatin-remodeled plastic state, whereas Gem-R in vivo is more strongly enriched for a classical epithelial / progenitor-like state.",
    "These results support the idea that gemcitabine resistance is context-dependent: one component is shared across resistance settings, but the microenvironment appears to steer the in vivo resistant cells toward a different surviving program than the in vitro resistant cells.",
]

BIOLOGICAL_READOUT = [
    "The untreated GFP+ compartment is relatively enriched for an adhesive epithelial-plastic hybrid state and less enriched for the plastic survivor state than either resistant condition.",
    "Gem-R in vitro preferentially expands a mesenchymal / chromatin-remodeled state together with the shared plastic survivor state, consistent with a more cell-autonomous adaptation to drug exposure.",
    "Gem-R in vivo preferentially expands a classical epithelial / progenitor-like program together with the shared plastic survivor state, suggesting that resistance in tissue preserves or re-establishes epithelial lineage identity.",
    "The in vivo and in vitro resistant cells therefore appear related but not equivalent: both are resistant, but they occupy different ecological solutions to that resistance.",
]


def load_batch6_data() -> Batch6Data:
    runs = pd.read_csv(ANALYSIS_DIR / "available_runs_snapshot.csv")
    aligned_meta = pd.read_csv(ANALYSIS_DIR / "aligned_metadata_for_run.csv")
    return Batch6Data(
        analysis_dir=ANALYSIS_DIR,
        figures_dir=FIG_DIR,
        run_summary=runs.iloc[0],
        condition_mean_weights=pd.read_csv(ANALYSIS_DIR / "condition_mean_weights.csv"),
        condition_entropy_summary=pd.read_csv(ANALYSIS_DIR / "condition_entropy_summary.csv"),
        pairwise_differences=pd.read_csv(ANALYSIS_DIR / "archetype_pairwise_condition_differences.csv"),
        dominant_condition_fraction=pd.read_csv(ANALYSIS_DIR / "dominant_archetype_per_condition_fraction.csv"),
        archetype_rankings=pd.read_csv(ANALYSIS_DIR / "archetype_marker_rankings.csv"),
        embedding_dim_top_markers=pd.read_csv(ANALYSIS_DIR / "embedding_dim_top_markers.csv"),
        reconstruction_by_condition=pd.read_csv(ANALYSIS_DIR / "reconstruction_summary_by_condition.csv"),
        sample_condition_counts=(
            aligned_meta.groupby(["sample_name", "condition_binary"], dropna=False)
            .size()
            .reset_index(name="n_cells")
        ),
    )


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
) -> Table:
    display_df = df.copy()
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


def build_run_overview(data: Batch6Data) -> pd.DataFrame:
    run = data.run_summary
    return pd.DataFrame(
        [
            {
                "selected_run": run["run_id"],
                "K": int(run["K"]),
                "latent_d": int(run["d"]),
                "val_recon": float(run["val_recon"]),
                "mean_marker_corr_val": float(run["mean_marker_corr_val"]),
                "mean_entropy_val": float(run["mean_entropy_val"]),
                "batch_size": int(run["batch_size"]),
                "tau": float(run["tau"]),
            }
        ]
    )


def archetype_identity_table(data: Batch6Data, styles: dict[str, ParagraphStyle]) -> Table:
    rows: list[dict[str, Any]] = []
    for arch in data.condition_mean_weights.columns.tolist():
        if not str(arch).startswith("archetype_"):
            continue
        arch_num = str(arch).split("_")[1]
        pos = (
            data.archetype_rankings[
                (data.archetype_rankings["archetype"].astype(str) == arch_num)
                & (data.archetype_rankings["direction"] == "positive")
            ]
            .sort_values("rank")
            .head(6)
        )
        info = ARCHETYPE_LABELS.get(str(arch), {"label": str(arch), "rationale": ""})
        rows.append(
            {
                "archetype": arch,
                "assigned_state": info["label"],
                "core_logic": info["rationale"],
                "top_positive_markers": ", ".join(pos["marker"].astype(str).tolist()),
            }
        )
    return dataframe_to_table(pd.DataFrame(rows), styles, max_rows=None)


def build_difference_summary(data: Batch6Data) -> pd.DataFrame:
    pairwise = data.pairwise_differences.copy()
    pairwise = pairwise.sort_values(["comparison", "abs_difference"], ascending=[True, False])
    return pairwise


def build_dominant_fraction_pivot(data: Batch6Data) -> pd.DataFrame:
    pivot = data.dominant_condition_fraction.pivot(
        index="condition_binary",
        columns="dominant_component_label",
        values="fraction",
    ).fillna(0.0)
    return pivot.reset_index()


def build_embedding_summary(data: Batch6Data) -> pd.DataFrame:
    top = data.embedding_dim_top_markers.copy()
    top = top.groupby(["embedding_dim", "direction"], as_index=False).head(6)
    return top


def add_page_number(canvas: Any, doc: Any) -> None:
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawRightString(doc.pagesize[0] - 0.5 * inch, 0.35 * inch, f"Page {doc.page}")


def build_report() -> None:
    data = load_batch6_data()
    styles = build_styles()

    story: list[Any] = []
    story.append(Spacer(1, 0.3 * inch))
    story.append(paragraph(REPORT_TITLE, styles["title"]))
    story.append(paragraph(REPORT_SUBTITLE, styles["subtitle"]))
    story.append(paragraph(
        "This report summarizes the GFP+ Batch 6 CyEmbed analysis comparing untreated cells with gemcitabine-resistant cells selected in vitro and in vivo. "
        "The pairwise condition-difference analyses are oriented as resistant condition minus untreated where applicable.",
        styles["body"],
    ))

    story.append(paragraph("Executive Summary", styles["h1"]))
    for item in EXECUTIVE_SUMMARY:
        story.append(bullet(item, styles["bullet"]))

    story.append(paragraph("Run Overview", styles["h1"]))
    story.append(dataframe_to_table(build_run_overview(data), styles, max_rows=None))

    story.append(Spacer(1, 0.06 * inch))
    story.append(paragraph("Study Scope", styles["h2"]))
    story.append(paragraph(
        "The analysis is restricted to the GFP+ compartment and compares three condition groups: untreated, Gem-R in vitro, and Gem-R in vivo. "
        "The report focuses on archetype identities, condition-level occupancy, pairwise condition differences, dominant-state composition, and marker-embedding structure.",
        styles["body"],
    ))
    story.append(dataframe_to_table(data.sample_condition_counts, styles, max_rows=None))

    story.append(Spacer(1, 0.06 * inch))
    story.append(paragraph("Biological Interpretation", styles["h1"]))
    for item in BIOLOGICAL_READOUT:
        story.append(bullet(item, styles["bullet"]))

    story.append(paragraph("Archetype Identity Table", styles["h2"]))
    story.append(archetype_identity_table(data, styles))

    story.append(Spacer(1, 0.06 * inch))
    story.append(paragraph("Condition-Level Tables", styles["h2"]))
    story.append(paragraph("Mean archetype weights by condition", styles["small"]))
    story.append(dataframe_to_table(data.condition_mean_weights, styles, max_rows=None))
    story.append(Spacer(1, 0.05 * inch))
    story.append(paragraph("Entropy and dominant-weight summary by condition", styles["small"]))
    story.append(dataframe_to_table(data.condition_entropy_summary, styles, max_rows=None))
    story.append(Spacer(1, 0.05 * inch))
    story.append(paragraph("Reconstruction summary by condition", styles["small"]))
    story.append(dataframe_to_table(data.reconstruction_by_condition, styles, max_rows=None))
    story.append(Spacer(1, 0.05 * inch))
    story.append(paragraph("Dominant archetype fraction by condition", styles["small"]))
    story.append(dataframe_to_table(build_dominant_fraction_pivot(data), styles, max_rows=None))

    story.append(PageBreak())
    story.append(paragraph("Pairwise Differences", styles["h1"]))
    story.append(paragraph(
        "The pairwise difference table below directly compares Gem-R in vitro versus untreated, Gem-R in vivo versus untreated, and Gem-R in vitro versus Gem-R in vivo. "
        "Positive values indicate relative enrichment of the first-listed condition.",
        styles["body"],
    ))
    story.append(dataframe_to_table(build_difference_summary(data), styles, max_rows=18))

    story.append(Spacer(1, 0.06 * inch))
    story.append(paragraph("Embedding Summary", styles["h2"]))
    story.append(paragraph("Top positive and negative markers per embedding dimension", styles["small"]))
    story.append(dataframe_to_table(build_embedding_summary(data), styles, max_rows=24))

    story.append(PageBreak())
    story.append(paragraph("Figures", styles["h1"]))
    figure_specs = [
        (
            data.figures_dir / "selected_training_curve.png",
            "Training curve for the selected Batch 6 run.",
        ),
        (
            data.figures_dir / "condition_mean_archetype_weights.png",
            "Condition-level mean archetype weights.",
        ),
        (
            data.figures_dir / "archetype_condition_means_heatmap.png",
            "Heatmap of mean archetype weights across untreated, Gem-R in vitro, and Gem-R in vivo.",
        ),
        (
            data.figures_dir / "archetype_pairwise_condition_differences_heatmap.png",
            "Pairwise condition-difference heatmap. Resistant-versus-untreated rows are oriented as resistant condition minus untreated.",
        ),
        (
            data.figures_dir / "archetype_pairwise_condition_differences_barh.png",
            "Horizontal bar summaries of pairwise condition differences by archetype.",
        ),
        (
            data.figures_dir / "dominant_archetype_stackedbars_sample_condition.png",
            "Dominant-archetype composition across samples and conditions.",
        ),
        (
            data.figures_dir / "weight_entropy_by_condition.png",
            "Weight entropy by condition. Smaller differences here suggest resistance changes state occupancy more than global mixture sharpness.",
        ),
        (
            data.figures_dir / "dominant_weight_by_condition.png",
            "Dominant archetype weight by condition.",
        ),
        (
            data.figures_dir / "archetype_marker_programs_zscore.png",
            "Archetype marker programs displayed as row-wise z-scored marker loadings.",
        ),
        (
            data.figures_dir / "marker_embeddings_pca.png",
            "Marker embedding PCA, useful for inspecting broad epithelial, mesenchymal, and chromatin neighborhoods.",
        ),
        (
            data.figures_dir / "marker_embeddings_umap.png",
            "Marker embedding UMAP.",
        ),
        (
            data.figures_dir / "marker_embedding_cosine_similarity_heatmap.png",
            "Cosine-similarity heatmap of marker embeddings.",
        ),
    ]
    for fig_path, caption in figure_specs:
        if fig_path.exists():
            story.append(scaled_image(fig_path, max_width=9.6 * inch, max_height=5.8 * inch))
            story.append(paragraph(caption, styles["caption"]))

    story.append(PageBreak())
    story.append(paragraph("Integrated Readout", styles["h1"]))
    conclusion = [
        "Both resistant conditions lose the untreated adhesive epithelial-plastic hybrid state and gain a shared plastic survivor component.",
        "Gem-R in vitro preferentially shifts toward a mesenchymal / chromatin-remodeled plastic program.",
        "Gem-R in vivo preferentially shifts toward a classical epithelial / progenitor-like program.",
        "The strongest interpretation is therefore that gemcitabine resistance in Batch 6 is partly shared across contexts, but the in vivo microenvironment redirects the resistant cells into a different residual state than the in vitro selection regime.",
    ]
    for item in conclusion:
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
