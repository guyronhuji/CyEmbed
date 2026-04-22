from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
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
    archetype_delta: pd.DataFrame
    archetype_rankings: pd.DataFrame
    condition_mean_weights: pd.DataFrame
    condition_entropy_summary: pd.DataFrame
    components_mean: pd.DataFrame
    figures_dir: Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_ROOT = PROJECT_ROOT / "Analysis"
REPORT_DIR = ANALYSIS_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FIG_DIR = REPORT_DIR / "figures"
REPORT_FIG_DIR.mkdir(parents=True, exist_ok=True)

PDF_PATH = REPORT_DIR / "pdac_gfppos_chemo_k7_nature_style_report.pdf"
MD_PATH = REPORT_DIR / "pdac_gfppos_chemo_k7_nature_style_report.md"
PAGE_BODY_WIDTH = letter[0] - (0.6 * inch) - (0.6 * inch)

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


def load_batches() -> dict[str, BatchData]:
    out: dict[str, BatchData] = {}
    for key, cfg in BATCH_CONFIG.items():
        base = cfg["analysis_dir"]
        out[key] = BatchData(
            batch_key=key,
            batch_label=cfg["label"],
            dose_label=cfg["dose_label"],
            analysis_dir=base,
            run_summary=pd.read_csv(base / "available_runs_snapshot.csv").iloc[0],
            archetype_delta=pd.read_csv(base / "archetype_treated_minus_untreated.csv"),
            archetype_rankings=pd.read_csv(base / "archetype_marker_rankings.csv"),
            condition_mean_weights=pd.read_csv(base / "condition_mean_weights.csv"),
            condition_entropy_summary=pd.read_csv(base / "condition_entropy_summary.csv"),
            components_mean=pd.read_csv(base / "components_mean_matrix.csv", index_col=0),
            figures_dir=base / "figures",
        )
    return out


def build_alignment_map(batches: dict[str, BatchData], anchor_key: str = "batch2") -> dict[str, dict[str, tuple[str, float]]]:
    anchor = batches[anchor_key].components_mean.copy()
    out: dict[str, dict[str, tuple[str, float]]] = {anchor_key: {}}
    for arch in anchor.index:
        out[anchor_key][str(arch)] = (str(arch), 1.0)

    for other_key, batch in batches.items():
        if other_key == anchor_key:
            continue
        other = batch.components_mean.copy()
        corr = np.corrcoef(anchor.to_numpy(), other.to_numpy())[: anchor.shape[0], anchor.shape[0] :]
        best_perm: tuple[int, ...] | None = None
        best_score: float | None = None
        for perm in permutations(range(other.shape[0])):
            score = float(sum(corr[i, perm[i]] for i in range(anchor.shape[0])))
            if best_score is None or score > best_score:
                best_score = score
                best_perm = perm
        if best_perm is None:
            raise RuntimeError(f"could not align {other_key}")
        out[other_key] = {}
        for i, j in enumerate(best_perm):
            out[other_key][str(anchor.index[i])] = (str(other.index[j]), float(corr[i, j]))
    return out


def build_shared_delta_table(
    batches: dict[str, BatchData],
    alignment_map: dict[str, dict[str, tuple[str, float]]],
    anchor_key: str = "batch2",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    delta_lookup = {
        key: batch.archetype_delta.set_index("archetype")["treated_minus_untreated"].to_dict()
        for key, batch in batches.items()
    }
    delta_rows: list[dict[str, Any]] = []
    map_rows: list[dict[str, Any]] = []

    anchor_arches = list(batches[anchor_key].components_mean.index)
    for arch in anchor_arches:
        arch_str = str(arch)
        label = SHARED_ARCHETYPE_LABELS.get(arch_str, {}).get("label", arch_str)
        delta_row = {"shared_archetype": arch_str, "shared_label": label}
        map_row = {"shared_archetype": arch_str, "shared_label": label}
        for key, batch in batches.items():
            matched_arch, corr = alignment_map[key][arch_str]
            delta_row[f"{batch.batch_label} ({batch.dose_label})"] = float(delta_lookup[key][matched_arch])
            map_row[f"{batch.batch_label}_matched"] = matched_arch
            map_row[f"{batch.batch_label}_corr"] = float(corr)
            if corr >= 0.7:
                conf = "high"
            elif corr >= 0.5:
                conf = "moderate"
            else:
                conf = "low"
            map_row[f"{batch.batch_label}_confidence"] = conf
        delta_rows.append(delta_row)
        map_rows.append(map_row)
    return pd.DataFrame(delta_rows), pd.DataFrame(map_rows)


def top_markers(rankings: pd.DataFrame, arch_num: int, n: int = 5) -> list[str]:
    sub = rankings[
        (rankings["archetype"].astype(str) == str(arch_num))
        & (rankings["direction"].astype(str) == "positive")
    ].sort_values("value", ascending=False)
    return [str(x) for x in sub["marker"].head(n).tolist()]


def build_results_text(
    batches: dict[str, BatchData],
    alignment_map: dict[str, dict[str, tuple[str, float]]],
) -> tuple[str, list[str], list[str], list[str]]:
    shared_delta_df, mapping_df = build_shared_delta_table(batches, alignment_map)
    overview = (
        "CyEmbed deconvolution of the GFP+ compartment resolved seven recurrent archetypal programs "
        "across Batch 2, Batch 3, and Batch 4. Anchoring the cross-batch alignment to the refreshed "
        "Batch 2 K=7 solution separated basal-like/KRT5, epithelial-adhesion, classical epithelial, "
        "mesenchymal-plastic, proliferative/chromatin-active, CXCR4-CD109 receptor-high plasticity, "
        "and Polycomb-repressed/cycling chromatin states."
    )

    cross_batch = [
        "Across all three batches, the classical epithelial/secretory state remained the most stable cross-batch program, with strong marker-program matching in Batch 3 and Batch 4.",
        "The epithelial-adhesion hybrid state was reproducibly depleted by treatment in all three batches, indicating that chemotherapy consistently disfavors a CD24/CD44/E-cadherin-rich adhesive GFP+ state.",
        "By contrast, the states enriched after treatment differed by batch, supporting residual-state pluralism rather than a single universal resistant endpoint.",
    ]

    batch_lines: list[str] = []
    for key, batch in batches.items():
        label_map = {
            matched: SHARED_ARCHETYPE_LABELS[anchor]["label"]
            for anchor, (matched, _) in alignment_map[key].items()
        }
        delta = batch.archetype_delta.sort_values("treated_minus_untreated", ascending=False).reset_index(drop=True)
        gains = []
        losses = []
        for _, row in delta.iterrows():
            name = label_map.get(str(row["archetype"]), str(row["archetype"]))
            val = float(row["treated_minus_untreated"])
            if val > 0 and len(gains) < 2:
                gains.append(f"{name} ({val:+.3f})")
            if val < 0 and len(losses) < 2:
                losses.append(f"{name} ({val:+.3f})")
        batch_lines.append(
            f"{batch.batch_label} ({batch.dose_label}) showed enrichment of {', '.join(gains)} and depletion of {', '.join(losses)}."
        )

    caveats = []
    for _, row in mapping_df.iterrows():
        label = str(row["shared_label"])
        for batch_label in ["Batch 3", "Batch 4"]:
            if row[f"{batch_label}_confidence"] == "low":
                caveats.append(
                    f"{batch_label} alignment for {label} is low-confidence (corr={float(row[f'{batch_label}_corr']):.3f}) and should be interpreted cautiously."
                )
    return overview, cross_batch, batch_lines, caveats


def build_styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#555555"),
            spaceAfter=16,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            spaceAfter=6,
            textColor=colors.HexColor("#1b365d"),
        ),
        "body": ParagraphStyle(
            "Body",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=12.5,
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=12.5,
            leftIndent=12,
            firstLineIndent=-8,
            spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=9.5,
            textColor=colors.HexColor("#555555"),
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
    }


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text.replace("\n", "<br/>"), style)


def bullet(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(f"• {text}", style)


def scaled_image(path: Path, max_width: float, max_height: float) -> Image:
    img = PILImage.open(path)
    width, height = img.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        if np.isnan(value):
            return ""
        return f"{value:.4f}"
    return str(value)


def _make_table_style(base_styles: dict[str, ParagraphStyle], font_size: float) -> dict[str, ParagraphStyle]:
    header = ParagraphStyle(
        "TableHeader",
        parent=base_styles["small"],
        fontName="Helvetica-Bold",
        fontSize=font_size,
        leading=font_size + 1.5,
        alignment=TA_LEFT,
        textColor=colors.black,
        wordWrap="CJK",
    )
    cell = ParagraphStyle(
        "TableCell",
        parent=base_styles["small"],
        fontName="Helvetica",
        fontSize=font_size,
        leading=font_size + 1.5,
        alignment=TA_LEFT,
        textColor=colors.black,
        wordWrap="CJK",
    )
    return {"header": header, "cell": cell}


def table_from_df(df: pd.DataFrame, base_styles: dict[str, ParagraphStyle], max_width: float = PAGE_BODY_WIDTH) -> Table:
    ncols = len(df.columns)
    font_size = 7.0
    if ncols >= 8:
        font_size = 5.6
    elif ncols >= 6:
        font_size = 6.1
    elif ncols >= 4:
        font_size = 6.6

    tstyles = _make_table_style(base_styles, font_size)
    display_df = df.copy()
    display_df.columns = [str(c).replace("_", " ") for c in display_df.columns]

    weights: list[float] = []
    for col in display_df.columns:
        values = [_format_cell(v) for v in display_df[col].tolist()]
        max_len = max([len(str(col)), *(len(v) for v in values)] or [8])
        weights.append(float(min(28, max(6, max_len))))
    total_weight = sum(weights) or 1.0
    col_widths = [max_width * (w / total_weight) for w in weights]

    header_row = [Paragraph(escape(str(col)), tstyles["header"]) for col in display_df.columns]
    body_rows = [
        [Paragraph(escape(_format_cell(v)).replace("\n", "<br/>"), tstyles["cell"]) for v in row]
        for row in display_df.itertuples(index=False, name=None)
    ]

    tbl = Table([header_row] + body_rows, repeatRows=1, colWidths=col_widths, splitByRow=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9e2f3")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), font_size),
                ("LEADING", (0, 0), (-1, -1), font_size + 1.5),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b0b0b0")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#f7f7f7")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return tbl


def add_page_number(canvas: Any, doc: Any) -> None:
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawRightString(doc.pagesize[0] - 0.45 * inch, 0.35 * inch, f"Page {doc.page}")


def build_markdown(
    overview: str,
    cross_batch: list[str],
    batch_lines: list[str],
    caveats: list[str],
    shared_delta_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
) -> str:
    lines = [
        "# Nature-style GFP+ PDAC chemo report",
        "",
        "## Summary",
        "",
        overview,
        "",
        "## Cross-batch interpretation",
        "",
    ]
    lines.extend([f"- {x}" for x in cross_batch])
    lines.extend(["", "## Batch-specific treatment effects", ""])
    lines.extend([f"- {x}" for x in batch_lines])
    lines.extend(["", "## Shared-archetype deltas", "", shared_delta_df.to_markdown(index=False), ""])
    lines.extend(["## Alignment confidence", "", mapping_df.to_markdown(index=False), ""])
    if caveats:
        lines.extend(["## Cautions", ""])
        lines.extend([f"- {x}" for x in caveats])
    return "\n".join(lines) + "\n"


def build_pdf() -> None:
    batches = load_batches()
    alignment_map = build_alignment_map(batches)
    shared_delta_df, mapping_df = build_shared_delta_table(batches, alignment_map)
    overview, cross_batch, batch_lines, caveats = build_results_text(batches, alignment_map)

    shared_delta_df.to_csv(REPORT_DIR / "cross_batch_shared_archetype_deltas.csv", index=False)
    mapping_df.to_csv(REPORT_DIR / "cross_batch_shared_archetype_alignment_map.csv", index=False)
    MD_PATH.write_text(build_markdown(overview, cross_batch, batch_lines, caveats, shared_delta_df, mapping_df))

    styles = build_styles()
    story: list[Any] = []
    story.append(Spacer(1, 0.35 * inch))
    story.append(p("Nature-style PDAC GFP+ chemo report", styles["title"]))
    story.append(
        p(
            "Integrated seven-state interpretation of Batch 2 (5 doses), Batch 3 (5 doses), and Batch 4 (6 doses) CyEmbed analyses",
            styles["subtitle"],
        )
    )

    story.append(p("Summary", styles["h1"]))
    story.append(p(overview, styles["body"]))

    story.append(p("Cross-batch interpretation", styles["h1"]))
    for item in cross_batch:
        story.append(bullet(item, styles["bullet"]))

    heatmap_path = REPORT_FIG_DIR / "cross_batch_shared_archetype_deltas_heatmap.png"
    if heatmap_path.exists():
        story.append(Spacer(1, 0.08 * inch))
        story.append(scaled_image(heatmap_path, max_width=7.0 * inch, max_height=4.8 * inch))
        story.append(
            p(
                "Cross-batch heatmap of treated-minus-untreated shifts after collapsing each batch into seven shared archetype families anchored to the Batch 2 K=7 solution.",
                styles["caption"],
            )
        )

    story.append(p("Batch-specific treatment effects", styles["h1"]))
    for item in batch_lines:
        story.append(bullet(item, styles["bullet"]))

    story.append(Spacer(1, 0.08 * inch))
    story.append(p("Shared-archetype delta table", styles["small"]))
    story.append(table_from_df(shared_delta_df.round(4), styles))

    story.append(Spacer(1, 0.08 * inch))
    story.append(p("Cross-batch mapping confidence", styles["small"]))
    story.append(table_from_df(mapping_df, styles))

    if caveats:
        story.append(Spacer(1, 0.08 * inch))
        story.append(p("Cautions", styles["h1"]))
        for item in caveats:
            story.append(bullet(item, styles["bullet"]))

    for key, batch in batches.items():
        story.append(PageBreak())
        story.append(p(f"{batch.batch_label}: selected K=7 run", styles["h1"]))
        story.append(
            p(
                f"Selected run `{batch.run_summary['run_id']}` was retained for {batch.batch_label} with "
                f"validation reconstruction loss {float(batch.run_summary['val_recon']):.4f} and "
                f"mean validation marker correlation {float(batch.run_summary['mean_marker_corr_val']):.3f}.",
                styles["body"],
            )
        )

        identity_rows = []
        for arch in batch.components_mean.index:
            markers = ", ".join(top_markers(batch.archetype_rankings, int(str(arch).split('_')[-1]), n=5))
            identity_rows.append(
                {
                    "archetype": str(arch),
                    "top positive markers": markers,
                }
            )
        story.append(p("Archetype identity", styles["small"]))
        story.append(table_from_df(pd.DataFrame(identity_rows), styles))

        story.append(Spacer(1, 0.08 * inch))
        story.append(p("Treated-minus-untreated delta", styles["small"]))
        show_delta = batch.archetype_delta[
            ["archetype", "treated_mean", "untreated_mean", "treated_minus_untreated", "welch_p_value"]
        ].copy()
        story.append(table_from_df(show_delta.round(4), styles))

        fig_path = batch.figures_dir / "archetype_treated_minus_untreated_barh.png"
        if fig_path.exists():
            story.append(Spacer(1, 0.08 * inch))
            story.append(scaled_image(fig_path, max_width=6.8 * inch, max_height=4.2 * inch))
            story.append(
                p(
                    f"{batch.batch_label} treated-minus-untreated archetype shifts for the selected GFP+ K=7 run.",
                    styles["caption"],
                )
            )

    doc = SimpleDocTemplate(
        str(PDF_PATH),
        pagesize=letter,
        leftMargin=0.6 * inch,
        rightMargin=0.6 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.55 * inch,
        title="Nature-style PDAC GFP+ chemo report",
        author="OpenAI Codex",
    )
    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)


if __name__ == "__main__":
    build_pdf()
