from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = REPORT_DIR / "cyembed_deconvolution_methods.pdf"
EQ_DIR = REPORT_DIR / "methods_eq"
EQ_DIR.mkdir(parents=True, exist_ok=True)

TITLE = "Methods: Probabilistic Archetype Deconvolution of Single-Cell CyTOF Profiles"
SUBTITLE = (
    "General methods description for simplex-constrained deconvolution, "
    "factorized archetype decoding, and probabilistic inference"
)

SECTIONS: list[tuple[str, list[tuple[str, str]]]] = [
    (
        "Overview",
        [
            (
                "p",
                "We modeled each cell as a convex combination of a small number of latent archetypal states. "
                "Let X denote the cell-by-marker matrix, where N is the number of cells and M is the number of retained protein features. "
                "Each row x_i corresponds to one cell and each column corresponds to one marker. "
                "The aim of the method is to learn a set of K archetypes representing extreme molecular programs and, for each cell, a simplex-constrained weight vector describing the relative contribution of each archetype to that cell.",
            ),
            (
                "eq",
                r"X \in \mathbb{R}^{N \times M}, \qquad \mathbf{w}_i = (w_{i1}, \ldots, w_{iK}), \qquad w_{ik} \geq 0, \qquad \sum_{k=1}^{K} w_{ik} = 1",
            ),
        ],
    ),
    (
        "Preprocessing",
        [
            (
                "p",
                "Marker intensities were standardized independently for each feature using statistics estimated on the training set. "
                "Under standard z-scoring, each marker m was transformed by subtracting its training-set mean and dividing by its training-set standard deviation. "
                "The framework also supports robust z-scoring based on the median and median absolute deviation, but the model itself is agnostic to the specific scaling choice.",
            ),
            (
                "eq",
                r"x'_{im} = \frac{x_{im} - \mu_m}{\sigma_m}",
            ),
            (
                "p",
                "A small positive constant was used when needed to avoid division by zero. "
                "Cells were then split into training and validation partitions. "
                "When a grouping variable such as sample identity or treatment condition was available, the split could be stratified so that each group was represented in both partitions.",
            ),
        ],
    ),
    (
        "Encoder and simplex weights",
        [
            (
                "p",
                "The encoder is a multilayer perceptron with rectified linear unit activations. "
                "In the deterministic formulation, the encoder maps the standardized marker vector x_i to a vector of archetype logits u_i. "
                "These logits are converted into simplex weights with a temperature-controlled softmax.",
            ),
            (
                "eq",
                r"\mathbf{u}_i = f_{\theta}(\mathbf{x}_i)",
            ),
            (
                "eq",
                r"w_{ik} = \frac{\exp(u_{ik}/\tau)}{\sum_{k'=1}^{K}\exp(u_{ik'}/\tau)}",
            ),
            (
                "p",
                "The temperature parameter tau > 0 controls the sharpness of the assignments. "
                "Smaller values yield more concentrated, nearly one-hot assignments; larger values yield smoother mixtures. "
                "The weights therefore provide a deconvolution of each cell into K interpretable latent programs.",
            ),
        ],
    ),
    (
        "Probabilistic formulation",
        [
            (
                "p",
                "In the probabilistic model, the encoder produces the parameters of a diagonal Gaussian posterior over latent logits rather than a single deterministic logit vector. "
                "This yields a logistic-normal model for the archetype weights after the softmax transform.",
            ),
            (
                "eq",
                r"q_{\phi}(\mathbf{u}_i \mid \mathbf{x}_i) = \mathcal{N}\!\left(\boldsymbol{\mu}^{(w)}_i,\ \mathrm{diag}\!\left(\exp(\log \boldsymbol{\sigma}^{2(w)}_i)\right)\right)",
            ),
            (
                "eq",
                r"\mathbf{u}_i = \boldsymbol{\mu}^{(w)}_i + \boldsymbol{\sigma}^{(w)}_i \odot \boldsymbol{\epsilon}_i, \qquad \boldsymbol{\epsilon}_i \sim \mathcal{N}(0, I)",
            ),
            (
                "eq",
                r"\mathbf{w}_i = \mathrm{softmax}(\mathbf{u}_i / \tau)",
            ),
            (
                "p",
                "Sampling is performed with the reparameterization trick. "
                "At evaluation time, predictions can be obtained from the posterior mean, from a single posterior sample, or by Monte Carlo averaging across multiple posterior samples.",
            ),
        ],
    ),
    (
        "Decoder parameterization",
        [
            (
                "p",
                "Two decoder parameterizations were used. "
                "In the direct decoder, the archetype matrix A is learned explicitly, with each row corresponding to the marker profile of one archetype. "
                "The reconstruction of cell i is the convex combination of the archetype profiles weighted by w_i.",
            ),
            (
                "eq",
                r"\hat{\mathbf{x}}_i = \mathbf{w}_i A = \sum_{k=1}^{K} w_{ik}\mathbf{a}_k",
            ),
            (
                "p",
                "In the factorized decoder, archetypes and markers are embedded jointly in a latent space of dimension d. "
                "Let Z denote the archetype embedding matrix, E the marker embedding matrix, and b a marker-specific bias term. "
                "The cell embedding implied by the archetype mixture is h_i = w_i Z, and the reconstructed marker vector is obtained by projecting h_i back into marker space.",
            ),
            (
                "eq",
                r"\mathbf{h}_i = \mathbf{w}_i Z",
            ),
            (
                "eq",
                r"\hat{\mathbf{x}}_i = \mathbf{h}_i E^{\top} + \mathbf{b}",
            ),
            (
                "eq",
                r"\hat{A} = Z E^{\top} + \mathbf{b}",
            ),
            (
                "p",
                "Thus, the factorized decoder implies an archetype-by-marker matrix A_hat without learning it directly. "
                "This low-rank parameterization constrains the archetype-marker relationships and yields an interpretable marker embedding that can be analyzed separately.",
            ),
        ],
    ),
    (
        "Optional residual latent",
        [
            (
                "p",
                "The probabilistic model optionally includes an additional residual latent variable to capture structured variation not fully explained by the convex archetype mixture. "
                "A second Gaussian posterior is inferred from the encoder trunk.",
            ),
            (
                "eq",
                r"q_{\phi}(\mathbf{r}_i \mid \mathbf{x}_i) = \mathcal{N}\!\left(\boldsymbol{\mu}^{(r)}_i,\ \mathrm{diag}\!\left(\exp(\log \boldsymbol{\sigma}^{2(r)}_i)\right)\right)",
            ),
            (
                "p",
                "In the factorized decoder, the residual latent is optionally projected into the same d-dimensional latent space and added to the archetype-derived latent representation before reconstruction.",
            ),
            (
                "eq",
                r"\mathbf{h}_i^{\mathrm{total}} = \mathbf{w}_i Z + \mathbf{r}_i P_r",
            ),
            (
                "eq",
                r"\hat{\mathbf{x}}_i = \mathbf{h}_i^{\mathrm{total}} E^{\top} + \mathbf{b}",
            ),
            (
                "p",
                "In the direct decoder, the residual latent contributes additively in marker space through a learned loading matrix G.",
            ),
            (
                "eq",
                r"\hat{\mathbf{x}}_i = \mathbf{w}_i A + \mathbf{r}_i G",
            ),
        ],
    ),
    (
        "Objective function",
        [
            (
                "p",
                "Model parameters were estimated by minimizing a composite objective consisting of a reconstruction term and several regularization terms designed to improve interpretability and prevent degenerate solutions. "
                "For the deterministic model, the loss is:",
            ),
            (
                "eq",
                r"\mathcal{L}_{\mathrm{det}} = \mathcal{L}_{\mathrm{recon}} + \lambda_{\mathrm{ent}}\mathcal{L}_{\mathrm{ent}} + \lambda_{\mathrm{sep}}\mathcal{L}_{\mathrm{sep}} + \lambda_{\mathrm{bal}}\mathcal{L}_{\mathrm{bal}}",
            ),
            (
                "p",
                "For the probabilistic model, Kullback-Leibler regularization terms are added for the logistic-normal archetype weights and, when present, the residual latent:",
            ),
            (
                "eq",
                r"\mathcal{L}_{\mathrm{prob}} = \mathcal{L}_{\mathrm{recon}} + \beta_w\mathcal{L}_{\mathrm{KL},w} + \beta_r\mathcal{L}_{\mathrm{KL},r} + \lambda_{\mathrm{ent}}\mathcal{L}_{\mathrm{ent}} + \lambda_{\mathrm{sep}}\mathcal{L}_{\mathrm{sep}} + \lambda_{\mathrm{bal}}\mathcal{L}_{\mathrm{bal}}",
            ),
        ],
    ),
    (
        "Reconstruction term",
        [
            (
                "p",
                "The reconstruction term was either mean squared error or Huber loss between the observed standardized marker matrix and its reconstruction. "
                "For mean squared error:",
            ),
            (
                "eq",
                r"\mathcal{L}_{\mathrm{recon}} = \frac{1}{NM}\sum_{i=1}^{N}\sum_{m=1}^{M}(x_{im} - \hat{x}_{im})^2",
            ),
            (
                "p",
                "For Huber loss with threshold delta, the reconstruction term is written as an average of a pointwise robust penalty:",
            ),
            (
                "eq",
                r"\mathcal{L}_{\mathrm{Huber}} = \frac{1}{NM}\sum_{i=1}^{N}\sum_{m=1}^{M}\rho_{\delta}(x_{im} - \hat{x}_{im})",
            ),
            (
                "eq",
                r"\rho_{\delta}(a) = \frac{1}{2}a^2, \qquad |a| \leq \delta",
            ),
            (
                "eq",
                r"\rho_{\delta}(a) = \delta |a| - \frac{1}{2}\delta^2, \qquad |a| > \delta",
            ),
        ],
    ),
    (
        "Regularization terms",
        [
            (
                "p",
                "To encourage sharp deconvolution, the mean Shannon entropy of the cell-wise archetype weights was penalized:",
            ),
            (
                "eq",
                r"\mathcal{L}_{\mathrm{ent}} = \frac{1}{N}\sum_{i=1}^{N}\left[-\sum_{k=1}^{K} w_{ik}\log w_{ik}\right]",
            ),
            (
                "p",
                "To encourage separation between archetypes, the mean squared off-diagonal cosine similarity between archetype representations was penalized. "
                "In the factorized model this penalty is applied to the rows of Z; in the direct model it is applied to the rows of A.",
            ),
            (
                "eq",
                r"\mathcal{L}_{\mathrm{sep}} = \frac{1}{K(K-1)}\sum_{k \neq k'}\left(\frac{\mathbf{a}_k^{\top}\mathbf{a}_{k'}}{\|\mathbf{a}_k\|\,\|\mathbf{a}_{k'}\|}\right)^2",
            ),
            (
                "p",
                "To discourage unused or dead archetypes, the average archetype usage vector w_bar was regularized toward the uniform distribution. "
                "Let w_bar_k = (1 / N) sum_i w_ik. The default balance penalty was:",
            ),
            (
                "eq",
                r"\bar{w}_k = \frac{1}{N}\sum_{i=1}^{N} w_{ik}, \qquad \mathcal{L}_{\mathrm{bal}} = \frac{1}{K}\sum_{k=1}^{K}\left(\bar{w}_k - \frac{1}{K}\right)^2",
            ),
        ],
    ),
    (
        "Variational regularization",
        [
            (
                "p",
                "For a diagonal Gaussian posterior q = N(mu, diag(exp(logvar))) and an isotropic standard normal prior p = N(0, I), the KL divergence was computed in closed form:",
            ),
            (
                "eq",
                r"D_{\mathrm{KL}}(q\|p) = -\frac{1}{2}\sum_{j}\left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)",
            ),
            (
                "p",
                "To stabilize optimization, the coefficients beta_w and beta_r could be linearly warmed up over a specified number of epochs:",
            ),
            (
                "eq",
                r"\beta_{\mathrm{eff}}(t) = \beta \cdot \min\left(1,\frac{t}{T_{\mathrm{warmup}}}\right)",
            ),
        ],
    ),
    (
        "Optimization",
        [
            (
                "p",
                "All models were optimized using Adam with mini-batches of cells. "
                "Key hyperparameters included the number of archetypes K, latent dimension d, encoder hidden-layer widths, learning rate, batch size, decoder type, temperature tau, and regularization coefficients. "
                "Training proceeded for a fixed maximum number of epochs with early stopping based on validation reconstruction loss. "
                "The best model state, defined by the minimum validation reconstruction loss, was restored before final evaluation.",
            ),
        ],
    ),
    (
        "Hyperparameter sweep and model selection",
        [
            (
                "p",
                "A structured grid of candidate configurations was evaluated across deterministic and probabilistic variants, values of K, latent dimensionality d, encoder width, learning rate, batch size, decoder parameterization, temperature, and regularization coefficients. "
                "Each run was assigned a stable hyperparameter fingerprint so that equivalent runs could be recognized and re-used. "
                "The primary selection criterion was validation reconstruction loss. "
                "Secondary diagnostics included mean per-marker correlation between observed and reconstructed values, entropy of the archetype weights, variance of average archetype usage, dominant-weight statistics, and the number of dead archetypes.",
            ),
        ],
    ),
    (
        "Outputs and downstream summaries",
        [
            (
                "p",
                "For each trained model, the main outputs were the cell-by-archetype weight matrix W, the reconstructed cell-by-marker matrix X_hat, the archetype-by-marker matrix A_hat, and, for factorized models, the archetype embedding matrix Z and marker embedding matrix E. "
                "For a given cell, the dominant archetype was defined as argmax_k w_ik. "
                "Group-level deconvolution summaries were obtained by averaging archetype weights within groups such as sample or treatment condition. "
                "Archetypes were identified biologically from their highest positive and negative marker loadings rather than from raw archetype indices, which are exchangeable across independent fits.",
            ),
        ],
    ),
    (
        "Marker embedding analysis",
        [
            (
                "p",
                "For factorized models, the learned marker embedding matrix E provides a low-dimensional representation of marker co-variation inferred jointly with the archetype model. "
                "Marker-to-marker relationships can therefore be summarized with cosine similarity and visualized using dimensionality-reduction methods such as principal component analysis or uniform manifold approximation and projection. "
                "These analyses are used only for interpretation of the fitted model and do not alter the optimization procedure itself.",
            ),
        ],
    ),
]


def paragraph(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def render_equation_png(tex: str, eq_index: int) -> Path:
    out_path = EQ_DIR / f"eq_{eq_index:02d}.png"
    fig = plt.figure(figsize=(14, 0.9), dpi=220)
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.01, 0.5, f"${tex}$", fontsize=18, ha="left", va="center", color="black")
    fig.savefig(out_path, dpi=220, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out_path


def equation_flowable(tex: str, eq_index: int, max_width: float = 6.35 * inch) -> Image:
    image_path = render_equation_png(tex, eq_index)
    with PILImage.open(image_path) as img:
        width_px, height_px = img.size
    scale = min(1.0, max_width / float(width_px))
    return Image(str(image_path), width=float(width_px) * scale, height=float(height_px) * scale)


def build_pdf() -> None:
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=19,
        leading=23,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#173b63"),
        spaceAfter=8,
    )
    subtitle_style = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#555555"),
        spaceAfter=16,
    )
    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        alignment=TA_LEFT,
        textColor=colors.HexColor("#173b63"),
        spaceBefore=7,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    story = [
        Spacer(1, 0.3 * inch),
        paragraph(TITLE, title_style),
        paragraph(SUBTITLE, subtitle_style),
    ]

    eq_index = 0
    for heading, blocks in SECTIONS:
        story.append(paragraph(heading, heading_style))
        for kind, text in blocks:
            if kind == "p":
                story.append(paragraph(text, body_style))
            else:
                eq_index += 1
                story.append(equation_flowable(text, eq_index))
                story.append(Spacer(1, 0.06 * inch))

    doc = SimpleDocTemplate(
        str(REPORT_PATH),
        pagesize=letter,
        leftMargin=0.72 * inch,
        rightMargin=0.72 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.65 * inch,
        title=TITLE,
        author="Codex",
    )
    doc.build(story)


if __name__ == "__main__":
    build_pdf()
    print(f"Wrote {REPORT_PATH}")
