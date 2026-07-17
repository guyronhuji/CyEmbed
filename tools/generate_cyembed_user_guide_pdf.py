"""Generate the CyEmbed user guide PDF: method, workflow, parameters, diagnostics.

Companion to generate_cyembed_methods_pdf.py (which is a paper-style methods section). This one
is the practical guide: how to run CyEmbed, on CyTOF and on scRNA-seq, with one sample or many,
and how to tell whether the result is real.

Findings tagged [measured] come from the benchmarks in this directory and are cited inline.
Everything else is the code's default or reasoned from it -- the distinction is deliberate.

Run:  python tools/generate_cyembed_user_guide_pdf.py
"""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
REPORT_PATH = REPORT_DIR / "cyembed_user_guide.pdf"

TITLE = "CyEmbed: Method, Workflow and Parameters"
SUBTITLE = (
    "Archetypal deconvolution of single-cell data &mdash; a practical guide for CyTOF and "
    "scRNA-seq, single-sample and multi-sample.<br/>"
    "Claims marked [measured] are benchmarked and cited; all others are defaults or reasoning."
)

ACCENT = colors.HexColor("#173b63")
MUTED = colors.HexColor("#555555")
RULE = colors.HexColor("#c8d3e0")
BAND = colors.HexColor("#eef2f7")
WARN = colors.HexColor("#8a2f2f")

# ---------------------------------------------------------------------------------------------
# Content. ("h", text) heading | ("p", text) paragraph | ("code", text) | ("t", (widths, rows))
# ("warn", text) | ("pb", None) page break
# ---------------------------------------------------------------------------------------------

SECTIONS: list[tuple[str, object]] = [
    ("h", "1. What the model does"),
    ("p",
     "CyEmbed models each cell as a <b>convex combination of K archetypes</b> &mdash; extreme "
     "molecular programs. It is an autoencoder whose bottleneck is a simplex."),
    ("code",
     "u_i = EncoderMLP(x_i)              # K logits from the cell's feature vector\n"
     "w_i = normalizer(u_i / tau)        # w_ik >= 0,  sum_k w_ik = 1"),
    ("p",
     "<b>w_i is the object you care about</b>: cell i's <i>proportions</i> over the K archetypes. "
     "This is the difference from NMF, whose loadings are unnormalised and therefore confounded "
     "with library size and total expression. Asking &quot;what fraction of this cell is program "
     "A?&quot; is a question only the simplex can answer."),
    ("p", "Decoding takes one of two forms:"),
    ("code",
     "factorized:  h_i   = w_i @ Z            # Z (K,d): archetype coords in a latent space\n"
     "             x_hat = h_i @ E.T + b      # E (M,d): a d-dim embedding per feature\n"
     "             A_hat = Z @ E.T + b        # archetype profiles in feature space\n"
     "\n"
     "direct:      x_hat = w_i @ A            # A (K,M) learned directly, no bias term\n"
     "             A_hat = A"),
    ("p",
     "With the optional per-sample offset, a learned intercept is added in feature space after "
     "either decoder, centred to sum to zero across samples:"),
    ("code", "x_hat += B_eff[s_i]        where  B_eff = B - B.mean(0)     # B is (S, M)"),
    ("h", "The geometry, which is the entire point"),
    ("p",
     "Because w lies on the simplex, every cell sits inside the <b>convex hull of K points</b> "
     "&mdash; a (K&minus;1)-dimensional object. Archetypes are the <i>vertices</i>; cells "
     "interpolate between them. That is a claim about biology &mdash; that cells lie on a "
     "continuum between extreme states &mdash; not a factorisation convenience. It is why d is "
     "not a capacity knob (&sect;5.2), why cells with high reconstruction error are "
     "scientifically interesting (they fall outside the hull), and why per-cell weight entropy "
     "is a meaningful coordinate rather than a summary statistic."),
    ("p", "The loss:"),
    ("code",
     "L = recon(x_hat, x)\n"
     "  + lambda_entropy  * mean_i H(w_i)         # per-cell weight entropy\n"
     "  + lambda_sep      * separation(Z or A)    # push archetypes apart\n"
     "  + lambda_balance  * balance(mean_i w_i)   # guard against dead archetypes\n"
     "  [+ beta_w * KL_w + beta_r * KL_r]         # probabilistic variant only"),

    ("pb", None),
    ("h", "2. Workflow: single sample"),
    ("p",
     "This is the simpler case and it is what both scRNA-seq datasets in this project actually "
     "are: BCK_44 and PDX_02 are <b>separate single-sample h5ads</b>, switched by one field in "
     "the YAML. Sample identity lives in the filename; .obs carries only cell_id. Nothing is "
     "concatenated."),
    ("code",
     "import anndata as ad\n"
     "from CyEmbed.data import extract_matrix, fit_scaler, preprocess_array, \\\n"
     "                        split_train_val_indices\n"
     "from CyEmbed.train import build_sweep_configs, run_sweep\n"
     "\n"
     "adata  = ad.read_h5ad('PDX_02_sct_pearson_residuals_hvg.h5ad')\n"
     "bundle = extract_matrix(adata=adata, source='X')      # residuals live in X, dense\n"
     "\n"
     "scaler, _ = fit_scaler(bundle.X, mode='none')         # scRNA: residuals are ~N(0,1)\n"
     "X = preprocess_array(bundle.X, scaler)\n"
     "\n"
     "train_idx, val_idx = split_train_val_indices(\n"
     "    n_cells=X.shape[0], val_fraction=0.2, seed=7,\n"
     "    stratify_labels=bundle.cluster_ids,   # or None if unlabelled\n"
     ")"),
    ("p",
     "With one sample there is no patient confounding to correct, so <b>use_sample_offset stays "
     "False</b> &mdash; there is nothing for B to absorb. sample_col is unnecessary and "
     "balanced_max_per_sample does nothing."),

    ("h", "3. Workflow: multiple samples"),
    ("p",
     "The goal is a <b>shared archetype basis with sample-specific composition</b>. Training per "
     "sample and matching archetypes afterwards defeats the purpose: you get K archetypes per "
     "sample with no correspondence, and post-hoc matching by cosine similarity is exactly the "
     "fragile step you were trying to avoid. Train jointly."),
    ("p",
     "<b>Step 1 &mdash; concatenate.</b> Nothing in the current pipeline does this. Use an inner "
     "join: HVGs are selected per sample, so the gene sets differ, and an outer join would "
     "fabricate residuals for genes never modelled in one sample."),
    ("code",
     "a = ad.read_h5ad('BCK_44_sct_pearson_residuals_hvg.h5ad'); a.obs['sample'] = 'BCK_44'\n"
     "b = ad.read_h5ad('PDX_02_sct_pearson_residuals_hvg.h5ad'); b.obs['sample'] = 'PDX_02'\n"
     "combined = ad.concat([a, b], join='inner')     # inner: HVG sets differ per sample"),
    ("p", "<b>Step 2 &mdash; extract, and assert the sample column resolved.</b>"),
    ("code",
     "bundle = extract_matrix(adata=combined, source='X', sample_col='sample')\n"
     "assert bundle.sample_ids is not None, 'sample_col did not resolve -- check adata.obs'"),
    ("warn",
     "sample_col fails silently. extract_matrix guards with `if sample_col in adata.obs`, so a "
     "typo yields sample_ids=None with no warning &mdash; and that quietly disables stratified "
     "splitting AND balanced scaling. The model trains fine and you never learn your sample "
     "handling did nothing. Always assert."),
    ("p",
     "<b>Step 3 &mdash; stratify the split by sample</b>, so every sample appears in both halves. "
     "split_train_val_indices splits <i>within</i> each group, which is what you want. This is "
     "<b>required</b> if you enable the offset: a grouped split that holds out whole samples "
     "leaves B untrained for exactly the samples you evaluate on."),
    ("code",
     "train_idx, val_idx = split_train_val_indices(\n"
     "    n_cells=bundle.X.shape[0], val_fraction=0.2, seed=7,\n"
     "    stratify_labels=bundle.sample_ids,      # every sample in BOTH halves\n"
     ")\n"
     "# CyTOF only: stop the scaler being dominated by your largest sample\n"
     "scaler, _ = fit_scaler(bundle.X, mode='zscore',\n"
     "                       sample_ids=bundle.sample_ids, balanced_max_per_sample=2000)"),
    ("p",
     "<b>Step 4 &mdash; run with the offset OFF first, and look.</b> This ordering matters more "
     "than the code does."),
    ("code",
     "from CyEmbed.analysis import load_run_outputs, summarize_by_group\n"
     "run   = load_run_outputs(best_run_dir)\n"
     "W     = run.get('W_mean', run['W'])\n"
     "stats = summarize_by_group(W, run['sample_ids'], group_name='sample')\n"
     "stats['dominant_fractions']     # <- the diagnostic"),
    ("t", ([1.45 * inch, 4.75 * inch], [
        ["What you see", "What to do"],
        ["An archetype with dominant_fraction ~1.0 in one sample and ~0 elsewhere",
         "An <b>identity archetype</b>. This is the failure the offset exists for. Turn it on."],
        ["Graded differences in archetype usage across samples",
         "This is your <b>result</b>, not a bug. Shared archetypes with sample-varying "
         "composition is the goal of joint training. Leave the offset off &mdash; correcting it "
         "deletes your finding."],
    ])),
    ("p",
     "<b>Step 5 &mdash; if you do enable it</b>, set use_sample_offset=True. B is created only "
     "when enabled, warm-started at the empirical per-sample mean, centred zero-sum each forward "
     "pass, and held out of weight decay. The key is that it only enters the config fingerprint "
     "when True, so leaving it off preserves your cached run directories."),
    ("warn",
     "In PDX and patient tumours, inter-sample variation substantially IS the biology &mdash; "
     "different tumours, different genotypes. B removes additive shifts indiscriminately, "
     "technical and biological alike. And if each patient was processed as its own batch, "
     "patient and batch are perfectly confounded: no method separates them, and you can only "
     "remove both. Decide explicitly which you are removing."),
    ("p",
     "<b>Step 6 &mdash; read B to find out what the sample effect actually is.</b> B is an "
     "(S, M) matrix of per-sample, per-gene offsets. Which genes shift most tells you whether "
     "you are correcting technical variation or erasing biology: ribosomal / mitochondrial / "
     "housekeeping is technical; a coherent biological program means stop."),
    ("p",
     "Scope and honesty about the offset: it removes <b>additive</b> shifts only, not "
     "multiplicative or interaction effects. It is a <b>soft</b> fix &mdash; it removes the "
     "<i>incentive</i> for w to encode sample identity, not the ability. The encoder stays "
     "sample-blind by design, so you can still project an unseen sample in one forward pass "
     "(it decodes at B = 0, the average baseline). A_hat excludes B: it is the archetype profile "
     "at the average sample."),

    ("pb", None),
    ("h", "4. Preprocessing"),
    ("t", ([1.15 * inch, 1.1 * inch, 1.1 * inch, 2.85 * inch], [
        ["Setting", "CyTOF", "scRNA-seq", "Why"],
        ["MarkerScaler mode", "<b>zscore</b>", "<b>none</b>",
         "SCT Pearson residuals are already ~N(0,1) per gene (measured mean &minus;0.0000, "
         "std 0.9858). Z-scoring on top undoes the stabilisation you paid for."],
        ["robust_zscore", "usable", "<b>NEVER</b>",
         "Broken on sparse data: any gene detected in &lt;50% of cells has median 0, so MAD is 0, "
         "scale clamps to eps=1e-8, and detections are multiplied by ~1e8 (data.py:63-70). "
         "Raises FloatingPointError."],
        ["Feature selection", "n/a", "HVGs first",
         "The package does no gene filtering. Subset before extract_matrix."],
        ["val_fraction", "0.2", "0.15&ndash;0.2", "&mdash;"],
        ["stratify_labels", "sample_ids", "sample_ids",
         "Splits within each group. Required with use_sample_offset."],
    ])),
    ("p",
     "For scRNA-seq the upstream route is SCTransform (or scanpy's analytic Pearson residuals) "
     "&rarr; HVGs by residual variance &rarr; h5ad, which is what notebook 300 in ProbAE_Deconv "
     "produces: residuals in X, dense float32, var indexed by gene."),

    ("h", "5. Parameters"),
    ("h", "5.1  K &mdash; number of archetypes"),
    ("p",
     "The only real modelling choice: a hypothesis about how many extreme programs your biology "
     "has. Sweep it. Selection is &sect;7."),
    ("h", "5.2  d &mdash; latent dimension (factorized decoder only)"),
    ("t", ([2.6 * inch, 1.8 * inch, 1.8 * inch], [
        ["", "CyTOF (~40 markers)", "scRNA-seq (2000 HVGs)"],
        ["Recommended", "8&ndash;16", "<b>16&ndash;32</b>"],
    ])),
    ("p",
     "<b>[measured]</b> w_recovery at K=5, 2000 HVGs: 0.520 (d=2), 0.852 (d=3), 0.722 (d=5), "
     "0.712 (d=8), <b>0.988 (d=16)</b>, 0.992 (d=32), 0.992 (d=64). The plateau is at "
     "<b>d ~ 16&ndash;32, not at d=K</b>."),
    ("p",
     "This is counter-intuitive and worth understanding. rank(A_hat) &le; min(K, d), so above "
     "d = K the rank constraint is <b>inactive</b> &mdash; d=16 and d=5 have identical "
     "expressiveness at K=5. The gap is pure optimisation: a wider product parametrisation "
     "descends better. <b>d is an optimisation setting, not a modelling choice.</b> Sweep it once "
     "to find the plateau, then hold it fixed while you sweep K."),
    ("p",
     "<b>What d does not control:</b> the dimension your cells live in. h = w @ Z with w on the "
     "simplex means cells occupy a (K&minus;1)-dimensional hull regardless of d. At d=64 with "
     "K=5, your cells still live on a 4-dimensional object. Reasoning about d by analogy to a "
     "VAE's latent &mdash; where z is free and its width really is capacity &mdash; is what makes "
     "people sweep it as though it mattered."),
    ("warn",
     "With decoder_type='direct', d does nothing at all: latent_dim is read only in the "
     "factorized branches (model.py:128-129, :263-268). But it IS fingerprinted, so sweeping d "
     "on a direct decoder produces one run directory per value containing bit-identical models, "
     "and skip_existing_runs will not dedupe them."),
    ("h", "5.3  decoder_type"),
    ("p",
     "<b>[measured]</b> Identical at 20 markers (w_recovery 0.993 each); at 2000 HVGs, "
     "<b>factorized 0.818 vs direct 0.575</b>. Use factorized for scRNA-seq. The gap is the "
     "reason; the mechanism is <b>not established</b> &mdash; it is not expressiveness (both "
     "A_hats are rank &le; K), and it is not &quot;direct has no intercept&quot; (because w sums "
     "to 1, adding a constant vector to every row of A shifts every cell alike). The likely cause "
     "is optimisation, consistent with &sect;5.2."),
    ("p",
     "<b>E is not a reason to prefer factorized.</b> It is tempting to mine E for gene modules. "
     "E_g ~ E_h does imply the genes load alike, so closeness in E is real information &mdash; "
     "but it tells you nothing A_hat does not. <b>[measured]</b> recovering planted gene groups "
     "by cosine: <b>AUC 1.000 from E and 1.000 from A_hat.T</b>, cross-seed agreement 0.764 vs "
     "0.775. Equivalent. Use either; A_hat.T is marginally safer (it is exactly the loadings, "
     "has no factorisation ambiguity even in principle, and exists for direct too)."),
    ("h", "5.4  model_type"),
    ("t", ([1.55 * inch, 2.3 * inch, 2.3 * inch], [
        ["w_recovery @ 2000 HVGs", "no regularisers", "with the lambda_* package"],
        ["deterministic", "0.834", "<b>0.992</b>"],
        ["probabilistic", "0.978", "0.856"],
    ])),
    ("p",
     "<b>[measured]</b> Read that as a 2&times;2 &mdash; the interaction is the whole story. "
     "<b>The KL and the explicit regularisers are substitutes, not complements.</b> Without "
     "lambda_*, the probabilistic model's KL does the regularising and helps a lot. With "
     "lambda_* already present, the KL over-regularises and costs you accuracy. The best "
     "configuration is <b>deterministic + regularisers</b>."),
    ("p",
     "So the reason to choose probabilistic is not accuracy. It is the per-cell posterior "
     "(mu_w, logvar_w): a shallow or ambiguous cell can report that its composition is uncertain "
     "rather than returning a confident point estimate. That is genuinely valuable on scRNA-seq, "
     "where depth varies enormously. But prob_eval_mode='mean' &mdash; the default, and what the "
     "mcf7 analysis notebook reads &mdash; takes the posterior mean and <b>discards the "
     "uncertainty</b>. If that is your workflow you are paying seven extra hyperparameters and "
     "some accuracy for something you then throw away. Use probabilistic when you will act on "
     "logvar_w; otherwise prefer deterministic."),

    ("pb", None),
    ("h", "6. Full parameter reference"),
    ("p",
     "All 41 run_config keys, enumerated from source. Blank recommendation = leave at default."),
    ("t", ([1.35 * inch, 0.72 * inch, 0.95 * inch, 0.95 * inch, 2.23 * inch], [
        ["Parameter", "Default", "CyTOF", "scRNA-seq", "Notes"],
        ["<b>Architecture</b>", "", "", "", ""],
        ["model_type", "deterministic", "either", "<b>deterministic</b>", "See 5.4. [measured]"],
        ["decoder_type", "factorized", "either", "<b>factorized</b>", "See 5.3. [measured]"],
        ["K", "required", "sweep 4&ndash;10", "sweep 3&ndash;10", "The real choice. See 8."],
        ["d", "required", "8&ndash;16", "<b>16&ndash;32</b>", "Optimisation, not capacity. [measured]"],
        ["hidden_dims", "(128,64)", "[128,64]", "[256,64]", "Encoder MLP widths."],
        ["dropout", "0.0", "0.0", "0.0&ndash;0.1", "Encoder hidden layers only."],
        ["<b>Simplex mapping</b>", "", "", "", ""],
        ["tau", "1.0", "0.7&ndash;1.0", "0.7&ndash;1.0",
         "Logits / tau. Lower = sharper weights. Must be &gt; 0."],
        ["logit_normalizer", "softmax", "<b>entmax</b>", "<b>entmax</b>",
         "entmax gives sparser weights (exact zeros reachable). Can NaN on MPS."],
        ["entmax_alpha", "1.5", "1.5", "1.5",
         "1.0=softmax, 1.5=entmax15, 2.0=sparsemax. Range [1,2]."],
        ["<b>Loss</b>", "", "", "", ""],
        ["recon_loss_type", "mse", "mse", "mse",
         "huber helps on heavy-tailed residuals, but mcRBM is hardcoded MSE &mdash; use mse if "
         "comparing."],
        ["huber_delta", "1.0", "1.0", "1.0", "Only if recon_loss_type=huber."],
        ["lambda_entropy", "<b>0.0</b>", "1e-3", "1e-3", "Penalises per-cell weight entropy."],
        ["lambda_sep", "<b>0.0</b>", "1e-3", "1e-3", "Pushes archetypes apart."],
        ["lambda_balance", "<b>0.0</b>", "5e-2", "5e-2",
         "Guards against dead archetypes. The most important of the three."],
        ["separation_mode", "cosine_sq", "cosine_sq", "cosine_sq",
         "cosine_mean | cosine_abs | cosine_sq | rbf"],
        ["balance_mode", "l2_uniform", "l2_uniform", "l2_uniform",
         "l2_uniform | kl_uniform | neg_entropy"],
        ["rbf_gamma", "1.0", "&mdash;", "&mdash;", "Only when separation_mode=rbf."],
        ["<b>Sample offset</b>", "", "", "", ""],
        ["use_sample_offset", "False", "multi-sample only", "multi-sample only",
         "See 3. Off first; enable only on identity archetypes. Fingerprinted only when True."],
        ["<b>Probabilistic only</b>", "", "", "", ""],
        ["use_residual_latent", "False", "either", "<b>False</b>",
         "On sparse data it absorbs dropout noise, defeating the simplex bottleneck."],
        ["residual_dim", "= d", "8", "8", "Only if the above is on."],
        ["beta_w", "1e-3", "1e-3", "1e-3", "KL on the logit posterior. Too high = collapse."],
        ["beta_r", "1e-3", "1e-3", "&mdash;", "KL on the residual latent."],
        ["kl_warmup_epochs", "0", "10", "10", "Ramps beta_* from 0. Prevents early collapse."],
        ["logvar_min", "&minus;10.0", "&minus;10", "&minus;10", "Posterior log-variance clamp."],
        ["logvar_max", "10.0", "5.0", "5.0", "Real configs use 5.0, tighter than default."],
        ["logvar_init_bias", "&minus;3.0", "&minus;3", "&minus;3", "Starts near-deterministic."],
        ["prob_eval_mode", "mean", "mean", "mean", "mean | sample | mc. mean discards logvar_w."],
        ["prob_eval_samples", "1", "3", "3", "Only used by mc."],
        ["<b>Optimisation</b>", "", "", "", ""],
        ["lr", "required", "1e-3", "1e-3", "Adam."],
        ["weight_decay", "<b>0.0</b>", "1e-5", "1e-4",
         "Adam not AdamW, so L2-into-gradient. B is excluded via its own param group."],
        ["epochs", "required", "1500", "400&ndash;3000", "A cap when early stopping is on."],
        ["batch_size", "required", "1024&ndash;2048", "512&ndash;1024", "&mdash;"],
        ["grad_clip_norm", "None", "5.0", "5.0", "Global norm. Useful with entmax."],
        ["<b>Early stopping</b>", "", "", "", ""],
        ["early_stopping", "True", "True", "True", "Tracks best val_recon."],
        ["patience", "20", "20&ndash;60", "30&ndash;60", "Epochs without improvement."],
        ["min_delta", "0.0", "1e-4", "1e-4", "Improvement threshold."],
        ["restore_best_weights", "True", "True", "True",
         "Saved W/B come from best_epoch, not the last."],
        ["<b>Reproducibility</b>", "", "", "", ""],
        ["seed", "0", "<b>[7,17,23]</b>", "<b>[7,17,23]</b>",
         "Sweep it. Without seeds you cannot select K. See 7. [measured]"],
        ["deterministic", "True", "True", "True", "Torch/cuDNN flag. Unrelated to model_type."],
        ["device", "auto", "<b>cpu</b> if small", "cpu / cuda",
         "auto picks MPS, which is 11.6x SLOWER than CPU on small tensors. [measured]"],
        ["<b>Logging (not fingerprinted)</b>", "", "", "", ""],
        ["print_every", "10", "", "", "Epoch 1, multiples, and last."],
        ["progress_epoch", "True", "", "", "tqdm per epoch."],
        ["progress_sweep", "True", "", "", "Read off base_config, not a run_sweep arg."],
        ["skip_existing_runs", "True", "", "", "Reloads a matching-fingerprint run dir."],
        ["run_name", "&mdash;", "", "", "Forces a folder name."],
    ])),

    ("pb", None),
    ("h", "7. The NB decoder &mdash; which is not in CyEmbed"),
    ("warn",
     "CyEmbed cannot do this. Its reconstruction loss is mse or huber only (losses.py:17-21); "
     "there is no negative-binomial path anywhere in the package. Pearson residuals are the "
     "workaround that makes a Gaussian loss defensible on count data. If you want to model "
     "counts directly you must use a DIFFERENT codebase: ProbAE_Deconv's cytof_archetypes."),
    ("h", "7.1  Which route to take"),
    ("t", ([1.1 * inch, 2.55 * inch, 2.55 * inch], [
        ["", "Residuals + CyEmbed", "NB + ProbAE_Deconv"],
        ["Likelihood", "Gaussian, constant variance, on transformed data",
         "Negative binomial, on counts"],
        ["Principled?", "A workaround &mdash; you transform until MSE stops being wrong",
         "<b>Yes</b> &mdash; models the count process, no transform at all"],
        ["Input", "X = SCT Pearson residuals", "X = <b>raw counts</b>"],
        ["Needed for", "The mcRBM comparison (mcRBM's loss is hardcoded MSE)",
         "Defensibility. A reviewer will prefer it, and KL-NMF-style count likelihoods are the "
         "standard objection to the whole residual detour."],
        ["Config", "configs/sct_gaussian_k_sweep.yaml", "configs/bck44_scrna_nb_k_sweep.yaml"],
        ["Cells (BCK_44)", "330 (tumour-only, filtered upstream in Seurat)",
         "352 (its own scanpy QC)"],
    ])),
    ("p",
     "<b>Run both and compare.</b> If NB clearly wins, the residual transform is costing you "
     "signal and you should know that. If they agree, the residual route is fine and you keep "
     "access to CyEmbed and to the mcRBM comparison. Note the two are <b>not comparable "
     "cell-for-cell</b> &mdash; different QC gives BCK_44 352 cells one way and 330 the other, so "
     "reconcile on cell_id first."),
    ("h", "7.2  Why NB cannot simply be added to CyEmbed"),
    ("p",
     "The obvious question is why not port NB into CyEmbed and have one tool. The answer is "
     "structural: <b>NB and the factorized decoder are incompatible.</b>"),
    ("p",
     "CyEmbed decodes x_hat = w Z E.T + b &mdash; linear, and <b>signed by construction</b>. NB "
     "needs mu &ge; 0, and there are only two ways to get it. An <b>exp link</b> "
     "(mu = exp(w Z E.T + b) &middot; lib) moves the convex combination into log space, which "
     "means <b>geometric</b> mixing in count space &mdash; not what archetypal analysis claims, "
     "and not what cNMF, LDA or ProbAE do. <b>Non-negative profiles</b> "
     "(mu = lib &middot; (w @ P) with P = softplus(Z E.T + b)) give the right arithmetic mixing, "
     "but then A_hat is no longer a linear factorisation: rank(A_hat) &le; K stops holding, E "
     "stops being a linear gene embedding, and &sect;5.2 evaporates."),
    ("p",
     "So the correct geometry costs you the factorized decoder &mdash; the configuration measured "
     "at <b>0.818 vs direct's 0.575</b> at 2000 HVGs. You would trade CyEmbed's one measured "
     "architectural advantage to gain NB. ProbAE gets the right geometry precisely <i>because</i> "
     "it learns the archetype profile directly as (K, M) and softmaxes it over genes "
     "(probabilistic_archetypal_ae.py:80-90): rho = softmax(archetype_logits, dim=-1); "
     "rho_i = w @ rho; mu = lib &middot; rho_i. A (K,M) profile is what NB wants; a factorized "
     "Z@E.T is what NB cannot have."),
    ("p",
     "<b>If you need NB and per-sample correction together</b> &mdash; a combination that exists "
     "nowhere today, since ProbAE has no batch/patient conditioning of any kind &mdash; the cheap "
     "edit is in ProbAE, not CyEmbed. About five lines in _decode_nb: "
     "mu = clamp(lib &middot; rho_i &middot; exp(B_eff[s]), min=1e-8). Multiplicative in count "
     "space is additive in log space, the correct count-model analogue of CyEmbed's "
     "additive-in-residual-space B, and what scVI does for batch. Reuse the rest of the design "
     "validated here: B created only when enabled, centred zero-sum, warm-started at the "
     "empirical per-sample mean, out of weight decay, stratified split."),
    ("h", "7.3  The parameters do not carry over"),
    ("p",
     "A different package with different names. Nothing in &sect;6 applies to the NB route:"),
    ("t", ([1.7 * inch, 1.7 * inch, 2.8 * inch], [
        ["CyEmbed", "ProbAE_Deconv (NB)", "Note"],
        ["K", "n_archetypes", "&mdash;"],
        ["d", "&mdash;", "No factorized/direct distinction."],
        ["hidden_dims", "encoder_hidden_dims", "NB config uses [512,128]."],
        ["lambda_entropy", "entropy_reg_weight", "1e-3."],
        ["lambda_sep", "diversity_reg_weight", "1e-3."],
        ["lambda_balance", "&mdash;", "No direct equivalent; variance_reg_weight is separate."],
        ["epochs", "max_epochs", "5000 in the NB config."],
        ["grad_clip_norm", "grad_clip", "1.0."],
        ["recon_loss_type", "loss.type = nb_nll", "The whole point."],
        ["&mdash;", "decoder_family = nb", "Selects the count decoder."],
        ["&mdash;", "use_observed_library_size", "true &rarr; depth recomputed from X per split."],
        ["&mdash;", "dispersion", "'gene' &mdash; one NB dispersion per gene."],
    ])),
    ("h", "7.4  The NB config"),
    ("code",
     "# configs/bck44_scrna_nb_k_sweep.yaml (abridged)\n"
     "raw_data:   {tenx_h5: '.../BCK_44/filtered_feature_bc_matrix.h5'}\n"
     "processed_data: {output_h5ad: 'data/bck44_scrna_hvg_counts.h5ad'}\n"
     "qc:                                   # the NB route does its OWN scanpy QC\n"
     "  min_genes_per_cell: 200   max_genes_per_cell: 9000\n"
     "  max_counts_per_cell: 80000  min_counts_per_cell: 500\n"
     "  max_pct_mt: 35.0   max_pct_ribo: 65.0   min_cells_per_gene: 3\n"
     "  run_doublet_scoring: true  filter_strong_doublet: true  doublet_score_max: 0.35\n"
     "hvg:    {n_top_genes: 2000, flavor: 'seurat', span: 0.3}\n"
     "sweep:  {k_values: [4,5,6,7,8], seeds: [42]}      # <- one seed; see 8.2\n"
     "model:\n"
     "  type: 'archetypal_autoencoder'\n"
     "  decoder_family: 'nb'                # <- the count decoder\n"
     "  n_archetypes: 5\n"
     "  encoder_hidden_dims: [512, 128]\n"
     "  dropout: 0.1\n"
     "  use_observed_library_size: true\n"
     "  size_factor_key: null\n"
     "  dispersion: 'gene'\n"
     "data:\n"
     "  encoder_input: 'log1p_normalized'   # a MODE, not a layer name\n"
     "  decoder_target: 'raw_counts'        # must be exactly this\n"
     "loss:\n"
     "  type: 'nb_nll'\n"
     "  entropy_reg_weight: 1.0e-3\n"
     "  diversity_reg_weight: 1.0e-3\n"
     "  variance_reg_weight: 0.0\n"
     "training:\n"
     "  batch_size: 2048   lr: 5.0e-3   weight_decay: 1.0e-4\n"
     "  max_epochs: 5000   patience: 20   grad_clip: 1.0"),
    ("h", "7.5  Gotchas specific to the NB route"),
    ("t", ([1.5 * inch, 4.7 * inch], [
        ["Gotcha", "Detail"],
        ["<b>X must BE raw counts</b>",
         "_prepare_nb_split clips X to &ge;0 and uses it directly as the decoder target "
         "(trainer.py:132). Point input_path at the counts h5ad, not the residual one."],
        ["<b>encoder_input / decoder_target are MODE STRINGS, not layer names</b>",
         "log1p_normalized is derived arithmetically from X &mdash; x / library * 1e4 then log1p "
         "(trainer.py:115-123). The counts and log1p_norm <b>layers in the h5ad are written for "
         "provenance and never read.</b> This is the easiest thing here to misread."],
        ["<b>decoder_target must be 'raw_counts'</b>",
         "Anything else raises: 'nb decoder currently requires data.decoder_target=raw_counts' "
         "(trainer.py:171-173)."],
        ["<b>The preprocessing: block is IGNORED</b>",
         "For count decoders trainer.py:174 sets preprocessor = None. So the transform / "
         "normalization / clip_min / clip_max keys &mdash; and arcsinh_cofactor: 5.0 &mdash; in "
         "bck44_scrna_nb_k_sweep.yaml are <b>dead config</b>. They are serialised and read by "
         "nothing. Do not tune them expecting an effect."],
        ["<b>Library size is recomputed from X per split</b>",
         "use_observed_library_size: true with size_factor_key: null falls through to the "
         "observed depth. You do not supply size factors."],
        ["<b>Different QC &rArr; different cells</b>",
         "The NB route runs its own scanpy QC (mito/ribo caps, doublet scoring); the SCT route "
         "inherits tumour-only filtering from Seurat upstream. Same sample, different cell "
         "counts (BCK_44: 352 vs 330). The two routes are not comparable cell-for-cell."],
        ["<b>seeds: [42]</b>",
         "The NB config sweeps a single seed, like the SCT one. Everything in &sect;8.2 applies."],
    ])),

    ("pb", None),
    ("h", "8. How to tell whether it is working"),
    ("h", "8.1  Choosing K"),
    ("p", "<b>[measured]</b> against a planted ground truth of K=5, 2000 HVGs:"),
    ("t", ([0.5 * inch, 1.0 * inch, 0.7 * inch, 0.95 * inch, 1.05 * inch, 1.1 * inch], [
        ["K", "mean|cos A_hat|", "val_recon", "dead", "stability", "w_rec (oracle)"],
        ["3", "0.209", "1.2953", "0.0", "1.000", "0.490"],
        ["4", "0.324", "1.3291", "0.0", "0.564", "0.426"],
        ["<b>5</b>", "<b>0.078</b>", "<b>1.1920</b>", "0.3", "0.740", "<b>0.732</b>"],
        ["6", "1.000", "1.2949", "1.3", "0.702", "0.410"],
        ["7", "1.000", "1.5070", "1.0", "<b>1.000</b>", "0.029"],
        ["8", "0.985", "1.2261", "2.7", "0.679", "0.430"],
    ])),
    ("p",
     "<b>Lead with archetype redundancy.</b> mean|off-diagonal| of "
     "cosine_similarity_matrix(A_hat) has a sharp minimum at the truth (0.078) and then screams: "
     "<b>1.000 means the archetypes have become literally identical.</b> Above K_TRUE this model "
     "does not gracefully split archetypes, it <b>collapses</b> &mdash; almost certainly "
     "lambda_balance forcing uniform usage across more archetypes than the data supports. Use "
     "mean, not max: max hits 1.000 at K=4 from a single duplicate pair while mean is 0.324."),
    ("p",
     "<b>val_recon works too</b>, and is not the trap it looks like. It is <i>not</i> monotone in "
     "K, because early stopping plus the lambda_* package makes excess archetypes cost on "
     "held-out data. Verify non-monotonicity on your own data: if it falls all the way to your "
     "largest K, it is not selecting and you must fall back."),
    ("p",
     "<b>Dead archetypes work</b>, but use a <i>relative</i> threshold (usage &lt; 0.5/K). "
     "CyEmbed's dead_archetypes_lt_1pct is <b>absolute</b> (w_bar &lt; 0.01), which gets less "
     "strict as K rises (0.01 is 5% of uniform at K=5 but 20% at K=20), biasing you toward large "
     "K."),
    ("warn",
     "Cross-seed stability FAILS as a selector. It picks K=7 with a perfect 1.000 &mdash; where "
     "the model recovers essentially nothing (w_recovery 0.029) with a dead archetype. Every "
     "seed collapses to the SAME degenerate solution, so agreement is perfect. Use stability to "
     "reject a K, never to select one."),
    ("h", "8.2  Seeds are not optional"),
    ("p",
     "<b>[measured]</b> Two runs of an identical config gave val_recon 1.2586 and 1.3291 at K=4 "
     "&mdash; <b>5% run-to-run variance</b> from torch nondeterminism alone. The winning K beat "
     "its rivals by ~6%. <b>On a single seed you cannot distinguish signal from noise</b>, and "
     "gene modules are only ~0.77 correlated across seeds. This dominates every other "
     "recommendation here. No code change is needed:"),
    ("code",
     "SWEEP_GRID = {..., 'seed': [7, 17, 23]}\n"
     "# build_sweep_configs is a cartesian product and _config_fingerprint includes seed,\n"
     "# so each lands in its own run dir. Aggregate per-K across seeds and only believe\n"
     "# a difference larger than the seed spread."),
    ("h", "8.3  Diagnostics to run every time"),
    ("t", ([1.3 * inch, 2.0 * inch, 2.9 * inch], [
        ["Check", "How", "Failure looks like"],
        ["Depth leakage (scRNA)", "Spearman each W[:,k] vs total counts",
         "Any archetype tracking depth &rArr; preprocessing failed and everything downstream is "
         "suspect. Needs a depth covariate joined on cell_id &mdash; the SCT h5ad has none."],
        ["Sample leakage", "summarize_by_group(W, sample_ids)",
         "dominant_fraction ~1.0 in one sample, ~0 elsewhere &rArr; identity archetype."],
        ["Dead archetypes", "usage &lt; 0.5/K (relative!)", "K too high."],
        ["Archetype redundancy", "mean|off-diag| cosine_similarity_matrix(A_hat)",
         "Rising toward 1.000 &rArr; archetypes collapsing; K too high."],
        ["Convergence", "history.csv val_recon still descending at the last epoch",
         "Truncated, not converged &mdash; comparisons are meaningless. This produced a spurious "
         "4.5x 'regression' that vanished at convergence."],
        ["Cells outside the hull", "per-cell mean residual, high percentiles",
         "Cells the archetype model fails on: K too low, or a genuinely distinct state."],
        ["Biology", "UCell signatures joined on cell_id, Spearman vs W",
         "The only criterion here that is not self-referential."],
    ])),
    ("h", "8.4  Outputs worth reading that are easy to miss"),
    ("t", ([1.2 * inch, 5.0 * inch], [
        ["Output", "What it tells you"],
        ["weight_entropy(W)",
         "Each cell's position on the continuum. Low = the cell sits at a vertex (a pure "
         "program); high = it is interpolating between them. This is the "
         "archetypal-analysis-specific readout and the coordinate that justifies choosing "
         "archetypes over NMF for dosage gradients and treatment continua."],
        ["A_hat.T",
         "Gene loadings across archetypes. cosine_similarity_matrix(A_hat.T) gives gene modules "
         "&mdash; identifiable, interpretable, and available for both decoders."],
        ["B",
         "The per-sample, per-gene offsets. Which genes shift most tells you whether the sample "
         "effect is technical (ribosomal / mito / housekeeping) or biological (a coherent "
         "program &mdash; in which case stop correcting it)."],
        ["b",
         "The global per-gene baseline. On centred residuals with no offset it should be ~0; far "
         "from 0 means your residuals are not actually centred."],
        ["residuals",
         "Per-cell and per-gene reconstruction error. A targeted list of what the hull fails to "
         "span, not merely an error metric."],
    ])),

    ("pb", None),
    ("h", "9. Starting configurations"),
    ("h", "CyTOF (~40 markers, arcsinh-transformed)"),
    ("code",
     "GLOBAL_CFG = {'seed': 7, 'deterministic': True, 'device': 'auto'}\n"
     "scaler, _ = fit_scaler(bundle.X, mode='zscore',\n"
     "                       sample_ids=bundle.sample_ids, balanced_max_per_sample=2000)\n"
     "BASE = {'epochs': 1500, 'early_stopping': True, 'patience': 20, 'min_delta': 0.0,\n"
     "        'restore_best_weights': True, 'weight_decay': 1e-5, 'dropout': 0.0,\n"
     "        'logit_normalizer': 'entmax', 'entmax_alpha': 1.5, 'grad_clip_norm': 5.0,\n"
     "        'separation_mode': 'cosine_sq', 'balance_mode': 'l2_uniform'}\n"
     "GRID = {'model_type': ['deterministic'], 'decoder_type': ['factorized'],\n"
     "        'K': [4,5,6,7,8], 'd': [8, 16], 'hidden_dims': [[128,64]],\n"
     "        'lr': [1e-3], 'batch_size': [2048], 'recon_loss_type': ['mse'],\n"
     "        'lambda_entropy': [1e-3], 'lambda_sep': [1e-3], 'lambda_balance': [5e-2],\n"
     "        'tau': [0.7, 1.0], 'seed': [7, 17, 23]}"),
    ("h", "scRNA-seq (2000 HVGs, SCT Pearson residuals)"),
    ("code",
     "GLOBAL_CFG = {'seed': 7, 'deterministic': True, 'device': 'cpu'}\n"
     "scaler, _ = fit_scaler(bundle.X, mode='none')      # residuals are already ~N(0,1)\n"
     "BASE = {'epochs': 3000, 'early_stopping': True, 'patience': 30, 'min_delta': 1e-4,\n"
     "        'restore_best_weights': True, 'weight_decay': 1e-4, 'dropout': 0.0,\n"
     "        'logit_normalizer': 'entmax', 'entmax_alpha': 1.5, 'grad_clip_norm': 5.0,\n"
     "        'separation_mode': 'cosine_sq', 'balance_mode': 'l2_uniform'}\n"
     "GRID = {'model_type': ['deterministic'], 'decoder_type': ['factorized'],\n"
     "        'K': [3,4,5,6,7,8], 'd': [16, 32], 'hidden_dims': [[256,64]],\n"
     "        'lr': [1e-3], 'batch_size': [512], 'recon_loss_type': ['mse'],\n"
     "        'lambda_entropy': [1e-3], 'lambda_sep': [1e-3], 'lambda_balance': [5e-2],\n"
     "        'tau': [0.7, 1.0], 'seed': [7, 17, 23],\n"
     "        'use_sample_offset': [False]}   # multi-sample: enable only after diagnosing"),
    ("p",
     "Note base_config is the <b>merge</b> {**GLOBAL_CFG, **BASE}, not the training dict alone. "
     "And skip_existing_runs / progress_sweep are read off base_config, not passed to run_sweep."),

    ("h", "10. The traps, in one place"),
    ("t", ([0.3 * inch, 5.9 * inch], [
        ["#", "Trap"],
        ["1", "<b>robust_zscore on sparse data</b> &rarr; &times;1e8 blowup, FloatingPointError "
              "(data.py:63-70)."],
        ["2", "<b>mode='zscore' on Pearson residuals</b> &rarr; undoes the variance "
              "stabilisation."],
        ["3", "<b>sample_col typos fail silently</b> &rarr; sample_ids=None; stratification and "
              "balanced scaling quietly disabled. Assert it resolved."],
        ["4", "<b>marker_names is ignored</b> unless source='obsm'."],
        ["5", "<b>All three lambda_* default to 0.0</b> &mdash; omit them and you have no "
              "regularisation, and val_recon may stop working as a K-selector."],
        ["6", "<b>weight_decay defaults to 0.0.</b>"],
        ["7", "<b>d does nothing with decoder_type='direct'</b> but is still fingerprinted."],
        ["8", "<b>d ~ K is the wrong regime</b> &mdash; use 16&ndash;32."],
        ["9", "<b>One seed cannot select K</b> &mdash; the noise is the size of the signal."],
        ["10", "<b>device='auto' picks MPS</b>, up to 11.6&times; slower than CPU on small "
               "models."],
        ["11", "<b>Fixed epochs &ne; converged.</b> Use early stopping; check the curve."],
        ["12", "<b>Stability rewards reproducible collapse</b> &mdash; reject with it, do not "
               "select with it."],
        ["13", "<b>dead_archetypes_lt_1pct is absolute</b>, so it under-detects at large K."],
        ["14", "<b>use_sample_offset needs a stratified split</b>, or B is untrained for "
               "held-out samples."],
    ])),
    ("h", "11. Provenance"),
    ("p",
     "Claims marked [measured] come from: tools/verify_sample_offset.py (planted-shift recovery, "
     "20 markers, 8 patients, 3 seeds); tools/verify_sample_offset_scrna.py (2000 HVGs from "
     "simulated NB counts through analytic Pearson residuals, ~26% median gene detection, "
     "decoder &times; model_type &times; regulariser matrix); tools/select_k_and_d_scrna.py "
     "(d-sweep at fixed true K, K-sweep scoring four criteria against ground truth). All are "
     "single synthetic regimes with 2&ndash;3 seeds. They should inform your sweep, not replace "
     "it. Everything not marked [measured] is the code's default or reasoning from it, and is "
     "not evidence."),
]


def build_pdf() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=19,
        leading=23, alignment=TA_CENTER, textColor=ACCENT, spaceAfter=8)
    subtitle_style = ParagraphStyle(
        "SubtitleStyle", parent=styles["BodyText"], fontName="Helvetica", fontSize=9,
        leading=12, alignment=TA_CENTER, textColor=MUTED, spaceAfter=14)
    heading_style = ParagraphStyle(
        "HeadingStyle", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=12,
        leading=15, alignment=TA_LEFT, textColor=ACCENT, spaceBefore=9, spaceAfter=5)
    body_style = ParagraphStyle(
        "BodyStyle", parent=styles["BodyText"], fontName="Helvetica", fontSize=9,
        leading=12.5, alignment=TA_LEFT, spaceAfter=6)
    code_style = ParagraphStyle(
        "CodeStyle", parent=styles["BodyText"], fontName="Courier", fontSize=7.4,
        leading=9.4, textColor=colors.HexColor("#1d2b3a"), backColor=BAND,
        borderPadding=5, spaceBefore=2, spaceAfter=7)
    warn_style = ParagraphStyle(
        "WarnStyle", parent=styles["BodyText"], fontName="Helvetica", fontSize=8.6,
        leading=11.5, textColor=WARN, backColor=colors.HexColor("#fbf0f0"),
        borderPadding=6, borderColor=colors.HexColor("#e2c3c3"), borderWidth=0.6,
        spaceBefore=2, spaceAfter=8)
    cell_style = ParagraphStyle(
        "CellStyle", parent=styles["BodyText"], fontName="Helvetica", fontSize=7.3,
        leading=9.2, alignment=TA_LEFT, spaceAfter=0)
    head_cell_style = ParagraphStyle(
        "HeadCellStyle", parent=cell_style, fontName="Helvetica-Bold", textColor=colors.white)

    def make_table(widths, rows):
        data = []
        for r_i, row in enumerate(rows):
            st = head_cell_style if r_i == 0 else cell_style
            data.append([Paragraph(str(c), st) for c in row])
        t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        style = [
            ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("GRID", (0, 0), (-1, -1), 0.4, RULE),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]
        # Shade section-divider rows (those whose first cell is bold and rest are empty).
        for r_i, row in enumerate(rows[1:], start=1):
            if str(row[0]).startswith("<b>") and all(str(c) == "" for c in row[1:]):
                style.append(("BACKGROUND", (0, r_i), (-1, r_i), BAND))
        t.setStyle(TableStyle(style))
        return t

    story: list = [
        Spacer(1, 0.25 * inch),
        Paragraph(TITLE, title_style),
        Paragraph(SUBTITLE, subtitle_style),
    ]
    for kind, payload in SECTIONS:
        if kind == "h":
            story.append(Paragraph(str(payload), heading_style))
        elif kind == "p":
            story.append(Paragraph(str(payload), body_style))
        elif kind == "code":
            story.append(Preformatted(str(payload), code_style))
        elif kind == "warn":
            story.append(Paragraph(str(payload), warn_style))
        elif kind == "t":
            widths, rows = payload  # type: ignore[misc]
            story.append(Spacer(1, 0.02 * inch))
            story.append(make_table(widths, rows))
            story.append(Spacer(1, 0.09 * inch))
        elif kind == "pb":
            story.append(PageBreak())

    doc = SimpleDocTemplate(
        str(REPORT_PATH), pagesize=letter,
        leftMargin=0.72 * inch, rightMargin=0.72 * inch,
        topMargin=0.62 * inch, bottomMargin=0.6 * inch,
        title=TITLE, author="CyEmbed",
    )
    doc.build(story)
    print(f"wrote {REPORT_PATH}")


if __name__ == "__main__":
    build_pdf()
