from __future__ import annotations

from typing import Iterable

import torch
from entmax import entmax15, entmax_bisect, sparsemax
from torch import Tensor, nn


def simplex_weights_from_logits(
    logits: Tensor,
    *,
    tau: float,
    logit_normalizer: str = "softmax",
    entmax_alpha: float = 1.5,
) -> Tensor:
    """Map logits onto the simplex with the configured normalizer."""
    tau = max(float(tau), 1e-6)
    normalized = str(logit_normalizer).lower()
    scaled_logits = logits / tau
    if normalized == "softmax":
        return torch.softmax(scaled_logits, dim=-1)
    if normalized == "entmax":
        if abs(float(entmax_alpha) - 1.0) < 1e-6:
            return torch.softmax(scaled_logits, dim=-1)
        if abs(float(entmax_alpha) - 1.5) < 1e-6:
            return entmax15(scaled_logits, dim=-1)
        if abs(float(entmax_alpha) - 2.0) < 1e-6:
            return sparsemax(scaled_logits, dim=-1)
        return entmax_bisect(scaled_logits, alpha=float(entmax_alpha), dim=-1)
    raise ValueError("logit_normalizer must be 'softmax' or 'entmax'.")


class EncoderMLP(nn.Module):
    """MLP encoder that outputs archetype logits."""

    def __init__(
        self,
        input_dim: int,
        num_archetypes: int,
        hidden_dims: Iterable[int] = (128, 64),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], num_archetypes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


def reparameterize_gaussian(mu: Tensor, logvar: Tensor) -> Tensor:
    """Reparameterization trick for diagonal Gaussian posteriors."""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + std * eps


def gaussian_kl_standard_normal(mu: Tensor, logvar: Tensor) -> Tensor:
    """Per-sample KL(q||p) where q=N(mu,diag(exp(logvar))) and p=N(0,I)."""
    return -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)


def apply_sample_offset(x_hat: Tensor, B: Tensor | None, sample_idx: Tensor | None) -> Tensor:
    """Add the centred per-patient offset B_eff[sample_idx] to x_hat.

    B is centred across patients (zero-sum) so the global baseline stays in b/A and B carries
    only per-patient deviation. Returns x_hat unchanged when the offset is disabled (B is None)
    or when no sample index is supplied (e.g. generating at the average patient baseline).
    """
    if B is None or sample_idx is None:
        return x_hat
    B_eff = B - B.mean(0, keepdim=True)
    return x_hat + B_eff[sample_idx]


class ArchetypeEmbeddingModel(nn.Module):
    """Archetype model with simplex cell weights and configurable decoder."""

    def __init__(
        self,
        num_markers: int,
        num_archetypes: int,
        latent_dim: int,
        hidden_dims: Iterable[int] = (128, 64),
        tau: float = 1.0,
        logit_normalizer: str = "softmax",
        entmax_alpha: float = 1.5,
        decoder_type: str = "factorized",
        dropout: float = 0.0,
        n_samples: int | None = None,
        use_sample_offset: bool = False,
    ) -> None:
        super().__init__()
        if decoder_type not in {"factorized", "direct"}:
            raise ValueError("decoder_type must be 'factorized' or 'direct'.")
        if tau <= 0:
            raise ValueError("tau must be > 0.")
        if str(logit_normalizer).lower() not in {"softmax", "entmax"}:
            raise ValueError("logit_normalizer must be 'softmax' or 'entmax'.")
        if not 1.0 <= float(entmax_alpha) <= 2.0:
            raise ValueError("entmax_alpha must be in [1.0, 2.0].")

        self.num_markers = int(num_markers)
        self.num_archetypes = int(num_archetypes)
        self.latent_dim = int(latent_dim)
        self.decoder_type = decoder_type
        self.tau = float(tau)
        self.logit_normalizer = str(logit_normalizer).lower()
        self.entmax_alpha = float(entmax_alpha)
        self.use_sample_offset = bool(use_sample_offset)

        self.encoder = EncoderMLP(
            input_dim=num_markers,
            num_archetypes=num_archetypes,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

        if self.decoder_type == "factorized":
            self.Z = nn.Parameter(torch.randn(num_archetypes, latent_dim) * 0.02)
            self.E = nn.Parameter(torch.randn(num_markers, latent_dim) * 0.02)
            self.b = nn.Parameter(torch.zeros(num_markers))
            self.A = None
        else:
            self.A = nn.Parameter(torch.randn(num_archetypes, num_markers) * 0.02)
            self.Z = None
            self.E = None
            self.b = None

        # Per-patient additive decoder intercept: B[s] is added to x_hat for cells of
        # sample s, so archetypes model deviation from each patient's own baseline. For the
        # direct decoder (self.b is None) this becomes the decoder's only intercept.
        if self.use_sample_offset:
            if n_samples is None or int(n_samples) <= 0:
                raise ValueError("n_samples must be a positive int when use_sample_offset=True.")
            self.B = nn.Parameter(torch.zeros(int(n_samples), num_markers))
        else:
            self.B = None

    def _simplex_weights(self, logits: Tensor, tau_override: float | None = None) -> Tensor:
        tau = float(tau_override if tau_override is not None else self.tau)
        return simplex_weights_from_logits(
            logits,
            tau=tau,
            logit_normalizer=self.logit_normalizer,
            entmax_alpha=self.entmax_alpha,
        )

    def decode_from_weights(
        self, w: Tensor, sample_idx: Tensor | None = None
    ) -> tuple[Tensor | None, Tensor, Tensor]:
        if self.decoder_type == "factorized":
            h = w @ self.Z
            x_hat = h @ self.E.T + self.b
            a_hat = self.Z @ self.E.T + self.b
        else:
            h = None
            x_hat = w @ self.A
            a_hat = self.A
        # a_hat stays batch-independent (archetype profile at the average patient baseline).
        x_hat = apply_sample_offset(x_hat, self.B, sample_idx)
        return h, x_hat, a_hat

    def forward(
        self,
        x: Tensor,
        tau_override: float | None = None,
        sample_idx: Tensor | None = None,
    ) -> dict[str, Tensor | None]:
        u = self.encoder(x)
        w = self._simplex_weights(u, tau_override=tau_override)
        h, x_hat, a_hat = self.decode_from_weights(w, sample_idx=sample_idx)
        return {
            "U": u,
            "W": w,
            "H": h,
            "X_hat": x_hat,
            "A_hat": a_hat,
        }

    def archetype_separation_tensor(self) -> Tensor:
        if self.decoder_type == "factorized":
            return self.Z
        return self.A


class ProbabilisticArchetypeModel(nn.Module):
    """Probabilistic archetype model with logistic-normal archetype weights."""

    def __init__(
        self,
        num_markers: int,
        num_archetypes: int,
        latent_dim: int,
        hidden_dims: Iterable[int] = (128, 64),
        tau: float = 1.0,
        logit_normalizer: str = "softmax",
        entmax_alpha: float = 1.5,
        decoder_type: str = "factorized",
        use_residual_latent: bool = False,
        residual_dim: int = 8,
        dropout: float = 0.0,
        logvar_min: float = -10.0,
        logvar_max: float = 10.0,
        logvar_init_bias: float = -3.0,
        n_samples: int | None = None,
        use_sample_offset: bool = False,
    ) -> None:
        super().__init__()
        if decoder_type not in {"factorized", "direct"}:
            raise ValueError("decoder_type must be 'factorized' or 'direct'.")
        if tau <= 0:
            raise ValueError("tau must be > 0.")
        if str(logit_normalizer).lower() not in {"softmax", "entmax"}:
            raise ValueError("logit_normalizer must be 'softmax' or 'entmax'.")
        if not 1.0 <= float(entmax_alpha) <= 2.0:
            raise ValueError("entmax_alpha must be in [1.0, 2.0].")

        self.num_markers = int(num_markers)
        self.num_archetypes = int(num_archetypes)
        self.latent_dim = int(latent_dim)
        self.decoder_type = decoder_type
        self.tau = float(tau)
        self.logit_normalizer = str(logit_normalizer).lower()
        self.entmax_alpha = float(entmax_alpha)
        self.use_residual_latent = bool(use_residual_latent)
        self.residual_dim = int(residual_dim)
        self.logvar_min = float(logvar_min)
        self.logvar_max = float(logvar_max)
        self.use_sample_offset = bool(use_sample_offset)

        trunk_dims = [num_markers, *hidden_dims]
        trunk_layers: list[nn.Module] = []
        for in_dim, out_dim in zip(trunk_dims[:-1], trunk_dims[1:]):
            trunk_layers.append(nn.Linear(in_dim, out_dim))
            trunk_layers.append(nn.ReLU())
            if dropout > 0:
                trunk_layers.append(nn.Dropout(dropout))
        self.trunk = nn.Sequential(*trunk_layers) if trunk_layers else nn.Identity()
        head_in_dim = trunk_dims[-1]

        self.mu_w_head = nn.Linear(head_in_dim, num_archetypes)
        self.logvar_w_head = nn.Linear(head_in_dim, num_archetypes)
        nn.init.constant_(self.logvar_w_head.bias, logvar_init_bias)

        if self.use_residual_latent:
            self.mu_r_head = nn.Linear(head_in_dim, residual_dim)
            self.logvar_r_head = nn.Linear(head_in_dim, residual_dim)
            nn.init.constant_(self.logvar_r_head.bias, logvar_init_bias)
        else:
            self.mu_r_head = None
            self.logvar_r_head = None

        if self.decoder_type == "factorized":
            self.Z = nn.Parameter(torch.randn(num_archetypes, latent_dim) * 0.02)
            self.E = nn.Parameter(torch.randn(num_markers, latent_dim) * 0.02)
            self.b = nn.Parameter(torch.zeros(num_markers))
            self.A = None
            if self.use_residual_latent and self.residual_dim != self.latent_dim:
                self.P_r = nn.Parameter(torch.randn(self.residual_dim, self.latent_dim) * 0.02)
            else:
                self.P_r = None
            self.G = None
        else:
            self.A = nn.Parameter(torch.randn(num_archetypes, num_markers) * 0.02)
            self.Z = None
            self.E = None
            self.b = None
            self.P_r = None
            if self.use_residual_latent:
                self.G = nn.Parameter(torch.randn(self.residual_dim, num_markers) * 0.02)
            else:
                self.G = None

        # Per-patient additive decoder intercept (see ArchetypeEmbeddingModel for rationale).
        # For the direct decoder (self.b is None) this becomes the decoder's only intercept.
        if self.use_sample_offset:
            if n_samples is None or int(n_samples) <= 0:
                raise ValueError("n_samples must be a positive int when use_sample_offset=True.")
            self.B = nn.Parameter(torch.zeros(int(n_samples), num_markers))
        else:
            self.B = None

    def _simplex_weights(self, logits: Tensor, tau_override: float | None = None) -> Tensor:
        tau = float(tau_override if tau_override is not None else self.tau)
        return simplex_weights_from_logits(
            logits,
            tau=tau,
            logit_normalizer=self.logit_normalizer,
            entmax_alpha=self.entmax_alpha,
        )

    def archetype_separation_tensor(self) -> Tensor:
        if self.decoder_type == "factorized":
            return self.Z
        return self.A

    def _encode(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor | None, Tensor | None]:
        h = self.trunk(x)
        mu_w = self.mu_w_head(h)
        logvar_w = self.logvar_w_head(h).clamp(self.logvar_min, self.logvar_max)

        if self.use_residual_latent:
            if self.mu_r_head is None or self.logvar_r_head is None:
                raise RuntimeError("Residual heads expected but not initialized.")
            mu_r = self.mu_r_head(h)
            logvar_r = self.logvar_r_head(h).clamp(self.logvar_min, self.logvar_max)
        else:
            mu_r = None
            logvar_r = None
        return mu_w, logvar_w, mu_r, logvar_r

    def _decode(
        self,
        w: Tensor,
        r: Tensor | None = None,
        sample_idx: Tensor | None = None,
    ) -> tuple[Tensor | None, Tensor | None, Tensor, Tensor]:
        if self.decoder_type == "factorized":
            h_main = w @ self.Z
            h_total = h_main
            if r is not None:
                r_proj = r if self.P_r is None else (r @ self.P_r)
                h_total = h_main + r_proj
            x_hat = h_total @ self.E.T + self.b
            a_hat = self.Z @ self.E.T + self.b
            # a_hat stays batch-independent; only x_hat gets the per-patient offset.
            x_hat = apply_sample_offset(x_hat, self.B, sample_idx)
            return h_main, h_total, x_hat, a_hat

        x_hat_main = w @ self.A
        if r is not None and self.G is not None:
            x_hat = x_hat_main + (r @ self.G)
        else:
            x_hat = x_hat_main
        x_hat = apply_sample_offset(x_hat, self.B, sample_idx)
        return None, None, x_hat, self.A

    def forward(
        self,
        x: Tensor,
        tau_override: float | None = None,
        sample: bool = True,
        use_posterior_mean: bool = False,
        sample_idx: Tensor | None = None,
    ) -> dict[str, Tensor | None]:
        mu_w, logvar_w, mu_r, logvar_r = self._encode(x)

        if use_posterior_mean or not sample:
            u_sample = mu_w
        else:
            u_sample = reparameterize_gaussian(mu_w, logvar_w)
        w = self._simplex_weights(u_sample, tau_override=tau_override)
        w_mean = self._simplex_weights(mu_w, tau_override=tau_override)

        r_sample: Tensor | None = None
        if self.use_residual_latent and mu_r is not None and logvar_r is not None:
            if use_posterior_mean or not sample:
                r_sample = mu_r
            else:
                r_sample = reparameterize_gaussian(mu_r, logvar_r)

        h_main, h_total, x_hat, a_hat = self._decode(w, r=r_sample, sample_idx=sample_idx)

        kl_w = gaussian_kl_standard_normal(mu_w, logvar_w).mean()
        if self.use_residual_latent and mu_r is not None and logvar_r is not None:
            kl_r = gaussian_kl_standard_normal(mu_r, logvar_r).mean()
        else:
            kl_r = x_hat.new_tensor(0.0)

        entropy = -(w * (w.clamp_min(1e-8)).log()).sum(dim=1).mean()
        w_bar = w.mean(dim=0)
        k = max(w.shape[1], 1)
        balance = ((w_bar - (1.0 / float(k))) ** 2).mean()

        sep_matrix = self.archetype_separation_tensor()
        if sep_matrix.shape[0] < 2:
            sep = x_hat.new_tensor(0.0)
        else:
            norm = sep_matrix.norm(dim=1, keepdim=True).clamp_min(1e-8)
            sim = (sep_matrix / norm) @ (sep_matrix / norm).T
            mask = ~torch.eye(sep_matrix.shape[0], dtype=torch.bool, device=sim.device)
            sep = (sim[mask] ** 2).mean()

        return {
            "X_hat": x_hat,
            "W": w,
            "W_mean": w_mean,
            "U": u_sample,
            "U_mean": mu_w,
            "u_sample": u_sample,
            "mu_w": mu_w,
            "logvar_w": logvar_w,
            "kl_w": kl_w,
            "kl_r": kl_r,
            "r_sample": r_sample,
            "mu_r": mu_r,
            "logvar_r": logvar_r,
            "H_main": h_main,
            "H": h_total,
            "cell_latent_main": h_main,
            "cell_latent_total": h_total,
            "A_hat": a_hat,
            "z_archetypes": self.Z if self.decoder_type == "factorized" else None,
            "marker_embeddings": self.E if self.decoder_type == "factorized" else None,
            "archetype_profiles": self.A if self.decoder_type == "direct" else None,
            "entropy": entropy,
            "sep": sep,
            "balance": balance,
        }
