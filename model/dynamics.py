import math
from dataclasses import dataclass
from typing import Optional, List
from .utils import create_temporal_mask
import torch
import torch.nn as nn
from model.blocks import EfficientTransformerBlock


# ------------------------------------------------------------
# 0. Utilities
# ------------------------------------------------------------

def ramp_weight(tau: torch.Tensor) -> torch.Tensor:
    """
    Eq. (8): w(τ) = 0.9 τ + 0.1

    tau: (B, T) or (B, T, 1, 1)
    returns: same shape as tau
    """
    return 0.9 * tau + 0.1


# ------------------------------------------------------------
# 1. DiscreteEmbedder (your original)
# ------------------------------------------------------------

class DiscreteEmbedder(nn.Module):
    def __init__(self, n_states, n_dim):
        super().__init__()
        self.n_states = n_states

        # (n_states, n_dim) — each row = embedding for one discrete state
        self.embeddings = nn.Parameter(torch.zeros(n_states, n_dim))

        # good idea: initialize like nn.Embedding
        nn.init.normal_(self.embeddings, std=0.02)

    def forward(self, x):
        """
        x: LongTensor of shape (B,) or (B, T) containing indices in [0, n_states)
        returns: embeddings of shape (B, n_dim) or (B, T, n_dim)
        """
        x = x.long()
        return self.embeddings[x]  # fancy indexing works


# ------------------------------------------------------------
# 2. Config for the denoiser / dynamics model
# ------------------------------------------------------------

@dataclass
class DreamerV4DenoiserCfg:
    num_action_tokens: int          # S_a
    num_latent_tokens: int
    num_register_tokens: int
    max_context_length: int
    model_dim: int
    latent_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: Optional[int] = None
    dropout_prob: float = 0.0
    qk_norm: bool = True
    K_max: int = 32                # finest grid size for τ (must be power of 2)
    # Action structure
    n_discrete_actions: int = 0    # n_d
    n_continuous_actions: int = 0  # n_c
    # For discrete actions: number of bins per component, length = n_discrete_actions
    discrete_action_bins: Optional[List[int]] = None


# ------------------------------------------------------------
# 3. ActionTokenizer (n_d discrete, n_c continuous components)
# ------------------------------------------------------------

class ActionTokenizer(nn.Module):
    """
    Encode actions into S_a tokens per (B, T).

    Inputs:
      discrete_actions  : (B, T, n_d) long or None
      continuous_actions: (B, T, n_c) float or None   (values >= 0 are fine)

    Output:
      action_offsets    : (B, T, S_a, D_model)
        These are offsets to be added to a learned base action embedding.
    """

    def __init__(
        self,
        model_dim: int,
        num_action_tokens: int,         # S_a
        n_discrete: int,
        n_continuous: int,
        discrete_action_bins: Optional[List[int]] = None,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.S_a = num_action_tokens
        self.n_discrete = n_discrete
        self.n_continuous = n_continuous

        if n_discrete > 0:
            assert discrete_action_bins is not None
            assert len(discrete_action_bins) == n_discrete

        # --- Discrete components: one embedding table per component ---
        self.discrete_embedders = nn.ModuleList()
        for bins in (discrete_action_bins or []):
            emb = nn.Embedding(bins, model_dim)
            nn.init.normal_(emb.weight, std=0.02)
            self.discrete_embedders.append(emb)

        # --- Continuous components: one small linear per component ---
        self.continuous_mlps = nn.ModuleList()
        for _ in range(n_continuous):
            mlp = nn.Linear(1, model_dim)   # scalar -> D_model
            nn.init.xavier_uniform_(mlp.weight)
            nn.init.zeros_(mlp.bias)
            self.continuous_mlps.append(mlp)

        # --- Per-component base tokens: shape (n_components, 1, 1, S_a, D) ---
        n_components = n_discrete + n_continuous
        if n_components > 0:
            self.component_tokens = nn.Parameter(
                torch.zeros(n_components, 1, 1, num_action_tokens, model_dim)
            )
            nn.init.normal_(self.component_tokens, std=0.02)
        else:
            self.register_buffer(
                "component_tokens",
                torch.zeros(0, 1, 1, num_action_tokens, model_dim)
            )

    def forward(
        self,
        discrete_actions: Optional[torch.Tensor],    # (B, T, n_d) or None
        continuous_actions: Optional[torch.Tensor],  # (B, T, n_c) or None
    ) -> torch.Tensor:
        """
        Returns:
          action_offsets: (B, T, S_a, D_model)
        """
        B = None
        T = None

        # Infer B, T from whichever is present
        if discrete_actions is not None:
            assert discrete_actions.dim() == 3  # (B, T, n_d)
            B, T, n_d = discrete_actions.shape
            assert n_d == self.n_discrete

        if continuous_actions is not None:
            assert continuous_actions.dim() == 3  # (B, T, n_c)
            Bc, Tc, n_c = continuous_actions.shape
            if B is None:
                B, T = Bc, Tc
            else:
                assert B == Bc and T == Tc
            assert n_c == self.n_continuous

        if B is None:
            # No actions: return an empty tensor; caller will ignore or use base tokens only.
            return torch.zeros(
                0, 0, self.S_a, self.model_dim, device=self.component_tokens.device
            )

        device = self.component_tokens.device
        # (B, T, S_a, D)
        action_offsets = torch.zeros(B, T, self.S_a, self.model_dim, device=device)

        comp_idx = 0

        # --- Discrete components ---
        if self.n_discrete > 0 and discrete_actions is not None:
            for i in range(self.n_discrete):
                # disc_i: (B, T)
                disc_i = discrete_actions[..., i].to(device)
                # e_i: (B, T, D_model)
                e_i = self.discrete_embedders[i](disc_i)

                # Expand over S_a: (B, T, 1, D) -> (B, T, S_a, D)
                e_i_tokens = e_i.unsqueeze(-2).expand(-1, -1, self.S_a, -1)

                # Component base tokens: (1, 1, S_a, D) -> (B, T, S_a, D)
                base_i = self.component_tokens[comp_idx].expand(B, T, -1, -1)

                # Add to offsets
                action_offsets = action_offsets + (base_i + e_i_tokens)
                comp_idx += 1

        # --- Continuous components ---
        if self.n_continuous > 0 and continuous_actions is not None:
            for j in range(self.n_continuous):
                # cont_j: (B, T)
                cont_j = continuous_actions[..., j].to(device)
                # v_j: (B, T, D_model)
                v_j = self.continuous_mlps[j](cont_j.unsqueeze(-1))  # (B, T, 1) -> (B, T, D)

                # (B, T, 1, D) -> (B, T, S_a, D)
                v_j_tokens = v_j.unsqueeze(-2).expand(-1, -1, self.S_a, -1)

                base_j = self.component_tokens[comp_idx].expand(B, T, -1, -1)
                action_offsets = action_offsets + (base_j + v_j_tokens)
                comp_idx += 1

        return action_offsets


# ------------------------------------------------------------
# 4. Dynamics model / denoiser
# ------------------------------------------------------------


class DreamerV4Denoiser(nn.Module):
    """
    Dynamics model / denoiser with:
      - per-frame latent tokens
      - per-frame discrete τ index (0..K_max-1)
      - per-frame discrete step index (0..max_pow2), where 0 ↔ d_min, max_pow2 ↔ 1
      - action conditioning via ActionTokenizer
    """

    def __init__(self, cfg: DreamerV4DenoiserCfg):
        super().__init__()
        self.cfg = cfg

        max_pow2 = int(math.log2(cfg.K_max))

        # --- Discrete embeddings for diffusion τ and shortcut step index ---
        self.diffuion_embedder = DiscreteEmbedder(cfg.K_max, cfg.model_dim)  # τ index: 0..K_max-1
        self.shortcut_embedder = DiscreteEmbedder(
            max_pow2 + 1, cfg.model_dim
        )  # step index: 0..max_pow2

        # --- Register tokens: (1, 1, S_r, D) ---
        self.register_tokens = nn.Parameter(
            torch.zeros(1, 1, cfg.num_register_tokens, cfg.model_dim)
        )

        # --- Base action tokens: (1, 1, S_a, D) ---
        self.action_tokens = nn.Parameter(
            torch.zeros(1, 1, cfg.num_action_tokens, cfg.model_dim)
        )

        self.max_seq_len = max(
            cfg.num_action_tokens + cfg.num_latent_tokens + cfg.num_register_tokens + 2,
            cfg.max_context_length,
        ) + 32  # buffer

        # --- Transformer layers ---
        self.layers = nn.ModuleList([
            EfficientTransformerBlock(
                model_dim=cfg.model_dim,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                dropout_prob=cfg.dropout_prob,
                qk_norm=cfg.qk_norm,
                max_seq_len=self.max_seq_len,
            )
            for _ in range(cfg.n_layers)
        ])

        # --- Latent projections ---
        self.latent_projector = nn.Linear(cfg.latent_dim, cfg.model_dim, bias=False)
        self.output_projector = nn.Linear(cfg.model_dim, cfg.latent_dim, bias=False)

        # --- Combine shortcut + diffusion embeddings into a single control token ---
        self.diff_control_proj = nn.Linear(cfg.model_dim * 2, cfg.model_dim, bias=False)

        # --- Precomputed temporal mask for causal attention ---
        temporal_mask_full = create_temporal_mask(
            self.max_seq_len,
            device=torch.device("cpu"),
        )
        self.register_buffer("temporal_mask_full", temporal_mask_full)

        # --- Action tokenizer submodule ---
        # self.action_tokenizer = ActionTokenizer(
        #     model_dim=cfg.model_dim,
        #     num_action_tokens=cfg.num_action_tokens,
        #     n_discrete=cfg.n_discrete_actions,
        #     n_continuous=cfg.n_continuous_actions,
        #     discrete_action_bins=cfg.discrete_action_bins,
        # )
        self.action_proj = nn.Linear(cfg.n_continuous_actions, cfg.model_dim)
        # Initialize learnable tokens
        nn.init.normal_(self.register_tokens, std=0.02)
        nn.init.normal_(self.action_tokens, std=0.02)

    def forward(
        self,
        latent_tokens: torch.Tensor,          # (B, T, N_latent, D_latent)
        diffusion_step: torch.Tensor,         # (B, T) long, τ index on finest grid
        shortcut_length: torch.Tensor,        # (B, T) long, step index (0..max_pow2; 0 ↔ d_min)
        discrete_actions: Optional[torch.Tensor] = None,    # (B, T, n_d)
        continuous_actions: Optional[torch.Tensor] = None,  # (B, T, n_c)
        causal: bool = True,
    ) -> torch.Tensor:
        B, T, N_lat, D_latent = latent_tokens.shape

        # --- Temporal mask for causal dynamics ---
        if causal:
            temporal_mask = self.temporal_mask_full[:T, :T]  # (T, T)
        else:
            temporal_mask = None

        # --- Encode diffusion τ and shortcut d into single control token ---
        # diff_step_token: (B, T, 1, D_model)
        diff_step_token = self.diffuion_embedder(diffusion_step).unsqueeze(-2)
        # shortcut_token : (B, T, 1, D_model)
        shortcut_token = self.shortcut_embedder(shortcut_length).unsqueeze(-2)

        # concat along channels: (B, T, 1, 2*D_model) -> (B, T, 1, D_model)
        diff_control_token = torch.cat([shortcut_token, diff_step_token], dim=-1)
        diff_control_token = self.diff_control_proj(diff_control_token)  # (B, T, 1, D_model)

        # --- Register tokens replicated per time step ---
        # reg_tokens: (1, 1, S_r, D) -> (B, T, S_r, D)
        reg_tokens = self.register_tokens.expand(B, T, -1, -1)

        # --- Action tokens: base + encoded components ---
        # base_action_tokens: (1, 1, S_a, D) -> (B, T, S_a, D)
        base_action_tokens = self.action_tokens.expand(B, T, -1, -1)
        # action_offsets: (B, T, S_a, D)
        # action_offsets = self.action_tokenizer(discrete_actions, continuous_actions)
        # if action_offsets.numel() == 0:
        #     act_tokens = base_action_tokens
        # else:
        #     act_tokens = base_action_tokens + action_offsets  # (B, T, S_a, D)
        if continuous_actions is not None:
            action_offsets = self.action_proj(continuous_actions).unsqueeze(-2) #ToDo: Make this more general 
            act_tokens = base_action_tokens + action_offsets
        else:
            act_tokens = base_action_tokens

        # --- Project latents to model dim ---
        # latent_proj: (B, T, N_lat, D_model)
        latent_proj = self.latent_projector(latent_tokens)

        # --- Concatenate tokens:
        #    [latent_tokens : register_tokens : diff_control_token : action_tokens]
        # latent_proj      : (B, T, N_lat,   D)
        # reg_tokens       : (B, T, S_r,     D)
        # diff_control_tok : (B, T, 1,       D)
        # act_tokens       : (B, T, S_a,     D)
        x = torch.cat(
            [latent_proj, reg_tokens, diff_control_token, act_tokens],
            dim=-2,  # token dimension
        )  # x: (B, T, N_lat + S_r + 1 + S_a, D_model)

        # --- Transformer dynamics ---
        for layer in self.layers:
            x = layer(x, temporal_mask=temporal_mask)

        # --- Project back to latent dim, return only latent slice ---
        # x: (B, T, N_lat + S_r + 1 + S_a, D_model) -> (B, T, N_lat + ..., D_latent)
        x = self.output_projector(x)

        # denoised latents: (B, T, N_lat, D_latent)
        return x[:, :, :self.cfg.num_latent_tokens, :]


# ------------------------------------------------------------
# 5. Forward diffusion with per-frame (τ_t, d_t) using YOUR convention
# ------------------------------------------------------------

class ForwardDiffusionWithShortcut(nn.Module):
    """
    Dyadic shortcut schedule with per-frame (τ_t, d_t).

    K_max must be a power of 2, e.g. 32 or 64.
    Finest step size: d_min = 1 / K_max.

    Index convention used:
      max_pow2 = log2(K_max)
      step_index_raw ∈ {0..max_pow2}  (defines d = 1 / 2^step_index_raw)
      step_index      = max_pow2 - step_index_raw
        -> step_index = 0       ↔ d = d_min
        -> step_index = max_pow2 ↔ d = 1
    """

    def __init__(self, K_max=32):
        super().__init__()
        assert (K_max & (K_max - 1)) == 0, "K_max must be a power of 2"
        self.K_max = int(K_max)
        self.max_pow2 = int(math.log2(self.K_max))      # e.g. K_max=32 -> max_pow2=5
        self.d_min = 1.0 / float(self.K_max)            # finest step

    def sample_step_noise(self, batch_size, seq_len, device):
        """
        Returns per-frame diffusion parameters:

          tau             : (B, T) float   τ_t ∈ {0, d_t, 2 d_t, ..., 1 - d_t}
          step            : (B, T) float   d_t ∈ {1, 1/2, ..., 1/K_max}
          tau_index       : (B, T) long    τ index on finest grid: τ_t = tau_index * d_min
          step_index      : (B, T) long    our index in [0..max_pow2]; 0 ↔ d_min, max_pow2 ↔ 1
          half_step_index : (B, T) long    index for d_t/2 in same convention (step_index - 1)
          tau_plus_half_index: (B, T) long index on finest grid for τ_t + d_t/2
        """
        B, T = batch_size, seq_len

        # step_index_raw ∈ {0, ..., max_pow2}, defines d_t = 1 / 2^{step_index_raw}
        step_index_raw = torch.randint(
            low=0,
            high=self.max_pow2 + 1,
            size=(B, T),
            device=device,
            dtype=torch.long,
        )  # (B, T)

        # d_t = 1 / 2^{step_index_raw}
        step = 1.0 / (2.0 ** step_index_raw.float())     # (B, T) float

        # Our convention: step_index = max_pow2 - step_index_raw
        # step_index = 0  ↔ d = d_min,  step_index = max_pow2 ↔ d = 1
        step_index = self.max_pow2 - step_index_raw      # (B, T) long in [0..max_pow2]

        # Number of τ levels for this d_t: 1 / d_t = 2^{step_index_raw}
        num_levels = (2 ** step_index_raw).float()       # (B, T) float

        # m_t ∈ {0, ..., 2^{step_index_raw} - 1}
        m = torch.floor(
            torch.rand(B, T, device=device) * 0.9999 * num_levels
        ).long()                                         # (B, T) long

        tau = m.float() * step                           # (B, T) float, τ_t ∈ {0, d_t, ..., 1-d_t}

        # τ index on the finest grid of size K_max:
        # τ_t = m * d_t; d_t / d_min = 2^{step_index}; so τ_t / d_min = m * 2^{step_index}
        tau_index = m * (2 ** step_index)                # (B, T) long in [0 .. K_max-1]

        # stride on finest grid for step d_t: d_t / d_min = 2^{step_index}
        stride_full = (2 ** step_index)                  # (B, T) long
        delta_tau_index = stride_full // 2               # corresponds to d_t/2

        tau_plus_half_index = tau_index + delta_tau_index     # (B, T) long
        tau_plus_half_index = torch.clamp(
            tau_plus_half_index, min=0, max=self.K_max - 1
        )

        # half-step index in our convention: step_index - 1
        half_step_index = torch.clamp(step_index - 1, min=0)  # (B, T) long

        return dict(
            tau=tau,                                      # (B, T) float
            step=step,                                    # (B, T) float
            tau_index=tau_index,                          # (B, T) long
            step_index=step_index,                        # (B, T) long
            half_step_index=half_step_index,              # (B, T) long
            tau_plus_half_index=tau_plus_half_index,      # (B, T) long
        )

    def forward(self, z_clean: torch.Tensor):
        """
        z_clean: (B, T, N_lat, D_latent)  # clean latents (x = z_1)
        """
        B, T, N_lat, D_lat = z_clean.shape
        device = z_clean.device

        # Sample z_0 ~ N(0, I) in latent space
        z0 = torch.randn_like(z_clean)                      # (B, T, N_lat, D_lat)

        diff = self.sample_step_noise(B, T, device)

        tau = diff["tau"].unsqueeze(-1).unsqueeze(-1)       # (B, T, 1, 1)
        step = diff["step"].unsqueeze(-1).unsqueeze(-1)     # (B, T, 1, 1)

        # z_τ = (1 - τ) z_0 + τ z_1
        z_tau = (1.0 - tau) * z0 + tau * z_clean            # (B, T, N_lat, D_lat)

        return {
            "x": z_clean,                          # (B, T, N_lat, D_lat)  clean z_1
            "x_tau": z_tau,                        # (B, T, N_lat, D_lat)  noisy z_τ
            "tau": diff["tau"],                    # (B, T) float
            "step": diff["step"],                  # (B, T) float
            "tau_index": diff["tau_index"],        # (B, T) long
            "step_index": diff["step_index"],      # (B, T) long, 0 ↔ d_min
            "half_step_index": diff["half_step_index"],            # (B, T) long
            "tau_plus_half_index": diff["tau_plus_half_index"],    # (B, T) long
            "d_min": self.d_min,                   # scalar float
            "K_max": self.K_max,
        }


# ------------------------------------------------------------
# 6. apply_denoiser helper
# ------------------------------------------------------------

def apply_denoiser(
    denoiser: DreamerV4Denoiser,
    x_tau: torch.Tensor,              # (B, T, N_lat, D_latent)
    tau_index: torch.Tensor,          # (B, T) long
    step_index: torch.Tensor,         # (B, T) long
    discrete_actions: Optional[torch.Tensor] = None,    # (B, T, n_d)
    continuous_actions: Optional[torch.Tensor] = None,  # (B, T, n_c)
    no_grad: bool = False,
) -> torch.Tensor:
    """
    Wrapper around DreamerV4Denoiser, returning z_hat ∈ ℝ^{B,T,N_lat,D_latent}.
    """
    if no_grad:
        with torch.no_grad():
            return denoiser(
                latent_tokens=x_tau,
                diffusion_step=tau_index,
                shortcut_length=step_index,
                discrete_actions=discrete_actions,
                continuous_actions=continuous_actions,
                causal=True,
            )
    else:
        return denoiser(
            latent_tokens=x_tau,
            diffusion_step=tau_index,
            shortcut_length=step_index,
            discrete_actions=discrete_actions,
            continuous_actions=continuous_actions,
            causal=True,
        )


# ------------------------------------------------------------
# 7. Shortcut / bootstrap loss (Eq. 7 + Eq. 8) using YOUR convention
# ------------------------------------------------------------

def compute_bootstrap_diffusion_loss(
    info: dict,
    denoiser: DreamerV4Denoiser,
    discrete_actions: Optional[torch.Tensor] = None,     # (B, T, n_d)
    continuous_actions: Optional[torch.Tensor] = None,   # (B, T, n_c)
):
    """
    info: dict from ForwardDiffusionWithShortcut.forward, with keys:
      x                   : (B, T, N_lat, D_latent)
      x_tau               : (B, T, N_lat, D_latent)
      tau                 : (B, T) float
      step                : (B, T) float
      tau_index           : (B, T) long
      step_index          : (B, T) long  (0 ↔ d_min, >0 ↔ larger steps)
      half_step_index     : (B, T) long
      tau_plus_half_index : (B, T) long
      d_min               : scalar float

    Returns:
      flow_loss      : scalar (flow branch, d = d_min)
      bootstrap_loss : scalar (bootstrap branch, d > d_min)
    """
    x = info["x"]                         # (B, T, N_lat, D_lat)
    x_tau = info["x_tau"]                 # (B, T, N_lat, D_lat)
    tau = info["tau"]                     # (B, T) float
    step = info["step"]                   # (B, T) float
    tau_index = info["tau_index"]         # (B, T) long
    step_index = info["step_index"]       # (B, T) long (0..max_pow2)
    half_step_index = info["half_step_index"]              # (B, T) long
    tau_plus_half_index = info["tau_plus_half_index"]      # (B, T) long
    d_min = info["d_min"]                 # scalar float

    B, T, N_lat, D_lat = x.shape

    tau_b = tau.view(B, T, 1, 1)          # (B, T, 1, 1)
    step_b = step.view(B, T, 1, 1)        # (B, T, 1, 1)

    # -------------------------------------------------
    # 1) Big-step prediction:  \hat z_1 = f_θ(z_τ, τ, d)
    # -------------------------------------------------
    z_hat = apply_denoiser(
        denoiser,
        x_tau,                 # (B, T, N_lat, D_lat)
        tau_index,             # (B, T)
        step_index,            # (B, T)
        discrete_actions=discrete_actions,
        continuous_actions=continuous_actions,
        no_grad=False,
    )                           # -> (B, T, N_lat, D_lat)

    # =================================================
    # 2) Flow loss branch (d = d_min): || ẑ_1 - z_1 ||^2
    # =================================================
    flow_residual = z_hat - x                      # (B, T, N_lat, D_lat)
    # average over tokens & dims: (B, T)
    flow_sq = flow_residual.pow(2).mean(dim=(-1, -2))

    # d = d_min ↔ step_index == 0 in this convention
    mask_small = (step_index == 0).float()         # (B, T), 1 where d = d_min

    # ramp weight w(τ) from Eq. (8), τ ∈ [0, 1]
    w_tau = ramp_weight(tau)                       # (B, T)

    flow_loss_per = w_tau * flow_sq * mask_small  # (B, T)
    denom_flow = mask_small.sum().clamp_min(1.0)  # avoid div by 0
    flow_loss = flow_loss_per.sum() / denom_flow  # scalar

    # =================================================
    # 3) Bootstrap branch (d > d_min):
    #    v_target = (b1 + b2)/2, with
    #    b1 = [f(z_τ, τ, d/2) - z_τ] / (1 - τ)
    #    b2 = [f(z', τ + d/2, d/2) - z'] / (1 - (τ + d/2))
    #    Loss: (1 - τ)^2 || v_hat - sg(v_target) ||^2 * w(τ)
    # =================================================
    with torch.no_grad():
        # ------- First half-step -------
        f1 = apply_denoiser(
            denoiser,
            x_tau,                  # (B, T, N_lat, D_lat)
            tau_index,              # (B, T)  use same τ
            half_step_index,        # (B, T)  d/2 index
            discrete_actions=discrete_actions,
            continuous_actions=continuous_actions,
            no_grad=True,
        )                           # (B, T, N_lat, D_lat)

        # b1: (B, T, N_lat, D_lat)
        b1 = (f1 - x_tau) / (1.0 - tau_b)

        # z' = z_τ + b1 * d/2
        z_prime = x_tau + b1 * (step_b / 2.0)      # (B, T, N_lat, D_lat)

        # ------- Second half-step -------
        f2 = apply_denoiser(
            denoiser,
            z_prime,                # (B, T, N_lat, D_lat)
            tau_plus_half_index,    # (B, T)  τ + d/2
            half_step_index,        # (B, T)  d/2
            discrete_actions=discrete_actions,
            continuous_actions=continuous_actions,
            no_grad=True,
        )                           # (B, T, N_lat, D_lat)

        denom2 = 1.0 - (tau_b + step_b / 2.0)      # (B, T, 1, 1)
        b2 = (f2 - z_prime) / denom2               # (B, T, N_lat, D_lat)

        # v_target: average of the two half-step velocities
        v_target = 0.5 * (b1 + b2)                 # (B, T, N_lat, D_lat)  (sg via no_grad)

    # predicted velocity from big step:
    v_hat = (z_hat - x_tau) / (1.0 - tau_b)        # (B, T, N_lat, D_lat)

    boot_err = v_hat - v_target                    # (B, T, N_lat, D_lat)

    # x-space scaling (1 - τ)^2 and ramp weight w(τ)
    boot_sq = ((1.0 - tau_b) ** 2 * boot_err.pow(2)).mean(dim=(-1, -2))  # (B, T)

    mask_large = (step_index > 0).float()          # (B, T)  d > d_min
    boot_loss_per = w_tau * boot_sq * mask_large   # (B, T)

    denom_boot = mask_large.sum().clamp_min(1.0)
    bootstrap_loss = boot_loss_per.sum() / denom_boot

    return flow_loss, bootstrap_loss