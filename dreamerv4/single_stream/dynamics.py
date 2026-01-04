import math
from dataclasses import dataclass
from typing import Optional, List
from .utils import create_temporal_mask
import torch
import torch.nn as nn
from .blocks import EfficientTransformerBlock, DiscreteEmbedder

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
    n_actions: int = 0  # number of action components

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

        self.action_proj = nn.Linear(cfg.n_actions, cfg.model_dim)
        # Initialize learnable tokens
        nn.init.normal_(self.register_tokens, std=0.02)
        nn.init.normal_(self.action_tokens, std=0.02)

    def forward(
        self,
        latent_tokens: torch.Tensor,          # (B, T, N_latent, D_latent)
        diffusion_step: torch.Tensor,         # (B, T) long, τ index on finest grid
        shortcut_length: torch.Tensor,        # (B, T) long, step index (0..max_pow2; 0 ↔ d_min)
        actions: Optional[torch.Tensor] = None,  # (B, T, n_c)
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
        if actions is not None:
            action_offsets = self.action_proj(actions).unsqueeze(-2) #ToDo: Make this more general 
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