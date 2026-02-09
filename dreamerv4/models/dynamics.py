import math
from dataclasses import dataclass
from typing import Optional, List
from .blocks import create_temporal_mask
import torch
import torch.nn as nn
from .blocks import EfficientTransformerBlock
from omegaconf import DictConfig, OmegaConf

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

@dataclass
class DreamerV4DenoiserCfg:
    num_action_tokens: int          # S_a
    num_latent_tokens: int
    num_register_tokens: int
    max_sequence_length: int
    context_length: int
    model_dim: int
    latent_dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: Optional[int] = None
    dropout_prob: float = 0.0
    qk_norm: bool = True
    num_noise_levels: int = 32                # finest grid size for τ (must be power of 2)
    n_actions: int = 0  # number of action components
    dual_stream: bool = False

class DreamerV4Denoiser(nn.Module):
    """
    Dynamics model / denoiser with:
      - per-frame latent tokens
      - per-frame discrete τ index (0..num_noise_levels-1)
      - per-frame discrete step index (0..max_pow2), where 0 ↔ d_min, max_pow2 ↔ 1
      - action conditioning via ActionTokenizer
    """

    def __init__(self, cfg: DreamerV4DenoiserCfg, max_num_forward_steps=None):
        super().__init__()
        self.cfg = cfg
        # --- Discrete embeddings for diffusion τ and shortcut step index ---
        self.diffusion_embedder = DiscreteEmbedder(cfg.num_noise_levels, cfg.model_dim)  # τ index: 0..num_noise_levels-1
        self.shortcut_embedder = DiscreteEmbedder(int(math.log2(cfg.num_noise_levels)) + 1, cfg.model_dim)  # step index: 0..max_pow2

        # --- Register tokens: (1, 1, S_r, D) ---
        self.register_tokens = nn.Parameter(
            torch.zeros(1, 1, cfg.num_register_tokens, cfg.model_dim)
        )

        # --- Base action tokens: (1, 1, S_a, D) ---
        self.action_tokens = nn.Parameter(
            torch.zeros(1, 1, cfg.num_action_tokens, cfg.model_dim)
        )

        self.num_modality_tokens = cfg.num_action_tokens + \
                                   cfg.num_latent_tokens + \
                                   cfg.num_register_tokens + \
                                    1 # noise level + shortcut tokens        

        # --- Transformer layers ---
        self.layers = nn.ModuleList([
            EfficientTransformerBlock(
                model_dim=cfg.model_dim,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                dropout_prob=cfg.dropout_prob,
                qk_norm=cfg.qk_norm,
                modality_dim_max_seq_len=self.num_modality_tokens,
                temporal_dim_max_seq_len= (max_num_forward_steps if max_num_forward_steps is not None else cfg.max_sequence_length),   
                context_length=cfg.context_length,             
            )
            for _ in range(cfg.n_layers)
        ])

        # --- Latent projections ---
        self.latent_projector = nn.Linear(cfg.latent_dim, cfg.model_dim, bias=False)
        self.output_projector = nn.Linear(cfg.model_dim, cfg.latent_dim, bias=False)

        # --- Combine shortcut + diffusion embeddings into a single control token ---
        self.diff_control_proj = nn.Linear(cfg.model_dim * 2, cfg.model_dim, bias=False)

        self.action_proj = nn.Linear(cfg.n_actions, cfg.model_dim)
        # Initialize learnable tokens
        nn.init.normal_(self.register_tokens, std=0.02)
        nn.init.normal_(self.action_tokens, std=0.02)

    def forward(
        self,
        action: Optional[torch.Tensor],  # (B, T, n_c)
        noisy_z: torch.Tensor,          # (B, T, N_latent, D_latent)
        sigma_idx: torch.Tensor,         # (B, T) long, τ index on finest grid
        step_idx: torch.Tensor,        # (B, T) long, step index (0..max_pow2; 0 ↔ d_min)
    ) -> torch.Tensor:
        B, T, N_lat, D_latent = noisy_z.shape

        # --- Encode diffusion τ and shortcut d into single control token ---
        # diff_step_token: (B, T, 1, D_model)
        diff_step_token = self.diffusion_embedder(sigma_idx).unsqueeze(-2)
        # shortcut_token : (B, T, 1, D_model)
        shortcut_token = self.shortcut_embedder(step_idx).unsqueeze(-2)

        # concat along channels: (B, T, 1, 2*D_model) -> (B, T, 1, D_model)
        diff_control_token = torch.cat([shortcut_token, diff_step_token], dim=-1)
        diff_control_token = self.diff_control_proj(diff_control_token)  # (B, T, 1, D_model)

        # --- Register tokens replicated per time step ---
        # reg_tokens: (1, 1, S_r, D) -> (B, T, S_r, D)
        reg_tokens = self.register_tokens.expand(B, T, -1, -1)

        # --- Action tokens: base + encoded components ---
        # base_action_tokens: (1, 1, S_a, D) -> (B, T, S_a, D)
        base_action_tokens = self.action_tokens.expand(B, T, -1, -1)
        if action is not None:
            action_offsets = self.action_proj(action).unsqueeze(-2) #ToDo: Make this more general 
            act_tokens = base_action_tokens[:, :action_offsets.shape[1]] + action_offsets
        else:
            act_tokens = base_action_tokens

        # --- Project latents to model dim ---
        # latent_proj: (B, T, N_lat, D_model)
        latent_proj = self.latent_projector(noisy_z)

        # --- Concatenate tokens:
        #[latent_tokens : register_tokens : diff_control_token : action_tokens]
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
            x = layer(x)

        # --- Project back to latent dim, return only latent slice ---
        # x: (B, T, N_lat + S_r + 1 + S_a, D_model) -> (B, T, N_lat + ..., D_latent)
        x = self.output_projector(x)

        # denoised latents: (B, T, N_lat, D_latent)
        return x[:, :, :self.cfg.num_latent_tokens, :]
    
    def forward_step(self, 
                     action: torch.Tensor, 
                     noisy_z: torch.Tensor, 
                     sigma_idx: torch.Tensor, 
                     step_idx: torch.Tensor,
                     start_step_idx: int,
                     update_cache: bool = True):
        
        B, T, N_lat, D_latent = noisy_z.shape

        # --- Encode diffusion τ and shortcut d into single control token ---
        # diff_step_token: (B, T, 1, D_model)
        diff_step_token = self.diffusion_embedder(sigma_idx).unsqueeze(-2)
        # shortcut_token : (B, T, 1, D_model)
        shortcut_token = self.shortcut_embedder(step_idx).unsqueeze(-2)

        # concat along channels: (B, T, 1, 2*D_model) -> (B, T, 1, D_model)
        diff_control_token = torch.cat([shortcut_token, diff_step_token], dim=-1)
        diff_control_token = self.diff_control_proj(diff_control_token)  # (B, T, 1, D_model)

        # --- Register tokens replicated per time step ---
        # reg_tokens: (1, 1, S_r, D) -> (B, T, S_r, D)
        reg_tokens = self.register_tokens.expand(B, T, -1, -1)

        # --- Action tokens: base + encoded components ---
        # base_action_tokens: (1, 1, S_a, D) -> (B, T, S_a, D)
        base_action_tokens = self.action_tokens.expand(B, T, -1, -1)
        if action is not None:
            action_offsets = self.action_proj(action).unsqueeze(-2) #ToDo: Make this more general 
            act_tokens = base_action_tokens[:, :action_offsets.shape[1]] + action_offsets
        else:
            act_tokens = base_action_tokens

        # --- Project latents to model dim ---
        # latent_proj: (B, T, N_lat, D_model)
        latent_proj = self.latent_projector(noisy_z)

        # --- Concatenate tokens:
        #[latent_tokens : register_tokens : diff_control_token : action_tokens]
        # latent_proj      : (B, T, N_lat,   D)
        # reg_tokens       : (B, T, S_r,     D)
        # diff_control_tok : (B, T, 1,       D)
        # act_tokens       : (B, T, S_a,     D)
        x = torch.cat(
            [latent_proj, reg_tokens, diff_control_token, act_tokens],
            dim=-2,  # token dimension
        )  # x: (B, T, N_lat + S_r + 1 + S_a, D_model)
        # 3. Apply Layers
        for i, layer in enumerate(self.layers):
            x = layer.forward_step(x, start_step_idx=start_step_idx, spatial_mask=None, update_cache=update_cache)

        # 4. Output
        x = self.output_projector(x)
        # denoised latents: (B, T, N_lat, D_latent)
        return x[:, :, :self.cfg.num_latent_tokens, :]
    
    def init_cache(self, batch_size: int, device: torch.device, context_length: int, dtype: torch.dtype):
        """Initializes KV caches for all temporal layers."""
        for layer in self.layers:
            layer.init_cache(batch_size, device, context_length, dtype)
    
class DenoiserWrapper(nn.Module):

    def __init__(self, cfg: DictConfig, max_num_forward_steps=None):
        super().__init__()
        self.cfg = cfg
        denoiser_cfg = DreamerV4DenoiserCfg(**OmegaConf.to_object(cfg.denoiser)) 
        self.model = DreamerV4Denoiser(denoiser_cfg, max_num_forward_steps=max_num_forward_steps)

    def forward(
        self,
        action: Optional[torch.Tensor],  # (B, T, n_c)
        noisy_z: torch.Tensor,          # (B, T, N_latent, D_latent)
        sigma_idx: torch.Tensor,         # (B, T) long, τ index on finest grid
        step_idx: torch.Tensor,        # (B, T) long, step index (0..max_pow2; 0 ↔ d_min)
    ) -> torch.Tensor:
        return self.model(action, noisy_z, sigma_idx, step_idx)
    
    def forward_step(self, 
                     action: torch.Tensor, 
                     noisy_z: torch.Tensor, 
                     sigma_idx: torch.Tensor, 
                     step_idx: torch.Tensor,
                     start_step_idx: int,
                     update_cache: bool = True):
        return self.model.forward_step(action, 
                                noisy_z, 
                                sigma_idx, 
                                step_idx, 
                                start_step_idx, 
                                update_cache)
    
    def init_cache(self, 
                   batch_size: int, 
                   device: torch.device, 
                   context_length: int, 
                   dtype: torch.dtype):
        self.model.init_cache(batch_size, device, context_length, dtype)
