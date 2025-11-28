import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .blocks import EfficientTransformerBlock
from .utils import create_temporal_mask, create_encoder_spatial_mask, create_decoder_spatial_mask
from dataclasses import dataclass
import math

@dataclass
class CausalTokenizerConfig:
    num_modality_tokens: int
    num_latent_tokens: int
    max_context_length: int
    model_dim: int
    latent_dim: int
    enc_num_layers: int 
    dec_num_layers: int 
    n_heads: int
    n_kv_heads: Optional[int] = None
    dropout_prob: float = 0.0
    qk_norm: bool = True
    patch_size: int = 14
   
class CausalTokenizerEncoder(nn.Module):
    """
    Encoder:
      Input:  x ∈ ℝ^{B, T, S_mod, D_model}
      Output: latents ∈ ℝ^{B, T, N_lat, D_latent},
              patch_embeddings ∈ ℝ^{B, T, S_mod, D_latent}
    """

    def __init__(self, cfg: CausalTokenizerConfig):
        super().__init__()
        self.cfg = cfg

        model_dim = cfg.model_dim
        num_layers = cfg.enc_num_layers

        # S_mod + N_latent
        self.num_spatial_tokens = cfg.num_modality_tokens + cfg.num_latent_tokens

        # Temporal max length used for RoPE / masks
        self.max_seq_len = max(cfg.max_context_length, self.num_spatial_tokens)

        # Transformer blocks
        self.layers = nn.ModuleList([
            EfficientTransformerBlock(
                model_dim=model_dim,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                dropout_prob=cfg.dropout_prob,
                qk_norm=cfg.qk_norm,
                max_seq_len=self.max_seq_len,
            )
            for _ in range(num_layers)
        ])

        # Learned latent tokens appended at the end of each modality sequence
        # Shape: (1, 1, N_latent, D_model)
        self.learned_latent_tokens = nn.Parameter(
            torch.randn(1, 1, cfg.num_latent_tokens, model_dim) / math.sqrt(cfg.model_dim)
        )

        # Spatial mask is static → register as buffer so .to(device) moves it once
        spatial_mask = create_encoder_spatial_mask(
            cfg.num_modality_tokens,
            cfg.num_latent_tokens,
        )
        self.register_buffer("spatial_mask", spatial_mask)

        # Temporal causal mask for max_seq_len; we slice [:T, :T] in forward
        temporal_mask_full = create_temporal_mask(
            self.max_seq_len,
            device=torch.device("cpu"),
        )
        self.register_buffer("temporal_mask_full", temporal_mask_full)

        # Project to latent_dim (typically < model_dim)
        self.output_proj = nn.Linear(cfg.model_dim, cfg.latent_dim)
        self.output_nonlinearity = nn.Tanh()

    def forward(self, x: torch.Tensor, causal: bool = True):
        """
        x: (B, T, S_mod, D_model)
        returns:
          latents:          (B, T, N_latent, D_latent)
          patch_embeddings: (B, T, S_mod, D_latent)
        """
        assert x.dim() == 4, "Input must be (B, T, S_mod, D_model)"
        B, T, S_mod, D = x.shape
        assert S_mod == self.cfg.num_modality_tokens, \
            f"Expected S_mod={self.cfg.num_modality_tokens}, got {S_mod}"
        assert D == self.cfg.model_dim, \
            f"Expected D_model={self.cfg.model_dim}, got {D}"
        assert T <= self.max_seq_len, \
            f"T={T} exceeds max_seq_len={self.max_seq_len}"

        # Append learned latent tokens at the end of each modality sequence
        # learned_latent_tokens: (1, 1, N_latent, D)
        # -> (B, T, N_latent, D) via expand (view-only)
        latent_tokens = self.learned_latent_tokens.expand(B, T, -1, -1)
        x = torch.cat([x, latent_tokens], dim=2)  # (B, T, S_mod + N_latent, D)

        # Temporal mask (slice from precomputed)
        if causal:
            temporal_mask = self.temporal_mask_full[:T, :T]
        else:
            temporal_mask = None

        spatial_mask = self.spatial_mask  # already on correct device via .to(...)

        # Pass through transformer stack
        for layer in self.layers:
            x = layer(x, temporal_mask=temporal_mask, spatial_mask=spatial_mask)

        # Project to latent_dim and squash
        x = self.output_nonlinearity(self.output_proj(x))

        # Split back into [patch_tokens, latent_tokens]
        S_mod = self.cfg.num_modality_tokens
        patch_embeddings = x[:, :, :S_mod, :]                          # (B, T, S_mod, D_latent)
        latents = x[:, :, S_mod:, :]                                   # (B, T, N_latent, D_latent)

        return latents, patch_embeddings
        
class CausalTokenizerDecoder(nn.Module):
    """
    Decoder:
      Input:  latents ∈ ℝ^{B, T, N_latent, D_latent}
      Output: modality_tokens ∈ ℝ^{B, T, S_mod, D_model}
              (these will typically go into an image head)
    """

    def __init__(self, cfg: CausalTokenizerConfig):
        super().__init__()
        self.cfg = cfg

        model_dim = cfg.model_dim
        num_layers = cfg.dec_num_layers

        self.num_spatial_tokens = cfg.num_modality_tokens + cfg.num_latent_tokens
        self.max_seq_len = max(cfg.max_context_length, self.num_spatial_tokens)

        self.layers = nn.ModuleList([
            EfficientTransformerBlock(
                model_dim=model_dim,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                dropout_prob=cfg.dropout_prob,
                qk_norm=cfg.qk_norm,
                max_seq_len=self.max_seq_len,
            )
            for _ in range(num_layers)
        ])

        # Learned "modality" tokens that are prepended and will be decoded
        # Shape: (1, 1, S_mod, D_model)
        self.learned_modality_tokens = nn.Parameter(
            torch.randn(1, 1, cfg.num_modality_tokens, model_dim)/math.sqrt(cfg.model_dim)
        )

        # Spatial mask (static)
        spatial_mask = create_decoder_spatial_mask(
            cfg.num_modality_tokens,
            cfg.num_latent_tokens,
        )
        self.register_buffer("spatial_mask", spatial_mask)

        # Temporal mask (static at max_seq_len)
        temporal_mask_full = create_temporal_mask(
            self.max_seq_len,
            device=torch.device("cpu"),
        )
        self.register_buffer("temporal_mask_full", temporal_mask_full)

        # Project latents from latent_dim → model_dim
        self.input_proj = nn.Linear(cfg.latent_dim, cfg.model_dim)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        """
        x: (B, T, N_latent, D_latent)
        returns:
          modality_tokens: (B, T, S_mod, D_model)
        """
        assert x.dim() == 4, "Input tensor must be 4D (B, T, N_latent, D_latent)"
        B, T, S_lat, D_latent = x.shape

        assert D_latent == self.cfg.latent_dim, \
            f"Expected last dim={self.cfg.latent_dim}, got {D_latent}"
        assert S_lat == self.cfg.num_latent_tokens, \
            f"Expected spatial dim={self.cfg.num_latent_tokens}, got {S_lat}"
        assert T <= self.max_seq_len, \
            f"Temporal dimension {T} exceeds max_seq_len {self.max_seq_len}"

        # Project latents to model_dim
        x = self.input_proj(x)  # (B, T, N_latent, D_model)

        # Prepend learned modality tokens
        # learned_modality_tokens: (1, 1, S_mod, D_model) -> (B, T, S_mod, D_model)
        modality_tokens = self.learned_modality_tokens.expand(B, T, -1, -1).contiguous()
        x = torch.cat([modality_tokens, x], dim=2)  # (B, T, S_mod + N_latent, D_model)

        # Temporal mask
        if causal:
            temporal_mask = self.temporal_mask_full[:T, :T]
        else:
            temporal_mask = None

        spatial_mask = self.spatial_mask

        # Transformer stack
        for layer in self.layers:
            x = layer(x, temporal_mask=temporal_mask, spatial_mask=spatial_mask)

        # Return only the decoded modality tokens (drop the latent positions)
        S_mod = self.cfg.num_modality_tokens
        return x[:, :, :S_mod, :]   # (B, T, S_mod, D_model)

class ImagePatchifier(nn.Module):
    """
    Turn images into patch tokens.

    Input:  x ∈ ℝ^{B, T, C, H, W}
    Output: tokens ∈ ℝ^{B, T, N_patches, D_model}
    """

    def __init__(self, patch_size: int, model_dim: int, input_channels: int = 3):
        super().__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.model_dim = model_dim

        # Conv2d implements patchify: kernel_size=stride=patch_size
        self.cnn = nn.Conv2d(
            in_channels=input_channels,
            out_channels=model_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5, "Input should be of shape (B, T, C, H, W)"
        B, T, C, H, W = x.shape
        assert C == self.input_channels, \
            f"Image channels {C} != expected {self.input_channels}"

        # Merge B,T → apply conv → unmerge
        x = x.view(-1, C, H, W)                      # (B*T, C, H, W)
        x = self.cnn(x)                              # (B*T, D_model, h, w)
        x = x.flatten(-2, -1).transpose(-2, -1)      # (B*T, N_patches, D_model)
        x = x.view(B, T, -1, self.model_dim)         # (B, T, N_patches, D_model)
        return x
  
class TokensToImageHead(nn.Module):
    """
    Convert modality tokens back into images (like an unpatchify head).

    Input:  x ∈ ℝ^{B, T, N_modality, D_model}
    Output: imgs ∈ ℝ^{B, T, 3, H, W}
    """

    def __init__(self, model_dim: int, img_size: tuple[int, int], patch_size: int):
        super().__init__()
        self.img_size = img_size      # (H, W)
        self.patch_size = patch_size  # p

        H, W = img_size
        assert H % patch_size == 0 and W % patch_size == 0, \
            "img_size must be divisible by patch_size"

        self.num_patches_h = H // patch_size
        self.num_patches_w = W // patch_size

        # (B, D, h, w) -> (B, 3*p^2, h, w)
        self.conv = nn.Conv2d(
            in_channels=model_dim,
            out_channels=3 * (patch_size ** 2),
            kernel_size=1,
        )

        # (B, 3*p^2, h, w) -> (B, 3, H, W)
        self.pixel_shuffle = nn.PixelShuffle(patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N_modality, D_model), where N_modality = (H/p)*(W/p)
        """
        B, T, N, D = x.shape
        H, W = self.img_size

        assert N == self.num_patches_h * self.num_patches_w, \
            f"N_modality={N} != (H/p)*(W/p)={self.num_patches_h * self.num_patches_w}"

        # Arrange into (B*T, D, h, w)
        x = x.transpose(-1, -2)  # (B, T, D, N)
        x = x.contiguous().view(
            -1,
            D,
            self.num_patches_h,
            self.num_patches_w,
        )                        # (B*T, D, h, w)

        x = self.conv(x)         # (B*T, 3*p^2, h, w)
        x = self.pixel_shuffle(x)  # (B*T, 3, H, W)

        x = x.view(B, T, 3, H, W)
        return x