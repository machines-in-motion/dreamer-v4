import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .blocks import EfficientTransformerBlock
from .blocks import create_temporal_mask, create_encoder_spatial_mask, create_decoder_spatial_mask
from dataclasses import dataclass
from omegaconf import DictConfig, OmegaConf
import math

@dataclass
class CausalTokenizerConfig:
    num_modality_tokens: int
    num_latent_tokens: int
    max_sequence_length: int
    context_length: int
    model_dim: int
    latent_dim: int
    enc_num_layers: int 
    dec_num_layers: int 
    n_heads: int
    n_kv_heads: Optional[int] = None
    dropout_prob: float = 0.0
    qk_norm: bool = True
    patch_size: int = 14
    dual_stream: bool = False
   
class CausalTokenizerEncoder(nn.Module):
    """
    Encoder:
      Input:  x ∈ ℝ^{B, T, S_mod, D_model}
      Output: latents ∈ ℝ^{B, T, N_lat, D_latent},
              patch_embeddings ∈ ℝ^{B, T, S_mod, D_latent}
    """

    def __init__(self, cfg: CausalTokenizerConfig, max_num_forward_steps=None):
        super().__init__()
        self.cfg = cfg

        model_dim = cfg.model_dim
        num_layers = cfg.enc_num_layers

        # S_mod + N_latent
        self.modality_dim_max_seq_len = cfg.num_modality_tokens + cfg.num_latent_tokens

        # Transformer blocks
        self.layers = nn.ModuleList([
            EfficientTransformerBlock(
                model_dim=model_dim,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                dropout_prob=cfg.dropout_prob,
                qk_norm=cfg.qk_norm,
                temporal_dim_max_seq_len=(max_num_forward_steps if max_num_forward_steps is not None else cfg.max_sequence_length),
                modality_dim_max_seq_len=self.modality_dim_max_seq_len,
                context_length=cfg.context_length,
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
        self.register_buffer("spatial_mask", spatial_mask, persistent=True)

        # Project to latent_dim (typically < model_dim)
        self.output_proj = nn.Linear(cfg.model_dim, cfg.latent_dim)
        self.output_nonlinearity = nn.Tanh()

    def forward(self, x: torch.Tensor):
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

        # Append learned latent tokens at the end of each modality sequence
        # learned_latent_tokens: (1, 1, N_latent, D)
        # -> (B, T, N_latent, D) via expand (view-only)
        latent_tokens = self.learned_latent_tokens.expand(B, T, -1, -1)
        x = torch.cat([x, latent_tokens], dim=2)  # (B, T, S_mod + N_latent, D)

        # Pass through transformer stack
        for layer in self.layers:
            x = layer(x, spatial_mask=self.spatial_mask)

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

    def __init__(self, cfg: CausalTokenizerConfig, max_num_forward_steps=None):
        super().__init__()
        self.cfg = cfg

        model_dim = cfg.model_dim
        num_layers = cfg.dec_num_layers

        self.layers = nn.ModuleList([
            EfficientTransformerBlock(
                model_dim=model_dim,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                dropout_prob=cfg.dropout_prob,
                qk_norm=cfg.qk_norm,
                modality_dim_max_seq_len=cfg.num_modality_tokens + cfg.num_latent_tokens,
                temporal_dim_max_seq_len=(max_num_forward_steps if max_num_forward_steps is not None else cfg.max_sequence_length),
                context_length=cfg.context_length,
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
        self.register_buffer("spatial_mask", spatial_mask, persistent=True)
        # Project latents from latent_dim → model_dim
        self.input_proj = nn.Linear(cfg.latent_dim, cfg.model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


        # Project latents to model_dim
        x = self.input_proj(x)  # (B, T, N_latent, D_model)

        # Prepend learned modality tokens
        # learned_modality_tokens: (1, 1, S_mod, D_model) -> (B, T, S_mod, D_model)
        modality_tokens = self.learned_modality_tokens.expand(B, T, -1, -1).contiguous()
        x = torch.cat([modality_tokens, x], dim=2)  # (B, T, S_mod + N_latent, D_model)

        # Transformer stack
        for layer in self.layers:
            x = layer(x, spatial_mask=self.spatial_mask)

        # Return only the decoded modality tokens (drop the latent positions)
        S_mod = self.cfg.num_modality_tokens
        return x[:, :, :S_mod, :]   # (B, T, S_mod, D_model)
    
    def forward_step(self, x: torch.Tensor, start_step_idx: int, update_cache: bool = True) -> torch.Tensor:
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

        # Project latents to model_dim
        x = self.input_proj(x)  # (B, T, N_latent, D_model)

        # Prepend learned modality tokens
        # learned_modality_tokens: (1, 1, S_mod, D_model) -> (B, T, S_mod, D_model)
        modality_tokens = self.learned_modality_tokens.expand(B, T, -1, -1).contiguous()
        x = torch.cat([modality_tokens, x], dim=2)  # (B, T, S_mod + N_latent, D_model)

        # Transformer stack
        for layer in self.layers:
            x = layer.forward_step(x, 
                                   spatial_mask=self.spatial_mask,
                                   start_step_idx=start_step_idx, 
                                   update_cache=update_cache)

        # Return only the decoded modality tokens (drop the latent positions)
        S_mod = self.cfg.num_modality_tokens
        return x[:, :, :S_mod, :]   # (B, T, S_mod, D_model)
    
    def init_cache(self, batch_size: int, device: torch.device, context_length: int, dtype: torch.dtype):
        """Initializes KV caches for all temporal layers."""
        for layer in self.layers:
            layer.init_cache(batch_size, device, context_length, dtype)

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
    
class TokenMasker(nn.Module):
    """
    MAE-style patch dropout used in DreamerV4.
    Applies a per-(B,T) masking probability sampled uniformly
    from [0, max_mask_prob] and replaces masked tokens with a
    single learned mask token.

    Input:
        x: [B, T, N, D] patch tokens

    Output:
        x_masked: [B, T, N, D]
        mask:     [B, T, N] boolean mask
    """

    def __init__(self, model_dim, max_mask_prob=0.9, activate_masking=True):
        super().__init__()
        self.max_mask_prob = max_mask_prob
        self.activate_masking = activate_masking

        # Single learned mask token (shared across patches)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, model_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x, mask=None):
        """
        x: [B, T, N, D]
        mask: optional [B, T, N] boolean mask
        """
        B, T, N, D = x.shape

        # Only apply masking during training or if forced
        if (self.training or self.activate_masking) and self.max_mask_prob > 0.0:

            # --- Sample per-(b,t) dropout probability: [B, T] ---
            drop_p = torch.rand(B, T, device=x.device) * self.max_mask_prob

            # --- Sample random uniform noise per patch ---
            rand = torch.rand(B, T, N, device=x.device)

            # --- Build mask: rand < drop_p[b,t] ---
            if mask is None:
                mask = rand < drop_p.unsqueeze(-1)  # [B, T, N]DictConfig

            # --- Apply mask: broadcast mask_token to [B,T,N,D] ---
            x = torch.where(
                mask.unsqueeze(-1),                     # [B,T,N,1]
                self.mask_token.to(x.dtype),           # [1,1,1,D] → broadcast
                x                                      # original tokens
            )
        else:
            # If no masking, still return an all-False mask
            if mask is None:
                mask = torch.zeros(B, T, N, dtype=torch.bool, device=x.device)

        return x
    
class TokenizerWrapper(nn.Module):
    def __init__(self, cfg:DictConfig, max_num_forward_steps=None):
        super().__init__()
        self.cfg = cfg
        tokenizer_cfg = CausalTokenizerConfig(**OmegaConf.to_object(cfg.tokenizer)) 
        self.encoder = CausalTokenizerEncoder(tokenizer_cfg, max_num_forward_steps=max_num_forward_steps)
        self.decoder = CausalTokenizerDecoder(tokenizer_cfg, max_num_forward_steps=max_num_forward_steps)
        self.patchifier = ImagePatchifier(cfg.tokenizer.patch_size, cfg.tokenizer.model_dim)
        self.image_head = TokensToImageHead(cfg.tokenizer.model_dim, cfg.dataset.resolution, cfg.tokenizer.patch_size)
        self.masker = TokenMasker(cfg.tokenizer.model_dim)

    def forward(self, images):
        images = (images*2.)-1. # Translate the images in +-1 range
        tokens = self.patchifier(images)
        masked_tokens = self.masker(tokens)
        z, _ = self.encoder(masked_tokens)
        z_decoded = self.decoder(z)
        recon_images = self.image_head(z_decoded)
        # return  torch.clamp((recon_images + 1)/2., 0., 1.)
        return (recon_images + 1)/2. 
    
    def decode_step(self, x: torch.Tensor, start_step_idx: int, update_cache: bool = True):
        z_decoded = self.decoder.forward_step(x, start_step_idx, update_cache)
        imgs_recon = self.image_head(z_decoded)
        imgs_recon = (imgs_recon + 1.0) / 2.0  # [-1,1] → [0,1]
        imgs_recon = torch.clamp(imgs_recon, 0.0, 1.0)
        return imgs_recon
    
    def decode(self, latents):
        z_decoded = self.decoder(latents)
        imgs_recon = self.image_head(z_decoded)
        imgs_recon = (imgs_recon + 1.0) / 2.0  # [-1,1] → [0,1]
        imgs_recon = torch.clamp(imgs_recon, 0.0, 1.0)
        return imgs_recon
    
    def init_cache(self,
                   batch_size, 
                   context_length, 
                   device, 
                   dtype):
        self.decoder.init_cache(batch_size, device, context_length, dtype)

    def encode(self, images):
        images = (images*2.)-1. # Translate the images in +-1 range
        tokens = self.patchifier(images)
        z, _ = self.encoder(tokens)
        return z