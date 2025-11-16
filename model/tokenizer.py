import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .blocks import EfficientTransformerBlock
from .utils import create_temporal_mask, create_encoder_spatial_mask, create_decoder_spatial_mask
from dataclasses import dataclass

@dataclass
class CausalTokenizerConfig:
    num_modality_tokens: int
    num_latent_tokens: int
    max_context_lenghth: int
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
    def __init__(self, cfg: CausalTokenizerConfig):
        super().__init__()
        self.cfg = cfg
        model_dim = cfg.model_dim
        num_layers = cfg.enc_num_layers
        self.num_spatial_tokens = cfg.num_modality_tokens + cfg.num_latent_tokens 
        self.max_seq_len = max(cfg.max_context_lenghth, self.num_spatial_tokens)
        self.layers = nn.ModuleList( [EfficientTransformerBlock(model_dim=model_dim,
                                                                n_heads=cfg.n_heads,
                                                                n_kv_heads= cfg.n_kv_heads,
                                                                dropout_prob = cfg.dropout_prob,
                                                                qk_norm=cfg.qk_norm,
                                                                max_seq_len=self.max_seq_len) for _ in range(num_layers)])
        
        self.learned_latent_tokens = nn.Parameter(torch.randn(1, 1, cfg.num_latent_tokens, model_dim))
        self.spatial_mask = create_encoder_spatial_mask(cfg.num_modality_tokens,
                                                        cfg.num_latent_tokens)
        self.output_proj = nn.Linear(self.cfg.model_dim, cfg.latent_dim) # project to latent dim (smaller than model dim)
        self.output_nonlinearity = nn.Tanh()

        
        
    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, T, S, D = x.shape
        # Append learned latent tokens at the end of each modality sequence
        x = torch.cat([x,
                       self.learned_latent_tokens.expand(B, T, -1, -1)], dim=2)  # B, T, S+N_latent, D
        
        if causal:
            temporal_mask = create_temporal_mask(T, device=x.device)
        else:
            temporal_mask = None
        for layer in self.layers:
            x = layer(x, temporal_mask=temporal_mask, spatial_mask=self.spatial_mask)
        x = self.output_nonlinearity(self.output_proj(x))
        return x[:, :,self.cfg.num_modality_tokens:,:], x[:, :,:self.cfg.num_modality_tokens,:] # Latents, patch embeddings
    
class CausalTokenizerDecoder(nn.Module):
    def __init__(self, cfg: CausalTokenizerConfig):
        super().__init__()
        self.cfg = cfg
        model_dim = cfg.model_dim
        num_layers = cfg.dec_num_layers
        self.num_spatial_tokens = cfg.num_modality_tokens + cfg.num_latent_tokens 
        self.max_seq_len = max(cfg.max_context_lenghth, self.num_spatial_tokens)
        self.layers = nn.ModuleList( [EfficientTransformerBlock(model_dim=model_dim,
                                                                n_heads=cfg.n_heads,
                                                                n_kv_heads= cfg.n_kv_heads,
                                                                dropout_prob = cfg.dropout_prob,
                                                                qk_norm=cfg.qk_norm,
                                                                max_seq_len=self.max_seq_len) for _ in range(num_layers)])
        
        self.learned_latent_tokens = nn.Parameter(torch.randn(1, 1, cfg.num_modality_tokens, model_dim))
        self.spatial_mask = create_decoder_spatial_mask(cfg.num_modality_tokens,
                                                        cfg.num_latent_tokens)
        self.input_proj = nn.Linear(cfg.latent_dim, cfg.model_dim) # project to model dim
    
    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        assert x.dim() == 4, "Input tensor must be 4-dimensional (B, T, S, D)"
        assert x.size(-1) == self.cfg.latent_dim, \
            f"Expected last dim = {self.cfg.latent_dim}, got {x.size(-1)}"
        assert x.size(2) == self.cfg.num_latent_tokens, \
            f"Expected spatial dim = {self.cfg.num_latent_tokens}, got {x.size(2)}"
        assert x.size(1) <= self.max_seq_len, \
            f"Temporal dimension {x.size(1)} exceeds max_seq_len {self.max_seq_len}"
        B, T, S, D = x.shape
        x = self.input_proj(x)  # project input to model dim
        # Append learned latent tokens at the beginning of each modality sequence
        x = torch.cat([self.learned_latent_tokens.expand(B, T, -1, -1),
                       x], dim=2)  # B, T, S+N_latent, D
        
        if causal:
            temporal_mask = create_temporal_mask(T, device=x.device)
        else:
            temporal_mask = None
        for layer in self.layers:
            x = layer(x, temporal_mask=temporal_mask, spatial_mask=self.spatial_mask)
        return x[:, :, :self.cfg.num_modality_tokens,...]
        


class ImagePatchifier(nn.Module):
    def __init__(self, patch_size, model_dim, input_channels=3):
        super().__init__()
        self.patch_size = patch_size
        self.input_channels = input_channels
        self.model_dim = model_dim
        self.cnn = nn.Conv2d(input_channels, model_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        assert x.dim()== 5, 'Input should be of shape BxTxCxHxW'
        assert x.shape[-3] == self.input_channels, f'The number of image channels {x.shape[-3]}, does not match the number of input channels {self.input_channels}'
        B, T, C, H, W = x.shape
        x = x.view(-1, C, H, W)
        x = self.cnn(x).flatten(-2, -1).transpose(-2, -1).view(B, T, -1, self.model_dim)
        return x
    
class TokensToImageHead(nn.Module):
    def __init__(self, model_dim, img_size, patch_size):
        super().__init__()
        self.img_size = img_size  # (H, W)
        self.patch_size = patch_size  # p
        self.num_patches_h = img_size[0] // patch_size
        self.num_patches_w = img_size[1] // patch_size
        self.conv = nn.Conv2d(model_dim, 3 * (patch_size ** 2), kernel_size=1)  # (B, D, h, w) -> (B, 3*p^2, h, w)
        self.pixel_shuffle = nn.PixelShuffle(patch_size)  # (B, 3*p^2, h, w) -> (B, 3, H, W)

    def forward(self, x):  # x: (B, T, N_modality, D)
        B, T, N, D = x.shape
        x = x.transpose(-1, -2)  # (B, T, D, N)
        x = x.contiguous().view(-1, D, self.num_patches_h, self.num_patches_w)  # (B*T, D, h, w)
        x = self.conv(x)  # (B, 3*p^2, h, w)
        x = self.pixel_shuffle(x)  # (B, 3, H, W)
        return x.view(B, T, 3, self.img_size[0], self.img_size[1])
        
    