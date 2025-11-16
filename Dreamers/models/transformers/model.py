import torch
from enum import Enum
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
from .blocks import AxialSelfAttentionBlock, FeedForwardBlock, RopeEmbedding, AxialEncoderBlock
from .utils import create_temporal_mask, create_encoder_spatial_mask, create_decoder_spatial_mask
from dataclasses import dataclass

class LayerType(Enum):
    SPATIAL = "spatial"
    TEMPORAL = "temporal"

class EfficientTransformerBlock(nn.Module):
    """
    A block composed of spatial and/or temporal axial-attention layers,
    in the order specified by `layer_types`.

    Expected input:
        x: (B, T, S, D)
            B = batch size
            T = temporal dimension
            S = spatial dimension (e.g., patches)
            D = embedding dimension

    Args:
        model_dim: embedding dimension D
        n_heads: number of attention heads
        n_kv_heads: number of key/value heads
        max_seq_len: maximum sequence length for RoPE
        dropout_prob: dropout probability
        qk_norm: enable QK normalization
        layer_types: sequence of LayerType enums defining the order of layers
    """

    def __init__(
        self,
        model_dim: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        dropout_prob: float = 0.0,
        qk_norm: bool = True,
        layer_types: Optional[List[LayerType]] = None,
    ):
        super().__init__()

        # Default block: 3 spatial layers + 1 temporal layer
        if layer_types is None:
            layer_types = [
                LayerType.SPATIAL,
                LayerType.SPATIAL,
                LayerType.SPATIAL,
                LayerType.TEMPORAL,
            ]

        # Validate layer types
        if not all(isinstance(t, LayerType) for t in layer_types):
            raise TypeError("layer_types must be a list of LayerType enums")

        self.layer_types = layer_types
        self.model_dim = model_dim

        # Shared RoPE embedder
        self.rope = RopeEmbedding(model_dim//n_heads, max_seq_len)

        # Factory to build Axial blocks
        def make_layer():
            return AxialEncoderBlock(
                model_dim=model_dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                dropout_prob=dropout_prob,
                qk_norm=qk_norm,
                max_seq_len=max_seq_len,
                rope_embedder=self.rope,
            )

        # One block per layer specification
        self.layers = nn.ModuleList([make_layer() for _ in layer_types])

    def forward(
        self,
        x: torch.Tensor,
        temporal_mask: Optional[torch.Tensor] = None,
        spatial_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: input tensor of shape (B, T, S, D)
            temporal_mask: (T, T) or (B, T, T)
            spatial_mask: (S, S) or (B, S, S)
        """
        # Basic shape check
        assert x.size(-1) == self.model_dim, \
            f"Expected last dim = {self.model_dim}, got {x.size(-1)}"

        for layer_type, layer in zip(self.layer_types, self.layers):
            if layer_type == LayerType.SPATIAL:
                x = layer(x, dim=2, mask=spatial_mask)
            else:  # LayerType.TEMPORAL
                x = layer(x, dim=1, mask=temporal_mask)

        return x
    
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
        
        
    