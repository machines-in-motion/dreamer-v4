import torch
from enum import Enum
import math
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional, List


# Todo: verify the implementation of SwiGLU numically matches the paper
class SwiGLU(nn.Module):
    """
    SwiGLU activation block from:
    'GLU Variants Improve Transformer', Shazeer 2020.
    
    Computes: Swish(xW1) * (xW2)
    """
    def __init__(self, dim_in, dim_hidden):
        super().__init__()
        # Two linear projections: W and V in the paper
        self.W = nn.Linear(dim_in, dim_hidden, bias=True)
        self.V = nn.Linear(dim_in, dim_hidden, bias=True)

    def forward(self, x):
        """
        x: (B, T, D)
        """
        xW = self.W(x)                  # gate input
        xV = self.V(x)                  # value input

        # Swish(xW) = xW * sigmoid(xW)
        swish_gate = xW * torch.sigmoid(xW)

        return swish_gate * xV          # elementwise product

class RopeEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048, device = 'cpu'):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.device = device
        # Cache the sin/cos computations to avoid recomputation
        thetas = torch.tensor([10000**(-2*i/dim) for i in range(dim//2)])
        thetas = torch.stack([thetas, thetas], dim=0).transpose(0,1).reshape(-1)
        thetas_all = torch.stack([thetas*i for i in range(max_seq_len)], dim=0)
        cos_cache = thetas_all.cos() # TxD
        sin_cache = thetas_all.sin() # TxD
        self.register_buffer('cos_emb', cos_cache.unsqueeze(0).unsqueeze(0)) #1x1xTxD
        self.register_buffer('sin_emb', sin_cache.unsqueeze(0).unsqueeze(0)) #1x1xTxD

    def forward(self, q: torch.Tensor, k:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor] :
        B, H, T, d = q.shape
        cos_emb = self.cos_emb[:, :, :T, :]  # 1x1xTxD
        sin_emb = self.sin_emb[:, :, :T, :]  # 1x1xTxD 
        q_odd, q_even = q[..., ::2], q[..., 1::2]
        qJ = torch.stack([-q_even, q_odd], dim=-1).reshape_as(q)
        k_odd, k_even = k[..., ::2], k[..., 1::2]
        kJ = torch.stack([-k_even, k_odd], dim=-1).reshape_as(k)
        
        q_rot = (q * cos_emb) + (qJ * sin_emb)
        k_rot = (k * cos_emb) + (kJ * sin_emb)
        return q_rot, k_rot
    
class Attention(nn.Module):
    """
    Multi-head (self/cross) attention block with optional QK-Norm and RoPE and GQA.
    """
    def __init__(self, model_dim, n_heads, n_kv_heads=None, causal = False, dropout_prob = 0, qk_norm=True, max_seq_len=128, rope_embedder = None, flash_attention=False):
        super().__init__()
        assert model_dim%n_heads==0, 'Model dimension should be devisible by the number of heads'
        self.d = model_dim
        self.dk = model_dim // n_heads
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.dropout_prob = dropout_prob
        self.rope_embedder = rope_embedder
        self.qk_norm = qk_norm
        self.causal = causal
        self.flash_attention = flash_attention
        self.W_q = nn.Linear(self.d, self.d, bias=False)
        self.W_k = nn.Linear(self.d, self.dk*self.n_kv_heads, bias=False)
        self.W_v = nn.Linear(self.d, self.dk*self.n_kv_heads, bias=False)
        self.W_o = nn.Linear(self.d, self.d, bias=False)
        if not flash_attention:
            self.register_buffer("g", torch.tensor(math.log2(float(max_seq_len**2-max_seq_len)), dtype=torch.float32)) # The normalization constant in QK-Norm is active.

    def forward(self,
                 q: torch.Tensor,
                 k: torch.Tensor,
                 v: torch.Tensor,
                 mask: Optional[torch.Tensor] = None)-> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-head (self/cross) attention forward pass.
        Args:
            q: (B, T_q, D) queries
            k: (B, T_k, D) keys
            v: (B, T_k, D) values
            mask: (B, 1, T_q, T_k) or broadcastable mask
        
        Returns:
            y: (B, T_q, D) attention output
            A: (B, n_heads, T_q, T_k) attention weights
        """
        if mask is not None and mask.dtype not in (torch.bool, torch.float32, torch.float16, torch.bfloat16):
            mask = mask.to(torch.bool)
        B, T_q, _ = q.shape
        B, T_k, _ = k.shape
        Q = self.W_q(q) # BxTxd
        K = self.W_k(k) # BxTxdk
        V = self.W_v(v) # BxTxdk
        Q = Q.view(B, T_q, self.n_heads, self.dk).transpose(1,2).contiguous() # BxhxTxd
        K = K.view(B, T_k, self.n_kv_heads, self.dk).transpose(1,2).contiguous() # Bxn_kvxT_kxd
        V = V.view(B, T_k, self.n_kv_heads, self.dk).transpose(1,2).contiguous() # Bxn_kvxT_kxd
        # Normalize the features per head if qk_norm is active
        if self.qk_norm:
            Q = F.normalize(Q, dim=-1)
            K = F.normalize(K, dim=-1)

        if self.rope_embedder is not None:
            Q, K = self.rope_embedder(Q, K)
        
        if self.flash_attention:
            Y = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=mask, 
                dropout_p=self.dropout_prob if self.training else 0.0,
                is_causal=self.causal,
                scale = None,
                enable_gqa= False if self.n_kv_heads == self.n_heads else True,
            )  # [B, n_heads, Tq, dk]
        else:
            Y = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=mask, 
                dropout_p=self.dropout_prob if self.training else 0.0,
                is_causal=self.causal,
                scale = self.g,
                enable_gqa= False if self.n_kv_heads == self.n_heads else True,
            )  # [B, n_heads, Tq, dk]
        Y = Y.transpose(1, 2).contiguous().view(B, T_q, self.d)
        Y = self.W_o(Y)
        return Y

class AxialAttention(nn.Module):
    """
    Axial wrapper around the Attention block.
    Operates along a chosen 1D axis of a N-D tensor (B, dim1, ..., dimN, D).
    """
    def __init__(
        self,
        inner_attn: nn.Module,                      # Attention(...)
    ):
        super().__init__()
        self.attn = inner_attn  # expects (B*, T, D) → (B*, T, D)

    def forward(self,
                q: torch.Tensor, 
                k: torch.Tensor, 
                v:torch.Tensor, 
                dim: int, 
                mask: Optional[torch.Tensor] = None):
        
        dims_q = list(q.shape)
        assert dim > 0 and dim < q.dim(), "Dimension for attention must be between 1 and q.dim()-1"
        D = dims_q[-1]
        Tq_k = dims_q[dim]
        dims_kv = list(k.shape)
        Tk_k = dims_kv[dim]
        reordered_q = q.transpose(dim, -2).contiguous().view(-1, Tq_k, D)
        reordered_k = k.transpose(dim, -2).contiguous().view(-1, Tk_k, D)
        reordered_v = v.transpose(dim, -2).contiguous().view(-1, Tk_k, D)
        if mask is not None:
            assert mask.shape[-2] == q.shape[dim] and mask.shape[-1] == k.shape[dim], "Mask shape does not match the attention dimensions."
            if mask.dim() ==2:
                mask = mask.unsqueeze(0)  # 1xTq xTk
        Y = self.attn(reordered_q, reordered_k, reordered_v, mask)
        Y = Y.view((dims_q[:dim]+dims_q[dim+1:-1]+[dims_q[dim], D])).transpose(dim, -2).contiguous()
        return Y

class AxialSelfAttentionBlock(nn.Module):
    """
    Axial Self-Attention block with residual connection and RMSNorm.
    Operates along a chosen 1D axis of a N-D tensor (B, dim1, ..., dimN, D).
    """
    def __init__(self, model_dim, n_heads, n_kv_heads=None, causal = False, dropout_prob = 0, qk_norm=True, max_seq_len=128, rope_embedder = None):
        super().__init__()  
        self.model_dim = model_dim  
        self.n_heads = n_heads  
        self.n_kv_heads = n_kv_heads
        self.causal = causal    
        self.dropout_prob = dropout_prob
        self.qk_norm = qk_norm
        self.max_seq_len = max_seq_len
        self.rope_embedder = rope_embedder
        
        self.attnk = AxialAttention(
            inner_attn=Attention(
                model_dim=model_dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                causal=causal,
                dropout_prob=dropout_prob,
                qk_norm=qk_norm,
                max_seq_len=max_seq_len,
                rope_embedder=rope_embedder
            )
        )
        self.Dense = nn.Linear(model_dim, model_dim)
        self.rmsnorm = nn.RMSNorm(model_dim)
    
    def forward(self,
                x: torch.Tensor, 
                dim: int, 
                mask: Optional[torch.Tensor] = None):
        """
        x: (B, dim1, ..., dimN, D)
        dim: int, the dimension along which attention is applied
        mask: Optional[torch.Tensor], attention mask
        """
        x_norm = self.rmsnorm(x)
        attn_out = self.attnk(x_norm, x_norm, x_norm, dim, mask)
        out = self.Dense(attn_out) + x
        return out
    
class FeedForwardBlock(nn.Module):
    """
    Feed-Forward block with residual connection and RMSNorm.
    Uses SwiGLU nonlinearity.
    """
    def __init__(self, model_dim):
        super().__init__()
        self.model_dim = model_dim
        self.Dense1 = nn.Linear(model_dim, model_dim*4)
        self.Dense2 = nn.Linear(model_dim*4, model_dim)
        self.rmsnorm = nn.RMSNorm(model_dim)
        self.nonlinearity = SwiGLU(model_dim*4, model_dim*4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rmsnorm(x)
        x = self.Dense2(self.nonlinearity(self.Dense1(x))) + x
        return x
    
class AxialEncoderBlock(nn.Module):
    """
    Axial Transformer Encoder block with Axial Self-Attention and Feed-Forward Network.
    Operates along a chosen 1D axis of a N-D tensor (B, dim 1, ..., dim N, D).
    """
    def __init__(self, model_dim, n_heads, n_kv_heads=None, causal = False, dropout_prob = 0, qk_norm=True, max_seq_len=128, rope_embedder = None):
        super().__init__()
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.causal = causal
        self.dropout_prob = dropout_prob
        self.qk_norm = qk_norm
        self.max_seq_len = max_seq_len
        self.rope_embedder = rope_embedder
        self.attn_block = AxialSelfAttentionBlock(
            model_dim=model_dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            causal=causal,
            dropout_prob=dropout_prob,
            qk_norm=qk_norm,
            max_seq_len=max_seq_len,
            rope_embedder=rope_embedder
        )
        self.ffn_block = FeedForwardBlock(model_dim=model_dim)
    
    def forward(self, x: torch.Tensor, dim: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.attn_block(x, dim, mask)
        x = self.ffn_block(x)
        return x    

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
    
class AxialCrossAttentionBlock(nn.Module):
    """
    Axial Cross-Attention block with residual connection and RMSNorm.
    Operates along a chosen 1D axis of a N-D tensor (B, dim1, ..., dimN, D).
    """
    def __init__(self, model_dim, n_heads, n_kv_heads=None, causal = False, dropout_prob = 0, qk_norm=True, max_seq_len=128, rope_embedder = None):
        super().__init__()  
        self.model_dim = model_dim  
        self.n_heads = n_heads  
        self.n_kv_heads = n_kv_heads
        self.causal = causal    
        self.dropout_prob = dropout_prob
        self.qk_norm = qk_norm
        self.max_seq_len = max_seq_len
        self.rope_embedder = rope_embedder
        
        self.attnk = AxialAttention(
            inner_attn=Attention(
                model_dim=model_dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                causal=causal,
                dropout_prob=dropout_prob,
                qk_norm=qk_norm,
                max_seq_len=max_seq_len,
                rope_embedder=rope_embedder
            )
        )
        self.Dense = nn.Linear(model_dim, model_dim)
        self.rmsnorm1 = nn.RMSNorm(model_dim)
        self.rmsnorm2 = nn.RMSNorm(model_dim)

    def forward(self,
                x: torch.Tensor, 
                context: torch.Tensor,
                dim: int, 
                mask: Optional[torch.Tensor] = None):
        """
        x: (B, dim1, ..., dimN, D)
        context: (B, dim1, ..., dimM, D)
        dim: int, the dimension along which attention is applied
        mask: Optional[torch.Tensor], attention mask
        """
        x = self.rmsnorm1(x)
        context = self.rmsnorm2(context)
        attn_out = self.attnk(x, context, context, dim, mask)
        out = self.Dense(attn_out) + x
        return out