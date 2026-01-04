import torch
from enum import Enum
import math
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional, List
from torch.nn.attention import SDPBackend, sdpa_kernel
from .utils import create_encoder_spatial_mask, create_decoder_spatial_mask, create_temporal_mask

class LayerType(Enum):
    SPATIAL = "spatial"
    TEMPORAL = "temporal"

class FeedForwardSwiGLU(nn.Module):
    """
    SwiGLU FFN:
      y = down( dropout( SiLU(up(x)) * gate(x) ) )
    Hidden size defaults to ~8/3 * d_model as commonly used in LLaMA/HF.
    """
    def __init__(self, d_model: int, d_hidden: int | None = None, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        if d_hidden is None:
            # 8/3 expansion; round to multiple of 8 for tensor cores if desired
            d_hidden = (8 * d_model) // 3
        self.up = nn.Linear(d_model, d_hidden, bias=bias)
        self.gate = nn.Linear(d_model, d_hidden, bias=bias)
        self.down = nn.Linear(d_hidden, d_model, bias=bias)
        self.drop = nn.Dropout(dropout)

    #@torch.compile(fullgraph=False, mode="reduce-overhead", dynamic=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = F.silu(self.up(x))
        g = self.gate(x)
        h = self.drop(u * g)
        return self.down(h)
    
class RopeEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        thetas = torch.tensor([10000**(-2*i/dim) for i in range(dim//2)])
        thetas = torch.stack([thetas, thetas], dim=0).transpose(0,1).reshape(-1)
        # Pre-compute full table
        thetas_all = torch.stack([thetas*i for i in range(max_seq_len)], dim=0)
        cos_cache = thetas_all.cos() # TxD
        sin_cache = thetas_all.sin() # TxD
        self.register_buffer('cos_emb', cos_cache.unsqueeze(0).unsqueeze(0)) #1x1xTxD
        self.register_buffer('sin_emb', sin_cache.unsqueeze(0).unsqueeze(0)) #1x1xTxD

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, T, d = q.shape
        
        # Slice with offset for inference
        cos_emb = self.cos_emb[:, :, offset : offset + T, :]
        sin_emb = self.sin_emb[:, :, offset : offset + T, :]
        
        q_odd, q_even = q[..., ::2], q[..., 1::2]
        qJ = torch.stack([-q_even, q_odd], dim=-1).reshape_as(q)
        k_odd, k_even = k[..., ::2], k[..., 1::2]
        kJ = torch.stack([-k_even, k_odd], dim=-1).reshape_as(k)
        
        q_rot = (q * cos_emb) + (qJ * sin_emb)
        k_rot = (k * cos_emb) + (kJ * sin_emb)
        return q_rot, k_rot
    
@dataclass
class KVCache:
    k: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None
    max_window: int = 32 # The moving horizon size

    def update(self, new_k, new_v):
        # Concatenate along time dimension (dim=2)
        if self.k is None:
            self.k = new_k
            self.v = new_v
        else:
            self.k = torch.cat([self.k, new_k], dim=2)
            self.v = torch.cat([self.v, new_v], dim=2)
        
        # Enforce Moving Horizon
        if self.k.shape[2] > self.max_window:
            self.k = self.k[:, :, -self.max_window:, :]
            self.v = self.v[:, :, -self.max_window:, :]
        return self.k, self.v
    
    def get_view(self, new_k, new_v):
        """
        Returns combined K/V for attention computation WITHOUT updating the permanent cache.
        Used during the intermediate diffusion steps.
        """
        if self.k is None:
            return new_k, new_v
        return torch.cat([self.k, new_k], dim=2), torch.cat([self.v, new_v], dim=2)

class Attention(nn.Module):
    """
    Multi-head (self/cross) attention block with optional QK-Norm and RoPE and GQA.
    """
    def __init__(self, model_dim, n_heads, n_kv_heads=None, dropout_prob = 0, qk_norm=True, max_seq_len=128, rope_embedder = None, flash_attention=False):
        super().__init__()
        assert model_dim%n_heads==0, 'Model dimension should be devisible by the number of heads'
        self.d = model_dim
        self.dk = model_dim // n_heads
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.dropout_prob = dropout_prob
        self.rope_embedder = rope_embedder
        self.qk_norm = qk_norm
        self.flash_attention = flash_attention
        self.W_q = nn.Linear(self.d, self.d, bias=False)
        self.W_k = nn.Linear(self.d, self.dk*self.n_kv_heads, bias=False)
        self.W_v = nn.Linear(self.d, self.dk*self.n_kv_heads, bias=False)
        self.W_o = nn.Linear(self.d, self.d, bias=False)
        if self.qk_norm:
            self.g = nn.Parameter(torch.ones(1, self.n_heads, 1, 1) * math.log(max_seq_len))

    def forward(self,
                 q: torch.Tensor,
                 k: torch.Tensor,
                 v: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 causal: Optional[bool] = False,
                 kv_cache: Optional[KVCache] = None)-> Tuple[torch.Tensor, torch.Tensor]:
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
            Q = self.g*Q

        # Handle RoPE and Caching
        if kv_cache is not None:
            # 1. Determine offset based on what is already in cache
            current_cache_len = kv_cache.k.shape[2] if kv_cache.k is not None else 0
            
            # 2. Apply RoPE to the NEW keys/queries with the correct absolute position offset
            if self.rope_embedder is not None:
                Q, K = self.rope_embedder(Q, K, offset=current_cache_len)
            # 3. Update Cache (concatenates and slices window)
            K, V = kv_cache.update(K, V)
        else:
            if self.rope_embedder is not None:
                Q, K = self.rope_embedder(Q, K)
        
        if self.flash_attention and Q.dtype in (torch.float16, torch.bfloat16) and Q.is_cuda:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                Y = F.scaled_dot_product_attention(
                    Q, K, V,
                    attn_mask=mask, 
                    dropout_p=self.dropout_prob if self.training else 0.0,
                    is_causal=causal,
                    scale = None,
                    enable_gqa= False if self.n_kv_heads == self.n_heads else True,
                )  # [B, n_heads, Tq, dk]
        else:
                Y = F.scaled_dot_product_attention(
                    Q, K, V,
                    attn_mask=mask, 
                    dropout_p=self.dropout_prob if self.training else 0.0,
                    is_causal=causal,
                    scale = None,
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
    def __init__(self, inner_attn: nn.Module):
        super().__init__()
        self.attn = inner_attn  # expects (B*, T, D, ...) → (B*, T, D, ...)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        dim: int,
        mask: Optional[torch.Tensor] = None,
        causal: Optional[bool] = False,
        kv_cache: Optional[KVCache] = None,
    ):
        # q: (B, d1, ..., dN, D)
        dims_q = list(q.shape)
        assert dim > 0 and dim < q.dim() - 1, "dim must be between 1 and q.dim()-2"
        D = dims_q[-1]
        Tq = dims_q[dim]

        dims_k = list(k.shape)
        Tk = dims_k[dim]
        assert dims_k[:-1] == dims_q[:-1], "q and k must have same non-feature dims"
        assert v.shape[:-1] == k.shape[:-1], "k and v must have same non-feature dims"
        assert k.shape[-1] == v.shape[-1] == D, "Feature dim mismatch"

        # Move axial dim to -2 for all tensors
        q_t = q.transpose(dim, -2).contiguous()  # (B, ..., Tq, D)
        k_t = k.transpose(dim, -2).contiguous()  # (B, ..., Tk, D)
        v_t = v.transpose(dim, -2).contiguous()  # (B, ..., Tk, D)

        # Flatten all non-(T, D) dims into batch
        B_star = int(q_t.numel() // (Tq * D))
        q_flat = q_t.view(B_star, Tq, D)
        k_flat = k_t.view(B_star, Tk, D)
        v_flat = v_t.view(B_star, Tk, D)

        # Handle mask: expected to index (Tq, Tk)
        mask_flat = None
        if mask is not None:
            # Allowed shapes:
            # - (Tq, Tk)
            # - (1, Tq, Tk)
            # - (B, Tq, Tk) matching original batch
            # - (B, 1, Tq, Tk) matching original batch
            if mask.dim() == 2:
                assert mask.shape == (Tq, Tk)
                mask_flat = mask.unsqueeze(0)  # (1, Tq, Tk)
            elif mask.dim() == 3:
                assert mask.shape[-2:] == (Tq, Tk)
                # Assume first dim is original batch; broadcast along other spatial dims
                # Flatten along same non-(T, D) dims as q_t
                # Here we require that only batch is present: (B, Tq, Tk)
                assert mask.shape[0] == dims_q[0], "3D mask must be (B, Tq, Tk)"
                mask_flat = mask.repeat_interleave(
                    int(B_star // dims_q[0]), dim=0
                )  # (B*, Tq, Tk)
            elif mask.dim() == 4:
                assert mask.shape[-2:] == (Tq, Tk)
                # (B, 1, Tq, Tk) or (B, H, Tq, Tk); we ignore head dim and flatten batch
                B_mask = mask.shape[0]
                assert B_mask == dims_q[0], "4D mask batch must match q batch"
                # Collapse any head dim into batch
                mask_flat = mask.view(B_mask, -1, Tq, Tk)
                B_star_mask = mask_flat.shape[0] * mask_flat.shape[1]
                assert B_star_mask == B_star, "Flattened mask batch must match q_flat batch"
                mask_flat = mask_flat.view(B_star, Tq, Tk)
            else:
                raise ValueError("Unsupported mask rank for AxialAttention")

        # Call inner attention
        Y_flat = self.attn(q_flat, k_flat, v_flat, mask_flat, causal=causal, kv_cache=kv_cache)  # (B*, Tq, D)

        # Reshape back to original dims
        Y_t = Y_flat.view(*q_t.shape)  # (B, d1, ..., d_{dim-1}, d_{dim+1}, ..., dN, Tq, D)
        Y = Y_t.transpose(dim, -2).contiguous()  # (B, d1, ..., dN, D)
        return Y

class EfficientTransformerLayer(nn.Module):
    """
    A single axial transformer layer:
    - RMSNorm → AxialAttention (spatial or temporal)
    - RMSNorm → FeedForwardSwiGLU
    """

    def __init__(
        self,
        model_dim: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int,
        layer_type: LayerType,
        dropout_prob: float = 0.0,
        qk_norm: bool = True,
        rope_embedder: Optional[RopeEmbedding] = None,
    ):
        super().__init__()
        assert isinstance(layer_type, LayerType)

        self.layer_type = layer_type
        self.model_dim = model_dim

        # Use shared RoPE if available
        self.rope = rope_embedder or RopeEmbedding(model_dim // n_heads, max_seq_len)

        # Axial attention
        self.attn = AxialAttention(
            inner_attn=Attention(
                model_dim=model_dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                dropout_prob=dropout_prob,
                qk_norm=qk_norm,
                max_seq_len=max_seq_len,
                rope_embedder=self.rope,
            )
        )

        # FFN + norms
        self.norm1 = nn.RMSNorm(model_dim)
        self.norm2 = nn.RMSNorm(model_dim)
        self.ffn = FeedForwardSwiGLU(model_dim, None, dropout_prob)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, 
                x, 
                temporal_mask=None, 
                spatial_mask=None,
                layer_cache: Optional[KVCache] = None):
        """
        x: (B, T, S, D)
        """
        # Attention block
        h = self.norm1(x)
        if self.layer_type == LayerType.TEMPORAL:
            h = self.attn(h, h, h, dim=1, causal=True, kv_cache = layer_cache)
        else:
            h = self.attn(h, h, h, dim=2, mask=spatial_mask) # Spatial dimention has no cacheing
        x = x + self.dropout(h)

        # FFN block
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + self.dropout(h)

        return x

class EfficientTransformerBlock(nn.Module):
    """
    A sequence of EfficientTransformerLayer objects.
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

        # Default block layout
        if layer_types is None:
            layer_types = [
                LayerType.SPATIAL,
                LayerType.SPATIAL,
                LayerType.SPATIAL,
                LayerType.TEMPORAL,
            ]

        # Shared RoPE
        rope = RopeEmbedding(model_dim // n_heads, max_seq_len)

        # Build layers
        self.layers = nn.ModuleList([
            EfficientTransformerLayer(
                model_dim=model_dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                max_seq_len=max_seq_len,
                layer_type=t,
                dropout_prob=dropout_prob,
                qk_norm=qk_norm,
                rope_embedder=rope,
            )
            for t in layer_types
        ])

        self.model_dim = model_dim

    def forward(self, x, temporal_mask=None, spatial_mask=None, block_caches: Optional[List[KVCache]]=None):
        assert x.size(-1) == self.model_dim
        if block_caches is None:
            block_caches = [None] * len(self.layers)
        for i, layer in enumerate(self.layers):
            x = layer(
                x,
                temporal_mask=temporal_mask,
                spatial_mask=spatial_mask,
                layer_cache=block_caches[i],
            )

        return x
    
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