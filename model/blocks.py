import torch
from enum import Enum
import math
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional, List
from torch.nn.attention import SDPBackend, sdpa_kernel


# class FeedForward(nn.Module):
#     def __init__(self, d_model: int, d_hidden: int, dropout: float = 0.0):
#         super().__init__()
#         self.fc1 = nn.Linear(d_model, d_hidden)
#         self.fc2 = nn.Linear(d_hidden, d_model)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.silu(x)
#         x = self.dropout(x)
#         x = self.fc2(x)
#         return x
    
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
    
# class SwiGLU(nn.Module):
#     """
#     SwiGLU activation block from:
#     'GLU Variants Improve Transformer', Shazeer 2020.
    
#     Computes: Swish(xW1) * (xW2)
#     """
#     def __init__(self, dim_in, dim_hidden):
#         super().__init__()
#         # Two linear projections: W and V in the paper
#         self.W = nn.Linear(dim_in, dim_hidden, bias=False)
#         self.V = nn.Linear(dim_in, dim_hidden, bias=False)

#     def forward(self, x):
#         """
#         x: (B, T, D)
#         """
#         xW = self.W(x)                  # gate input
#         xV = self.V(x)                  # value input

#         # Swish(xW) = xW * sigmoid(xW)
#         swish_gate = xW * torch.sigmoid(xW)

#         return swish_gate * xV          # elementwise product

class RopeEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
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
        if self.qk_norm:
            self.g = nn.Parameter(torch.ones(1, self.n_heads, 1, 1) * math.log(max_seq_len))

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
            Q = self.g*Q

        if self.rope_embedder is not None:
            Q, K = self.rope_embedder(Q, K)
        
        if self.flash_attention and Q.dtype in (torch.float16, torch.bfloat16) and Q.is_cuda:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
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
                    scale = None,
                    enable_gqa= False if self.n_kv_heads == self.n_heads else True,
                )  # [B, n_heads, Tq, dk]
        Y = Y.transpose(1, 2).contiguous().view(B, T_q, self.d)
        Y = self.W_o(Y)
        return Y

# class AxialAttention(nn.Module):
#     """
#     Axial wrapper around the Attention block.
#     Operates along a chosen 1D axis of a N-D tensor (B, dim1, ..., dimN, D).
#     """
#     def __init__(
#         self,
#         inner_attn: nn.Module,                      # Attention(...)
#     ):
#         super().__init__()
#         self.attn = inner_attn  # expects (B*, T, D) → (B*, T, D)

#     def forward(self,
#                 q: torch.Tensor, 
#                 k: torch.Tensor, 
#                 v:torch.Tensor, 
#                 dim: int, 
#                 mask: Optional[torch.Tensor] = None):
        
#         dims_q = list(q.shape)
#         assert dim > 0 and dim < q.dim(), "Dimension for attention must be between 1 and q.dim()-1"
#         D = dims_q[-1]
#         Tq_k = dims_q[dim]
#         dims_kv = list(k.shape)
#         Tk_k = dims_kv[dim]
#         reordered_q = q.transpose(dim, -2).contiguous().view(-1, Tq_k, D)
#         reordered_k = k.transpose(dim, -2).contiguous().view(-1, Tk_k, D)
#         reordered_v = v.transpose(dim, -2).contiguous().view(-1, Tk_k, D)
#         if mask is not None:
#             assert mask.shape[-2] == q.shape[dim] and mask.shape[-1] == k.shape[dim], "Mask shape does not match the attention dimensions."
#             if mask.dim() ==2:
#                 mask = mask.unsqueeze(0)  # 1xTq xTk
#         Y = self.attn(reordered_q, reordered_k, reordered_v, mask) #...xT_dimxD
#         Y = Y.view((dims_q[:dim]+dims_q[dim+1:-1]+[dims_q[dim], D])).transpose(dim, -2).contiguous() # Verifiy if this line is actually needed
#         return Y

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
        Y_flat = self.attn(q_flat, k_flat, v_flat, mask_flat)  # (B*, Tq, D)

        # Reshape back to original dims
        Y_t = Y_flat.view(*q_t.shape)  # (B, d1, ..., d_{dim-1}, d_{dim+1}, ..., dN, Tq, D)
        Y = Y_t.transpose(dim, -2).contiguous()  # (B, d1, ..., dN, D)
        return Y

# class FeedForwardBlock(nn.Module):
#     """
#     Feed-Forward block with residual connection and RMSNorm.
#     Uses SwiGLU nonlinearity.
#     """
#     def __init__(self, model_dim, dropout_prob=0.1):
#         super().__init__()
#         self.model_dim = model_dim
#         self.swiglu = SwiGLU(model_dim, model_dim*4)  # or hidden_dim
#         self.Dense2 = nn.Linear(model_dim*4, model_dim)
#         self.dropout = nn.Dropout(dropout_prob)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return self.dropout(self.Dense2(self.swiglu(x)))

class LayerType(Enum):
    SPATIAL = "spatial"
    TEMPORAL = "temporal"

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
                causal=False,
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

    def forward(self, x, temporal_mask=None, spatial_mask=None):
        """
        x: (B, T, S, D)
        """

        # pick dimension and mask
        if self.layer_type == LayerType.SPATIAL:
            attn_dim = 2
            mask = spatial_mask
        else:
            attn_dim = 1
            mask = temporal_mask

        # Attention block
        h = self.norm1(x)
        h = self.attn(h, h, h, dim=attn_dim, mask=mask)
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

    def forward(self, x, temporal_mask=None, spatial_mask=None):
        assert x.size(-1) == self.model_dim

        for layer in self.layers:
            x = layer(
                x,
                temporal_mask=temporal_mask,
                spatial_mask=spatial_mask,
            )

        return x

# class EfficientTransformerBlock(nn.Module):
#     """
#     A block composed of spatial and/or temporal axial-attention layers,
#     in the order specified by `layer_types`.

#     Expected input:
#         x: (B, T, S, D)
#             B = batch size
#             T = temporal dimension
#             S = spatial dimension (e.g., patches)
#             D = embedding dimension

#     Args:
#         model_dim: embedding dimension D
#         n_heads: number of attention heads
#         n_kv_heads: number of key/value heads
#         max_seq_len: maximum sequence length for RoPE
#         dropout_prob: dropout probability
#         qk_norm: enable QK normalization
#         layer_types: sequence of LayerType enums defining the order of layers
#     """

#     def __init__(
#         self,
#         model_dim: int,
#         n_heads: int,
#         n_kv_heads: int,
#         max_seq_len: int,
#         dropout_prob: float = 0.0,
#         qk_norm: bool = True,
#         layer_types: Optional[List[LayerType]] = None,
#     ):
#         super().__init__()

#         # Default block: 3 spatial layers + 1 temporal layer
#         if layer_types is None:
#             layer_types = [
#                 LayerType.SPATIAL,
#                 LayerType.SPATIAL,
#                 LayerType.SPATIAL,
#                 LayerType.TEMPORAL,
#             ]

#         # Validate layer types
#         if not all(isinstance(t, LayerType) for t in layer_types):
#             raise TypeError("layer_types must be a list of LayerType enums")

#         self.layer_types = layer_types
#         self.model_dim = model_dim

#         # Shared RoPE embedder
#         self.rope = RopeEmbedding(model_dim//n_heads, max_seq_len)

#         # Factory to build Axial blocks
#         def make_layer():
#             return AxialAttention(
#                 inner_attn=Attention(
#                 model_dim=model_dim,
#                 n_heads=n_heads,
#                 n_kv_heads=n_kv_heads,
#                 causal=False,
#                 dropout_prob=dropout_prob,
#                 qk_norm=qk_norm,
#                 max_seq_len=max_seq_len,
#                 rope_embedder=self.rope)
#             )

#         # One block per layer specification
#         self.attentions = nn.ModuleList([make_layer() for _ in layer_types])
#         self.ffns = nn.ModuleList([FeedForwardSwiGLU(model_dim, None, dropout_prob) for _ in layer_types])
#         self.rms_norm1s = nn.ModuleList([nn.RMSNorm(model_dim) for _ in layer_types])
#         self.rms_norm2s = nn.ModuleList([nn.RMSNorm(model_dim) for _ in layer_types])
#         self.dropout = nn.Dropout(dropout_prob)

#     def forward(
#         self,
#         x: torch.Tensor,
#         temporal_mask: Optional[torch.Tensor] = None,
#         spatial_mask: Optional[torch.Tensor] = None,
#     ) -> torch.Tensor:
#         """
#         Args:
#             x: input tensor of shape (B, T, S, D)
#             temporal_mask: (T, T) or (B, T, T)
#             spatial_mask: (S, S) or (B, S, S)
#         """
#         # Basic shape check
#         assert x.size(-1) == self.model_dim, \
#             f"Expected last dim = {self.model_dim}, got {x.size(-1)}"

#         for i in range(len(self.layer_types)):
#             layer_type = self.layer_types[i]
#             norm1 = self.rms_norm1s[i]
#             norm2 = self.rms_norm2s[i]
#             ffn = self.ffns[i]
#             atn = self.attentions[i]

#             if layer_type == LayerType.SPATIAL:
#                 attn_input = norm1(x)
#                 x = x + self.dropout(atn(attn_input, attn_input, attn_input, dim=2, mask=spatial_mask))
#                 ffn_input = norm2(x)
#                 x = x + self.dropout(ffn(ffn_input))
#             else:
#                 # Temporal layers follow same pattern
#                 attn_input = norm1(x)
#                 x = x + self.dropout(atn(attn_input, attn_input, attn_input, dim=1, mask=temporal_mask))
#                 ffn_input = norm2(x)
#                 x = x + self.dropout(ffn(ffn_input))

#         return x