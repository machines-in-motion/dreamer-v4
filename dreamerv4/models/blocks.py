import torch
from enum import Enum
import math
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Tuple, Optional, List
from torch.nn.attention import SDPBackend, sdpa_kernel

def create_temporal_mask(T: int, device: str = "cpu") -> torch.Tensor:
    """
    Standard causal mask for attention:
      mask[q, k] = True  -> allowed (k <= q)
      mask[q, k] = False -> masked out (k > q)
    Shape: (T, T)
    """
    # Lower-triangular (including diagonal) is True
    arange = torch.arange(T, device=device)
    mask = arange.view(T, 1) >= arange.view(1, T)  # (q >= k)
    return mask  # dtype=bool, shape (T, T)

def create_encoder_spatial_mask(N_patch, N_latent, device="cpu"):
    S = N_patch + N_latent
    mask = torch.zeros(S, S, dtype=torch.bool, device=device)
    mask[0:N_patch, 0:N_patch] = True
    mask[N_patch:S, 0:] = True
    return mask

def create_decoder_spatial_mask(N_patch, N_latent, device="cpu"):
    S = N_patch + N_latent
    mask = torch.zeros(S, S, dtype=torch.bool, device=device)
    mask[0:N_patch, 0:] = True
    mask[N_patch:, N_patch:] = True
    return mask

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
        self.initialize_buffer(max_seq_len)
        
    def initialize_buffer(self, max_seq_len):
        thetas = 1.0 / (10000.0 ** (torch.arange(0, self.dim, 2).float() / self.dim))
        thetas = torch.stack([thetas, thetas], dim=0).transpose(0,1).reshape(-1)
        # Pre-compute full table
        thetas_all = torch.stack([thetas*i for i in range(max_seq_len)], dim=0)
        cos_cache = thetas_all.cos() # TxD
        sin_cache = thetas_all.sin() # TxD
        self.register_buffer('cos_emb', cos_cache.unsqueeze(0).unsqueeze(0)) #1x1xTxD
        self.register_buffer('sin_emb', sin_cache.unsqueeze(0).unsqueeze(0)) #1x1xTxD
    
    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, H, T, d = x.shape
        # 1. Slice the buffers
        # Note: slicing creates a 'view' that still points to the DDP-synced buffer memory
        if position_ids is None:
            cos_view = self.cos_emb[:, :,  :T, :]
            sin_view = self.sin_emb[:, :,  :T, :]
        else:
            cos_view = self.cos_emb[:, :, position_ids, :]
            sin_view = self.sin_emb[:, :, position_ids, :]
            if cos_view.dim() < 4:
                cos_view = cos_view.unsqueeze(2)
                sin_view = sin_view.unsqueeze(2)
                
        # 2. CLONE them to break the dependency on the DDP buffer
        # This ensures that if DDP syncs (overwrites) self.cos_emb later, 
        # the graph uses this safe copy.
        cos_emb = cos_view.clone()
        sin_emb = sin_view.clone()
        x1, x2 = x[..., ::2], x[..., 1::2]
        xJ = torch.stack([-x2, x1], dim=-1).reshape_as(x)        
        x_rot = (x * cos_emb) + (xJ * sin_emb)
        return x_rot
    
class KVCache:
    """
    Simple rolling buffer for KV caching.
    """
    def __init__(self, 
                 context_length: int, 
                 batch_size: int, 
                 num_heads: int, 
                 head_dim: int, 
                 device: torch.device, 
                 dtype: torch.dtype):
        
        self.context_length = context_length
        self.curr_len = 0
        # Preallocate buffers [B, H, S, D]
        self.k_cache = torch.zeros(batch_size, num_heads, context_length, head_dim, device=device, dtype=dtype)
        self.v_cache = torch.zeros(batch_size, num_heads, context_length, head_dim, device=device, dtype=dtype)

    def update(self, 
               k_new: torch.Tensor,
               v_new: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        k_new, v_new: [B, H, S_new, D] (usually S_new=1)
        Returns full concatenated history [B, H, S_total, D]
        """
        B, H, S_new, D = k_new.shape
        
        # If buffer is full, roll (shift left)
        if self.curr_len + S_new > self.context_length:
            shift = (self.curr_len + S_new) - self.context_length
            self.k_cache = torch.roll(self.k_cache, shifts=-shift, dims=2)
            self.v_cache = torch.roll(self.v_cache, shifts=-shift, dims=2)
            self.curr_len -= shift

        # Insert new tokens
        self.k_cache[:, :, self.curr_len : self.curr_len + S_new, :] = k_new
        self.v_cache[:, :, self.curr_len : self.curr_len + S_new, :] = v_new

        self.curr_len += S_new
        # Return valid slice
        return self.k_cache[:, :, :self.curr_len, :], self.v_cache[:, :, :self.curr_len, :]
    
    def reset(self):
        self.curr_len = 0
        self.k_cache.zero_()
        self.v_cache.zero_()

    def no_update(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns current valid cache without updating.
        """
        return self.k_cache[:, :, :self.curr_len, :], self.v_cache[:, :, :self.curr_len, :]

class Attention(nn.Module):
    """
    Multi-head (self/cross) attention block with optional QK-Norm and RoPE and GQA.
    """
    def __init__(self, 
                 model_dim, 
                 n_heads, 
                 n_kv_heads=None, 
                 dropout_prob = 0, 
                 qk_norm=True, 
                 max_seq_len=128, 
                 rope_embedder = None, 
                 flash_attention=False):
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
                 kv_cache: Optional[KVCache] = None,
                 update_cache: bool = True,
                 kv_position_ids: Optional[torch.Tensor] = None,
                 q_position_ids: Optional[torch.Tensor] = None)-> Tuple[torch.Tensor, torch.Tensor]:
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
        # During inference where q is only for one frame, is_causal=True means your only query is at index i=0,
        #  so it can attend to only key 0 — not all past keys. Hence, we set is_causal=False in that case to allow attending to all cached keys.     
        if q.shape[-2] == 1:
             is_causal = False
        else:
             is_causal = causal
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
            # If caching, we assume q is the *new* query (Sq=1) 
            # and k,v is the *new* KV (Skv=1)
            
            # Apply RoPE to the NEW k only
            # NOTE: RoPE needs the correct absolute position index for the new token.
            # The caller must pass correct kv_position_ids for the current step.
            if self.rope_embedder is not None:
                K = self.rope_embedder(K, position_ids=kv_position_ids)
            if update_cache:
                # Standard behavior: Add new keys to cache
                K, V = kv_cache.update(K, V)

            else: # During the diffusion steps, no temporal updates are made so the cache remains the same
                # Read-only behavior: 
                # We need to attend to [Cache, New_K].
                # So we manually concat without updating the ring buffer.
                # 1. Retrieve existing cache
                # cache_k: [B, H, S_curr, D]
                cache_k , cache_v = kv_cache.no_update()
                # 2. Concat with current step's K/V
                # k: [B, H, 1, D] -> [B, H, S_curr + 1, D]
                K = torch.cat([cache_k, K], dim=2)
                V = torch.cat([cache_v, V], dim=2)

            # Apply RoPE to the NEW q only
            if self.rope_embedder is not None:
                Q = self.rope_embedder(Q, position_ids=q_position_ids) 
                
        else:
            if self.rope_embedder is not None:
                # Standard behavior (uncached)
                Q = self.rope_embedder(Q, position_ids=q_position_ids)
                K = self.rope_embedder(K, position_ids=kv_position_ids)
                # Q, K = self.rope_embedder(Q, K)

        if is_causal:
            assert mask is None, "Causal flag should not be set when mask is provided."
        
        if self.flash_attention and Q.dtype in (torch.float16, torch.bfloat16) and Q.is_cuda:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                Y = F.scaled_dot_product_attention(
                    Q, K, V,
                    attn_mask=mask, 
                    dropout_p=self.dropout_prob if self.training else 0.0,
                    is_causal=is_causal,
                    scale = None,
                    enable_gqa= False if self.n_kv_heads == self.n_heads else True,
                )  # [B, n_heads, Tq, dk]
        else:
                Y = F.scaled_dot_product_attention(
                    Q, K, V,
                    attn_mask=mask, 
                    dropout_p=self.dropout_prob if self.training else 0.0,
                    is_causal=is_causal,
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
        update_cache: bool = True,
        kv_position_ids: Optional[torch.Tensor] = None,
        q_position_ids: Optional[torch.Tensor] = None,
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
        Y_flat = self.attn(q_flat, 
                           k_flat, 
                           v_flat, 
                           mask_flat, 
                           causal=causal, 
                           kv_cache=kv_cache,
                           update_cache=update_cache,
                           kv_position_ids=kv_position_ids,
                           q_position_ids=q_position_ids)  # (B*, Tq, D)

        # Reshape back to original dims
        Y_t = Y_flat.view(*q_t.shape)  # (B, d1, ..., d_{dim-1}, d_{dim+1}, ..., dN, Tq, D)
        Y = Y_t.transpose(dim, -2).contiguous()  # (B, d1, ..., dN, D)
        return Y

class EfficientTransformerLayer(nn.Module):
    """
    A single transformer layer as shown as one of the four layers in Fig.x of the paper.:
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
        is_causal: bool = True,
        rope_embedder: Optional[RopeEmbedding] = None,
    ):
        super().__init__()
        assert isinstance(layer_type, LayerType)

        self.layer_type = layer_type
        self.model_dim = model_dim
        self.is_causal = is_causal
        # Use shared RoPE if available
        self.rope = rope_embedder

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
                spatial_mask=None,
                kv_cache: Optional[KVCache] = None,
                update_cache: bool = True,
                position_ids: Optional[torch.Tensor] = None):
        """
        compute one transformer layer block as:
        x = drop_out(Attn(RMSNorm(x))) + x
        x = x + drop_out(FFN(RMSNorm(x)))
        Args:
            x: (B, T, S, D) input tensor
            spatial_mask: (S, S) or broadcastable mask for spatial attention
            layer_cache: optional KVCache for caching in temporal attention
        Returns:
            x: (B, T, S, D) output tensor
        """
        # Attention block
        h = self.norm1(x)
        if self.layer_type == LayerType.TEMPORAL:
            h = self.attn(h, h, h, dim=1, 
                          causal=self.is_causal, 
                          kv_cache = kv_cache,
                          update_cache=update_cache,
                          kv_position_ids=position_ids,
                          q_position_ids=position_ids,
                          ) # Temporal dimension has caching
        else:
            h = self.attn(h, h, h, dim=2, mask=spatial_mask) # Spatial dimension has no caching (check here)
        x = x + self.dropout(h)

        # FFN block
        h = self.norm2(x)
        h = self.ffn(h)
        x = x + self.dropout(h)

        return x

class EfficientTransformerBlock(nn.Module):
    """
    One stack of 3 spatial + 1 temporal EfficientTransformerLayer as shown in Fig.x of the paper.
    Allows custom layer layouts via layer_types argument.
    """

    def __init__(
        self,
        model_dim: int,
        n_heads: int,
        n_kv_heads: int,
        # max_seq_len: int,
        temporal_dim_max_seq_len: int,
        modality_dim_max_seq_len: int,
        dropout_prob: float = 0.0,
        qk_norm: bool = True,
        is_causal: bool = True,
        layer_types: Optional[List[LayerType]] = None,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.modality_dim_max_seq_len = modality_dim_max_seq_len
        self.temporal_dim_max_seq_len = temporal_dim_max_seq_len
        self.is_causal = is_causal
        self.caches: Optional[List[KVCache]] = None  # to be initialized later
        # Default block layout
        if layer_types is None:
            layer_types = [
                LayerType.SPATIAL,
                LayerType.SPATIAL,
                LayerType.SPATIAL,
                LayerType.TEMPORAL,
            ]

        self.layers = nn.ModuleList()
        for layer in layer_types:
            if layer == LayerType.TEMPORAL:
                self.layers.append(
                    EfficientTransformerLayer(
                        model_dim=model_dim,
                        n_heads=n_heads,
                        n_kv_heads=n_kv_heads,
                        max_seq_len=temporal_dim_max_seq_len,
                        layer_type=layer,
                        dropout_prob=dropout_prob,
                        qk_norm=qk_norm,
                        is_causal=is_causal,
                        rope_embedder=RopeEmbedding(model_dim // n_heads, temporal_dim_max_seq_len),
                    )
                )
            else:
                self.layers.append(
                    EfficientTransformerLayer(
                        model_dim=model_dim,
                        n_heads=n_heads,
                        n_kv_heads=n_kv_heads,
                        max_seq_len=modality_dim_max_seq_len,
                        layer_type=layer,
                        dropout_prob=dropout_prob,
                        qk_norm=qk_norm,
                        is_causal=False,
                        rope_embedder=RopeEmbedding(model_dim // n_heads, modality_dim_max_seq_len),
                    )
                )

    def forward(self, x, spatial_mask=None):
        assert x.size(-1) == self.model_dim
        for i, layer in enumerate(self.layers):
            x = layer(x, spatial_mask=spatial_mask)
        return x
    
    def forward_step(self, 
                     x: torch.Tensor,
                     spatial_mask: Optional[torch.Tensor],
                     start_step_idx: int,
                     update_cache: bool = True):
        assert self.is_causal, "KV caching only valid for causal models."
        assert self.caches is not None, "Caches not initialized. Call init_cache() before forward_step."
        B, T, _, _ = x.shape
        
        pos_ids = torch.arange(start_step_idx, start_step_idx + T, device=x.device, dtype=torch.long)
        # pos_ids = pos_ids.unsqueeze(0)
        # 3. Apply Layers

        for i, layer in enumerate(self.layers):
            if layer.layer_type == LayerType.TEMPORAL:
                x = layer(x, 
                          spatial_mask=spatial_mask,
                          kv_cache=self.caches[i], 
                          position_ids=pos_ids, 
                          update_cache=update_cache)
            else:
                x = layer(x, spatial_mask=spatial_mask)

        return x
    
    def init_cache(self, batch_size: int, device: torch.device, context_length: int, dtype: torch.dtype):
        assert self.is_causal, "KV caching only valid for causal models."
        """Initializes KV caches for all temporal layers."""
        self.caches = []
        for layer in self.layers:
            if layer.layer_type == LayerType.TEMPORAL:
                cache = KVCache(
                    context_length=context_length,
                    batch_size=int(batch_size*self.modality_dim_max_seq_len),
                    num_heads=int(self.n_kv_heads),
                    head_dim=int(self.model_dim//self.n_heads),
                    device=device,
                    dtype=dtype
                )
                self.caches.append(cache)
            else:
                self.caches.append(None)

    

