import torch
import torch.nn as nn
import torch.nn.functional as F

# def create_temporal_mask(T, device = "cpu"):
#     S = T
#     mask = torch.ones(S, S, dtype=torch.bool, device=device)
#     for t_q in range(T):
#         for t_k in range(T):
#             if t_k < t_q:
#                 mask[t_q, t_k] = False
#     return mask

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
    
# class TokenMasker(nn.Module):
#     def __init__(self, model_dim, n_patches, mask_prob_range=(0.0, 0.9)):
#         super().__init__()
#         self.mask_prob_range = mask_prob_range
#         self.n_patches = n_patches

#         # learned mask token for each patch position
#         self.learned_masks = nn.Parameter(
#             torch.zeros(1, 1, n_patches, model_dim)    # (1,1,N,D)
#         )

#     def forward(self, x):
#         """
#         x: (B, T, N, D)
#         """
#         B, T, N, D = x.shape
#         device = x.device
#         dtype = x.dtype

#         # Sample mask probability p
#         p_min, p_max = self.mask_prob_range
#         p = torch.empty((), device=device).uniform_(p_min, p_max)

#         # mask shape: (B, 1, N, 1)
#         mask = torch.rand(B, T, N, 1, device=device) < p

#         # Broadcast learned mask tokens to (B, T, N, D)
#         mask_tokens = self.learned_masks.expand(B, T, N, D)

#         # Apply mask using torch.where (no in-place ops)
#         x = torch.where(mask, mask_tokens, x)
#         return x

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
                mask = rand < drop_p.unsqueeze(-1)  # [B, T, N]

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
