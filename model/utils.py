import torch
import torch.nn as nn
import torch.nn.functional as F

def create_temporal_mask(T, device = "cpu"):
    S = T
    mask = torch.ones(S, S, dtype=torch.bool, device=device)
    for t_q in range(T):
        for t_k in range(T):
            if t_k < t_q:
                mask[t_q, t_k] = False
    return mask

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
    
class TokenMasker(nn.Module):
    def __init__(self, model_dim, n_patches, mask_prob_range=(0.0, 0.9)):
        super().__init__()
        self.mask_prob_range = mask_prob_range
        self.n_patches = n_patches

        # learned mask token for each patch position
        self.learned_masks = nn.Parameter(
            torch.zeros(1, 1, n_patches, model_dim)    # (1,1,N,D)
        )

    def forward(self, x):
        """
        x: (B, T, N, D)
        """
        B, T, N, D = x.shape
        device = x.device
        dtype = x.dtype

        # Sample mask probability p
        p_min, p_max = self.mask_prob_range
        p = torch.empty((), device=device).uniform_(p_min, p_max)

        # mask shape: (B, 1, N, 1)
        mask = torch.rand(B, 1, N, 1, device=device) < p

        # Broadcast learned mask tokens to (B, T, N, D)
        mask_tokens = self.learned_masks.expand(B, T, N, D)

        # Apply mask using torch.where (no in-place ops)
        x = torch.where(mask, mask_tokens, x)
        return x