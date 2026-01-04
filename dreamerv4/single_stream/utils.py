import torch
import torch.nn as nn
import torch.nn.functional as F

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
