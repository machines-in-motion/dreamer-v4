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