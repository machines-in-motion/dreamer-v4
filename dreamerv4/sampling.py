import math 
import numpy as np
import torch

def get_noise_index(noise_level, num_noise_levels):
    return int(np.clip(noise_level, 0., (num_noise_levels-1)/num_noise_levels)*num_noise_levels)

def get_step_index(step_length, num_noise_levels):
    num_steps = int(1./step_length)
    max_pow2 = int(math.log2(num_noise_levels))
    step_index = max_pow2 - int(math.log2(num_steps)) # Convention adopted in my loss
    return step_index

@torch.no_grad
def forward_dynamics_no_cache(
    denoiser,
    ctx_latents,
    actions=None,
    num_pred_steps=1,
    num_diffusion_steps=4,
    context_cond_tau=0.9,
    ):
    
    if actions is not None:
        assert actions.shape[1] == num_pred_steps + ctx_latents.shape[1], 'You should have one action per each context and prediction frames'
        actions = actions.to(device=ctx_latents.device, dtype=ctx_latents.dtype)
    
    B, _, N_lat, D_lat = ctx_latents.shape
    num_context_frames = ctx_latents.shape[1]

    # 1. Initialize pure noise at τ=0
    z = torch.randn(
        B,
        num_pred_steps+num_context_frames,
        N_lat,
        D_lat,
        device=ctx_latents.device,
        dtype=ctx_latents.dtype,
    )
    # Add a slight noise to the context tokens for robustness reasons according to the paper
    latents_cond = ctx_latents.clone()
    latents_cond = (1.0 - context_cond_tau) * torch.randn_like(latents_cond).to(latents_cond.device) + context_cond_tau * latents_cond
    z[:, :num_context_frames, ...] = latents_cond
    
    # Compute the discrete step index  
    step_size = 1.0 / num_diffusion_steps
    denoising_step_index = get_step_index(step_size, denoiser.cfg.denoiser.num_noise_levels)
    step_index_tensor = torch.full(
        (B, num_context_frames+num_pred_steps),
        denoising_step_index,
        dtype=torch.long,
        device=ctx_latents.device,
    )
    
    # Compute the discrete noise level for the context frames with slight noise added on them
    tau_cond_idx = get_noise_index(context_cond_tau, denoiser.cfg.denoiser.num_noise_levels) 

    # Start the shortcut denoising process
    for k in range(num_diffusion_steps):
        
        tau_current = k/num_diffusion_steps # Compute the noise level of the current step (parameter tau in the paper)
        tau_current_idx = get_noise_index(tau_current, denoiser.cfg.denoiser.num_noise_levels)
        tau_index_tensor = torch.full(
            (B, num_context_frames+num_pred_steps),
            tau_current_idx,
            dtype=torch.long,
            device=ctx_latents.device,
        )
        # ste the proper noise level for the context frames 
        tau_index_tensor[:,:num_context_frames] = tau_cond_idx 

        # Denoising
        z_hat = denoiser(
            noisy_z=z,
            action=actions,
            sigma_idx=tau_index_tensor,
            step_idx=step_index_tensor,
        )
        # v = (z_1 - z_τ) / (1 - τ)
        velocity = (z_hat - z) / (1.0 - tau_current)
        # Note: We only apply the denoising process on the future frames
        z[:,num_context_frames:] = z[:,num_context_frames:] + (velocity * step_size)[:,num_context_frames:]
    
    # return torch.cat([latents[:, :num_context_frames, ...], z_hat[:, num_context_frames:]], dim=-3) # z_hat is the output of the last denoiser step which is the predicted clean latents
    # z_hat is the output of the last denoiser step which is the predicted clean latents
    return z_hat[:, num_context_frames:] 

import time
class AutoRegressiveForwardDynamics:
    def __init__(self, 
                 denoiser, 
                 tokenizer, 
                 context_length=32, 
                 max_forward_steps = 5000,
                 context_cond_tau=0.9, 
                 denoising_step_count=4,
                 device="cuda", 
                 dtype=torch.float32):
        
        self.denoiser = denoiser
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.context_length = context_length
        self.context_cond_tau = context_cond_tau
        self.denoising_step_count = denoising_step_count
        self.max_forward_steps = max_forward_steps
        self.current_frame_index = 0        
    
    @torch.no_grad
    def reset(self, imgs_init, actions_init=None):
        
        self.current_frame_index=0
        self.actions_ctx = actions_init.to(device=self.device, dtype=self.dtype) if actions_init is not None else None
        batch_size = imgs_init.shape[0]

        # Encode the context to compute the context tokens
        latents = self.tokenizer.encode(imgs_init)

        self.current_z = latents[:, -1].unsqueeze(1)
        latents_cond = latents.clone()
        self.latents_cond = latents_cond
        #Initialize the tokenizer decoder KV cache
        self.tokenizer.init_cache(batch_size, context_length=self.context_length, device=self.device, dtype=self.dtype)
        self.tokenizer.decode_step(latents_cond,
                                            start_step_idx = 0,
                                            update_cache = True)
        
        #Initialize the dynamics KV cache
        self.denoiser.init_cache(batch_size, context_length=self.context_length, device=self.device, dtype=self.dtype)    
        latents_cond = (1.0 - self.context_cond_tau) * torch.randn_like(latents_cond).to(latents_cond.device) + self.context_cond_tau * latents_cond
        self.cond_tau_idx = get_noise_index(self.context_cond_tau, self.denoiser.cfg.denoiser.num_noise_levels)
        tau_index_tensor = torch.full(
            (batch_size, latents_cond.shape[1]),
            self.cond_tau_idx,
            dtype=torch.long,
            device=self.device,
        )
        step_size = 1./self.denoising_step_count
        denoising_step_index = get_step_index(step_size, self.denoiser.cfg.denoiser.num_noise_levels)
        step_index_tensor = torch.full(
            (batch_size, latents_cond.shape[1]),
            denoising_step_index,
            dtype=torch.long,
            device=self.device,
        )
        self.denoiser.forward_step(
                            noisy_z = latents_cond,
                            sigma_idx=tau_index_tensor,
                            step_idx=step_index_tensor,
                            action=self.actions_ctx,
                            start_step_idx = 0,
                            update_cache = True)
            
        self.current_frame_index += latents_cond.shape[1]

    @torch.no_grad
    def step(self, actions_t=None):
        if actions_t is not None:
            actions_t = actions_t.to(self.device).to(self.dtype)
        B, _, N, D = self.current_z.shape
        z_t = torch.randn(B, 1, N, D, device=self.device, dtype=self.dtype)
        step_length = 1 / self.denoising_step_count
        step_length_idx = get_step_index(step_length, self.denoiser.cfg.denoiser.num_noise_levels)
        
        for i in range(self.denoising_step_count):
            tau_curr = i / self.denoising_step_count
            curr_tau_idx = get_noise_index(tau_curr, self.denoiser.cfg.denoiser.num_noise_levels)
            tau_idxs = torch.full((B, 1), curr_tau_idx, dtype=torch.long, device=self.device)
            step_idxs = torch.full((B, 1), step_length_idx, dtype=torch.long, device=self.device)
            
            pred = self.denoiser.forward_step(
                action=actions_t, noisy_z=z_t, sigma_idx=tau_idxs,
                step_idx=step_idxs, start_step_idx=self.current_frame_index, update_cache=False
            )
            z_t = z_t + (pred - z_t) / max(1.0 - tau_curr, 1e-5) * step_length


        tau_idxs = torch.full((B, 1), self.cond_tau_idx, dtype=torch.long, device=self.device)
        d_min_idx = get_step_index(1./self.denoiser.cfg.denoiser.num_noise_levels, self.denoiser.cfg.denoiser.num_noise_levels)
        step_idxs = torch.full((B, 1), d_min_idx, dtype=torch.long, device=self.device)
        
        seq_cor_tau = torch.full((B, 1, 1, 1), self.context_cond_tau, dtype=torch.bfloat16, device=self.device)
        eps = torch.randn_like(z_t)
        cor_z_t = (1. - seq_cor_tau) * eps + seq_cor_tau * z_t
            
        self.denoiser.forward_step(
            action=actions_t, noisy_z=cor_z_t, sigma_idx=tau_idxs,
            step_idx=step_idxs, start_step_idx=self.current_frame_index, update_cache=True
        )

        imgs_recon = self.tokenizer.decode_step(z_t,
                                                           start_step_idx = self.current_frame_index,
                                                           update_cache = True)
        
        self.current_z = z_t.clone()
        self.current_frame_index += 1
        return imgs_recon[:,0, ...]