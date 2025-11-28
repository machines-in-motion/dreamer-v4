import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from model.blocks import EfficientTransformerBlock
from dataclasses import dataclass

@dataclass
class DreamerV4DenoiserCfg:
    num_action_tokens: int
    num_latent_tokens: int
    num_register_tokens: int
    max_context_length: int
    model_dim: int
    latent_dim: int
    n_layers: int 
    n_heads: int
    n_kv_heads: Optional[int] = None
    dropout_prob: float = 0.0
    qk_norm: bool = True
    K_max: int=32
    
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
   
class DreamerV4Denoiser(nn.Module):
    def __init__(self, cfg: DreamerV4DenoiserCfg):
        super().__init__()
        self.cfg = cfg
        self.diffuion_embedder = DiscreteEmbedder(cfg.K_max, cfg.model_dim//2)    
        self.shortcut_embedder = DiscreteEmbedder(torch.log2(torch.tensor(cfg.K_max)).to(torch.long)+1, cfg.model_dim//2)    
        self.register_tokens = nn.Parameter(torch.zeros(1, 1, cfg.num_register_tokens, cfg.model_dim)) # 1 x 1 x N_reg x D
        self.action_tokens = nn.Parameter(torch.zeros(1, 1, cfg.num_action_tokens, cfg.model_dim))     # 1 x 1 x N_action x D
        self.max_seq_len = cfg.max_context_length*1024
        self.layers = nn.ModuleList([
            EfficientTransformerBlock(
                model_dim=cfg.model_dim,
                n_heads=cfg.n_heads,
                n_kv_heads=cfg.n_kv_heads,
                dropout_prob=cfg.dropout_prob,
                qk_norm=cfg.qk_norm,
                max_seq_len=self.max_seq_len,
            )
            for _ in range(cfg.n_layers)
        ])
        self.latent_projector = nn.Linear(cfg.latent_dim, cfg.model_dim, bias=False)
        self.output_projector = nn.Linear(cfg.model_dim, cfg.latent_dim, bias=False)

    def forward(self, 
                latent_tokens: torch.Tensor,    # BxTxN_latentxD
                diffusion_step: torch.Tensor,   # BxT --> Different noise values for each frame
                shortcut_length: torch.Tensor,
                act_token: Optional[torch.Tensor]=None): # B   --> The same denoising shortcut length for all frames
        
        B, T, _ , D = latent_tokens.shape
        diff_step_token = self.diffuion_embedder(diffusion_step).unsqueeze(-2) # BxTx1xD
        shortcut_token  = self.shortcut_embedder(shortcut_length).unsqueeze(-2) # BxTx1xD
        diff_control_token = torch.cat([shortcut_token, diff_step_token], dim=-1)

        reg_tokens = self.register_tokens.expand(B, T, -1, -1)
        act_learned_tokens = self.action_tokens.expand(B, T, -1, -1)
        if act_token is not None:
            act_learned_tokens = act_learned_tokens + act_token
        #Formulate the input to the world model as BxTx|latent_tokens:register_tokens:concat(shortcut_token, diffusion_tokens): action_tokens|
        x = torch.cat([self.latent_projector(latent_tokens), reg_tokens, diff_control_token, act_learned_tokens], dim=-2)
        for layer in self.layers:
            x = layer(x)
        x = self.output_projector(x)
        return x[:, :, :self.cfg.num_latent_tokens, :] # Return the denoising output for the latents
    

class ForwardDiffusionWithShortcut(nn.Module):
    def __init__(self, K_max=128):
        super().__init__()
        self.max_pow2 = torch.floor(torch.log2(torch.tensor(K_max)))
        self.K_max = 2**self.max_pow2
        self.d_min = 1./self.K_max

    def sample_step_noise(self, batch_size, seq_len, device):
        step_idx = torch.randint(0, int(self.max_pow2+1), (batch_size, seq_len))
        # half_step_idx = step_idx+1
        step = 1/2**step_idx
        # half_step = step/2 
        noise_idx = torch.floor(torch.rand(*step.shape)*(step_idx**2).to(torch.float32))# <- Verify 
        # noise_plus_halfstep_idx = noise_idx+half_step/self.d_min
        noise = noise_idx*step
        # noise_plus_halfstep = noise_plus_halfstep_idx*self.d_min
        return dict(d=step.to(device), d_discrete=step_idx.to(device), tau=noise.to(device), tau_discrite=noise_idx.to(device))
    
    def forward(self, x):
        B, T, _, _ = x.shape
        x0 = torch.randn_like(x).to(x.device)
        diff_params = self.sample_step_noise(B, T, x.device) # BxT, BxT
        tau = diff_params['tau']
        tau = tau.unsqueeze(-1).unsqueeze(-1)
        x_tau = (1.-tau)*x0 + tau*x

        # The following converstions are not right
        d_disc = torch.log2(diff_params['d']/self.d_min)
        tau_disc = (diff_params['tau'])/self.d_min
        half_d_disc = d_disc+1
        tau_plus_half_step_disc = (diff_params['tau']+diff_params['d']/2)/self.d_min
        # shortcut_mask = diff_params['d'] > self.d_min
        # no_shortcut_mask = diff_params['d'] == self.d_min
        # with torch.no_grad():
        #     b_prime= self.f_theta(x_tau, tau_disc, half_d_disc)
        #     x_prime = x_tau + b_prime*diff_params['d'].unsqueeze(-1).unsqueeze(-1)/2
        #     b_dprime = self.f_theta(x_prime, tau_plus_half_step_disc, half_d_disc)
            
        # self.target[no_shortcut_mask] = (x-self.x0)[no_shortcut_mask]
        # self.target[shortcut_mask] = ((b_dprime+b_prime)/2).detach()[shortcut_mask]
        return dict(x_tau=x_tau, 
                    tau_d=tau_disc, 
                    step_d=d_disc,
                    tau_plus_half_d = tau_plus_half_step_disc, 
                    half_step_d = half_d_disc, 
                    tau = diff_params['tau'], 
                    step = diff_params['d']
                    )