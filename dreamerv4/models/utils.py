import torch
import torch.nn as nn
from .tokenizer import TokenizerWrapper
from .dynamics import DenoiserWrapper
from omegaconf import DictConfig

@torch.no_grad()
def load_tokenizer(cfg: DictConfig, device: torch.device, max_num_forward_steps=None, model_key="model") -> TokenizerWrapper:
    """
    Load tokenizer (encoder+decoder) and heads from checkpoint.
    """
    tokenizer_wrapper = TokenizerWrapper(cfg, max_num_forward_steps=max_num_forward_steps).to(device)
    state = torch.load(cfg.tokenizer_ckpt, map_location=device)
    sd = state[model_key]
    tokenizer_wrapper.load_state_dict(sd, strict=True)
    return tokenizer_wrapper


@torch.no_grad()
def load_denoiser(cfg: DictConfig, device: torch.device, max_num_forward_steps=None, model_key="model") -> nn.Module:
    """
    Load DreamerV4 denoiser from checkpoint.
    """
    denoiser = DenoiserWrapper(cfg, max_num_forward_steps=max_num_forward_steps).to(device)
    state = torch.load(cfg.dynamics_ckpt, map_location=device)
    sd = state[model_key]
    denoiser.load_state_dict(sd, strict=True)
    return denoiser





