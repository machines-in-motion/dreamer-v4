import torch
import torch.nn as nn
from .models import TokenizerWrapper, DenoiserWrapper
from omegaconf import DictConfig, OmegaConf


@torch.no_grad()
def load_tokenizer(cfg: DictConfig, device: torch.device, compile=False) -> TokenizerWrapper:
    """
    Load tokenizer (encoder+decoder) and heads from checkpoint.
    """
    tokenizer_wrapper = TokenizerWrapper(cfg).to(device)

    state = torch.load(cfg.tokenizer_ckpt, map_location=device)
    sd = state["model"]

    # Clean FSDP `_orig_mod.` keys
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    tokenizer_wrapper.load_state_dict(clean_sd, strict=True)
    tokenizer_wrapper.eval()

    if compile:
        tokenizer_wrapper = torch.compile(tokenizer_wrapper, mode="max-autotune", fullgraph=False)
    
    return tokenizer_wrapper


@torch.no_grad()
def load_denoiser(cfg: DictConfig, device: torch.device) -> nn.Module:
    """
    Load DreamerV4 denoiser from checkpoint.
    """
    denoiser = DenoiserWrapper(cfg).to(device)
    state = torch.load(cfg.dynamics_ckpt, map_location=device)
    sd = state["model"]
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    denoiser.dyn.load_state_dict(clean_sd, strict=True)
    denoiser.eval()
    return denoiser