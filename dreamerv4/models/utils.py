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

    # Clean FSDP `_orig_mod.` keys
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    if "encoder.temporal_mask_full" in clean_sd:
        del clean_sd["encoder.temporal_mask_full"]
        del clean_sd["decoder.temporal_mask_full"]
    to_delete = []
    for k in clean_sd.keys():
        if k.endswith("cos_emb") or k.endswith("sin_emb"):
            to_delete.append(k)
    for k in to_delete:
        del clean_sd[k]
    
    # tokenizer_wrapper.load_state_dict(clean_sd, strict=True)
    tokenizer_wrapper.load_state_dict(clean_sd, strict=False)
    
    return tokenizer_wrapper


@torch.no_grad()
def load_denoiser(cfg: DictConfig, device: torch.device, max_num_forward_steps=None, model_key="dyn") -> nn.Module:
    """
    Load DreamerV4 denoiser from checkpoint.
    """
    denoiser = DenoiserWrapper(cfg, max_num_forward_steps=max_num_forward_steps).to(device)
    state = torch.load(cfg.dynamics_ckpt, map_location=device)

    sd = state[model_key]
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    # Todo: Fix the key in the final checkpoints before publishing
    clean_sd['diffusion_embedder.embeddings'] = clean_sd['diffuion_embedder.embeddings']
    del clean_sd['diffuion_embedder.embeddings']
    
    # Remove the Rope buffers from the checkpoint
    to_delete = []
    for k in clean_sd.keys():
        if k.endswith("cos_emb") or k.endswith("sin_emb"):
            to_delete.append(k)
    for k in to_delete:
        del clean_sd[k]

    # denoiser.model.load_state_dict(clean_sd, strict=True)
    denoiser.model.load_state_dict(clean_sd, strict=False)
    return denoiser





