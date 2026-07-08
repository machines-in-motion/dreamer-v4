import torch
import torch.nn as nn
from .tokenizer import TokenizerWrapper
from .dynamics import DenoiserWrapper
from omegaconf import DictConfig


def _normalize_ckpt_key(k: str) -> str:
    """Normalize a legacy checkpoint key to the current module naming.

    Handles checkpoints saved:
      * from a ``torch.compile``d module (``_orig_mod.`` prefix),
      * from a DDP-wrapped module (``module.`` prefix),
      * before the ``diffuion_embedder`` -> ``diffusion_embedder`` typo fix.
    """
    while k.startswith("_orig_mod."):
        k = k[len("_orig_mod."):]
    while k.startswith("module."):
        k = k[len("module."):]
    k = k.replace("diffuion_embedder", "diffusion_embedder")
    return k


def _load_state_dict_compat(model: nn.Module, sd: dict, label: str = "model") -> None:
    """Load ``sd`` into ``model``, tolerating legacy key layouts.

    Fast path: a plain ``strict=True`` load (checkpoints produced by the
    current code). Fallback: normalize keys (strip compile/DDP prefixes, fix
    the historical embedder typo) and, if needed, add the ``model.`` wrapper
    prefix, matching each tensor by name AND shape. Buffers that the current
    code recomputes at init (RoPE tables, cached masks) are simply dropped.

    Raises if any *learnable* parameter of ``model`` cannot be matched, so a
    genuinely mismatched checkpoint never loads silently as a partial model.
    """
    try:
        model.load_state_dict(sd, strict=True)
        return
    except RuntimeError:
        pass

    target = model.state_dict()
    target_params = {n for n, _ in model.named_parameters()}

    remapped: dict = {}
    for k, v in sd.items():
        nk = _normalize_ckpt_key(k)
        for cand in (nk, "model." + nk):
            if (cand in target and cand not in remapped
                    and tuple(v.shape) == tuple(target[cand].shape)):
                remapped[cand] = v
                break

    missing = sorted(n for n in target_params if n not in remapped)
    if missing:
        raise RuntimeError(
            f"{label}: checkpoint could not supply {len(missing)} learnable "
            f"parameter(s), e.g. {missing[:5]}. Refusing to load a partial model."
        )

    incompatible = model.load_state_dict(remapped, strict=False)
    leftover = [k for k in incompatible.missing_keys if k in target_params]
    if leftover:
        raise RuntimeError(
            f"{label}: learnable keys left uninitialized after load: {leftover[:5]}"
        )
    print(f"[load] {label}: remapped legacy checkpoint "
          f"({len(remapped)}/{len(target)} tensors matched).")


def _extract_state_dict(state: dict, model_key: str) -> dict:
    """Pull the model state dict out of a checkpoint container."""
    if isinstance(state, dict) and model_key in state:
        return state[model_key]
    return state


@torch.no_grad()
def load_tokenizer(cfg: DictConfig, device: torch.device, max_num_forward_steps=None, model_key="model") -> TokenizerWrapper:
    """
    Load tokenizer (encoder+decoder) and heads from checkpoint.
    """
    tokenizer_wrapper = TokenizerWrapper(cfg, max_num_forward_steps=max_num_forward_steps).to(device)
    state = torch.load(cfg.tokenizer_ckpt, map_location=device)
    sd = _extract_state_dict(state, model_key)
    _load_state_dict_compat(tokenizer_wrapper, sd, label="tokenizer")
    return tokenizer_wrapper


@torch.no_grad()
def load_denoiser(cfg: DictConfig, device: torch.device, max_num_forward_steps=None, model_key="model") -> nn.Module:
    """
    Load DreamerV4 denoiser from checkpoint.
    """
    denoiser = DenoiserWrapper(cfg, max_num_forward_steps=max_num_forward_steps).to(device)
    state = torch.load(cfg.dynamics_ckpt, map_location=device)
    sd = _extract_state_dict(state, model_key)
    _load_state_dict_compat(denoiser, sd, label="denoiser")
    return denoiser
