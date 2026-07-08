# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Unofficial PyTorch implementation of the **DreamerV4 world model** (from [Training Agents Inside of Scalable World Models](https://arxiv.org/abs/2509.24527)). It implements the two-stage world model only — a **causal tokenizer** (image autoencoder) and an **action-conditioned dynamics denoiser** — plus real-time interactive rollout with KV-caching and distributed training scripts. There is no agent/RL/actor-critic component here.

## Commands

```bash
# Install
conda env create -f environment.yml && conda activate dreamer-v4 && pip install -e .

# Train the tokenizer (FSDP). --nproc_per_node = number of GPUs.
torchrun --standalone --nproc_per_node=1 scripts/train_tokenizer.py \
  --config-path scripts/config --config-name tokenizer/soar.yaml \
  dataset.data_dir=/path/to/soar_data_sharded train.batch_per_gpu=1 train.grad_accum_steps=1

# Train the dynamics denoiser (DDP). Requires a trained tokenizer checkpoint (set via config tokenizer_ckpt).
torchrun --standalone --nproc_per_node=1 scripts/train_dynamics.py \
  --config-path scripts/config --config-name dynamics/soar-small.yaml \
  dataset.data_dir=/path/to/soar_data_sharded train.batch_per_gpu=1 train.accum_grad_steps=1

# Interactive real-time rollout (OpenCV window; add --xbox for joystick, requires checkpoints)
python scripts/play-soar.py  --config-name dynamics/soar-small.yaml tokenizer_ckpt=... dynamics_ckpt=... dataset.data_dir=...
python scripts/play-pushT.py tokenizer_ckpt=... dynamics_ckpt=... dataset.data_dir=...

# Dataset prep: convert HuggingFace/LeRobot or RLDS datasets into the sharded HDF5 format this repo expects
python scripts/preprocessing/huggingface_to_hdf5.py --repo_id ... --output_dir ... [see README for flags]
python scripts/preprocessing/rlds_to_hdf5.py --rlds_dir ... --output_dir ...

# Inspect a prepared dataset (episode-length stats, window counts)
python -m dreamerv4.datasets --data_dir /path/to/sharded --window_size 96
```

There is **no test suite, linter, or CI config** in this repo. Overrides use Hydra's `key=value` CLI syntax (dotted paths, e.g. `train.lr=1e-4`, `denoiser.context_length=32`).

## Architecture

### Two-stage pipeline
1. **Tokenizer** (`dreamerv4/models/tokenizer.py`, `TokenizerWrapper`): `patchify → TokenMasker (MAE-style dropout) → CausalTokenizerEncoder → latents z → CausalTokenizerDecoder → TokensToImageHead`. Images enter in `[0,1]` and are internally rescaled to `[-1,1]`. Latents are `(B, T, num_latent_tokens, latent_dim)`, e.g. `256 × 32`. Trained standalone with **MSE + 0.2·LPIPS**, each term RMS-normalized (`RMSLossScaler`).
2. **Dynamics / Denoiser** (`dreamerv4/models/dynamics.py`, `DenoiserWrapper` / `DreamerV4Denoiser`): a shortcut-diffusion denoiser operating **in the tokenizer's latent space**, conditioned on actions. Its per-frame token sequence is `[latent_tokens | register_tokens | diff_control_token | action_tokens]`. During dynamics training the tokenizer is **frozen** (`eval()`, `requires_grad_(False)`) and used only to encode clean latents.

### Shared transformer backbone (`dreamerv4/models/blocks.py`)
Both models are stacks of `EfficientTransformerBlock`, each of which is **3 spatial + 1 temporal** `EfficientTransformerLayer` using **axial attention** (`AxialAttention` wraps a single `Attention`):
- **Spatial layers** attend over the token axis (`dim=2`), non-causal, use a static `spatial_mask`, **no KV cache**.
- **Temporal layers** attend over the time axis (`dim=1`), **causal**, windowed by `context_length`, and are the only layers that use **KV caching**.
- Attention includes RoPE (`RopeEmbedding`), optional QK-norm, GQA (`n_kv_heads`), SwiGLU FFN, RMSNorm.

### Real-time inference & KV cache
`init_cache()` + `forward_step()` (on blocks, tokenizer decoder, and denoiser) enable autoregressive real-time rollout. `KVCache` is a rolling buffer of length `context_length`. Key subtlety: during the inner diffusion sub-steps the temporal position does **not** advance, so those calls pass `update_cache=False` (read-only concat); only when a frame is committed is `update_cache=True`. `AutoRegressiveForwardDynamics` (`dreamerv4/sampling.py`) is the driver: `reset(init_imgs, init_actions)` primes the caches, then `step(action)` emits one predicted frame. `forward_dynamics_no_cache()` in the same file is the non-cached (slower, reference) equivalent.

### Diffusion / loss (`dreamerv4/loss.py`)
`ForwardDiffusionWithShortcut` implements the dyadic shortcut schedule (`num_noise_levels` must be a power of 2). `compute_bootstrap_diffusion_loss` returns a **flow loss** (fine step, `d = d_min`) and a **bootstrap/shortcut loss** (larger steps), combined during training. Index conventions (`tau_index`, `step_index`) are documented in the module docstrings; helpers `get_noise_index`/`get_step_index` in `sampling.py` map continuous τ / step-size to these indices for inference.

### Hydra config composition (`scripts/config/`)
Configs are composed, not flat. A **`dynamics/*.yaml`** pulls in a **`tokenizer/*.yaml`** via `defaults`, which in turn pulls in a **`dataset/*.yaml`**. So a dynamics run's config contains `denoiser`, `tokenizer`, `dataset`, and `train` sections together. The entry-point configs (`tokenizer/*`, `dynamics/*`, `video/*`) use `# @package _global_`, while `dataset/*` configs are namespaced under `dataset` (accessed as `cfg.dataset.*`). Note: per the README quickstart the **PushT world model uses the SOAR tokenizer checkpoint** — the `tokenizer/pushT` and `tokenizer/soar` configs share the same architecture, so a SOAR-trained tokenizer loads into either. `max_sequence_length` sizes the RoPE/temporal tables; for long rollouts, inference passes a larger `max_num_forward_steps` (e.g. `NUM_FORWARD_STEPS` in the play scripts).

### Data (`dreamerv4/datasets.py`)
`ShardedHDF5Dataset` is the training dataset: a directory of `shard_XXXX.h5` + `metadata.json`, where each shard holds `images (num_episodes, max_len, H, W, C) uint8`, `actions`, and `episode_lengths`. It builds sliding windows of `window_size` and does a **reproducible episode-level train/test split** (`split_seed`, `train_fraction`) so no episode leaks across splits. `__getitem__` returns `{'image': (T,C,H,W) float in [0,1], 'action': (T,A)}`. For seeding initial frames, `play-soar.py` samples from a `ShardedHDF5Dataset` window, while `play-pushT.py` uses `PushTDataset` (a per-episode HDF5 reader).

## Conventions & gotchas
- Training runs in **bfloat16** under `torch.autocast`. Tokenizer trains under **FSDP** (`transformer_auto_wrap_policy` on `EfficientTransformerLayer`); dynamics trains under **DDP**. Both optionally wrap with `torch.compile` when `train.use_compile` is set.
- Checkpoints are dicts with a `"model"` key (see `load_tokenizer`/`load_denoiser` in `dreamerv4/models/utils.py`); resume state also stores optimizer, scheduler, epoch, `global_update`, and the W&B run id.
- `num_modality_tokens` in a tokenizer config must equal `(resolution / patch_size)²` (e.g. `256/16 = 16`, `16² = 256`).
- Play scripts (`scripts/play-*.py`) have **hardcoded constants** near the top (`NUM_INIT_FRAMES`, `CONTEXT_LEN`, `NUM_FORWARD_STEPS`, and an episode path) that are not yet Hydra-exposed — edit them there, not via config.
- `create_distributed_dataloader` is duplicated in both `dreamerv4/datasets.py` and `scripts/train_tokenizer.py`; the dynamics script imports the former.
- Logging goes to both TensorBoard and W&B (rank 0 only), synced via `sync_tensorboard=True`.
