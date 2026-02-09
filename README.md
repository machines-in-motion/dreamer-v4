# Dreamer-V4 (PyTorch)

Unofficial PyTorch implementation of the Dreamer-V4 world model component (**causal tokenizer** and **interactive dynamics (denoiser)**) of [Training Agents Inside of Scalable World Models](https://arxiv.org/abs/2509.24527) paper. We train two scales of this world model on representative real-world robotic datasets and provide the real-time interactive environments with KV-caching implementation and training scripts for training on your own Huggingface/RLDS datasets. 

<p align="center">
  <img src="docs/assets/pushT-small-1.gif" width="32%" />
  <img src="docs/assets/soar-small-1.gif" width="32%" />
  <img src="docs/assets/soar-big-1.gif" width="32%" />
</p>
<p align="center">
  <img src="docs/assets/soar-big-1.gif" width="32%" />
  <img src="docs/assets/soar-big-2.gif" width="32%" />
  <img src="docs/assets/soar-big-3.gif" width="32%" />
</p>

---

## Installation

```bash
conda env create -f environment.yml
conda activate dreamer-v4
pip install -e .
```

## Quickstart: run the interactive demos

First download our [checkpoints](https://huggingface.co/Rooholla/dreamer-v4) and downscaled datasets:

| Checkpoint | Parameters | Dataset | Config | Resolution | Download |
|:-----------|:----------:|:-----------------:|:----------------:|:----------------:|:---------------:|
| Tokenizer           | 400M | [SOAR](https://rail.eecs.berkeley.edu/datasets/soar_release/) | [config](scripts/config/tokenizer/soar.yaml) |  256×256 | [📦](https://huggingface.co/Rooholla/dreamer-v4/resolve/main/tokenizer-soar-400M.pt?download=true) |
| Dynamics      | 110M | [MiM PushT](https://huggingface.co/datasets/Rooholla/MiM-PushT) | [config](scripts/config/dynamics/pushT.yaml)  |  256×256 | [📦](https://huggingface.co/Rooholla/dreamer-v4/resolve/main/dynamics-pushT-110M.pt?download=true) |
| Dynamics | 110M | [SOAR](https://rail.eecs.berkeley.edu/datasets/soar_release/)     | [config](scripts/config/dynamics/soar-small.yaml) | 256×256 | [📦](https://huggingface.co/Rooholla/dreamer-v4/resolve/main/dynamicsl-soar-110M.pt?download=true) |
| Dynamics | 1.5B | [SOAR](https://rail.eecs.berkeley.edu/datasets/soar_release/)     | [config](scripts/config/dynamics/soar-large.yaml)  | 128×128 | Soon! |


Now you can run the interactive world environments as follows. We sddume an X-Box USB joystick controller is connected. In this case, add the `--xbox` argument to the play scripts:

### SOAR World Model 
```bash
python scripts/play-soar.py \ 
  --config-name dynamics/soar-small.yaml \ # soar-large.yaml for the large model
  tokenizer_ckpt=path/to/tokenizer.pt \
  dynamics_ckpt=path/to/dynamics.pt \
  dataset.data_dir=/path/to/soar_data_sharded
```
### PushT World Model 
```
# PushT
python scripts/play-pushT.py \
  tokenizer_ckpt=path/to/soar/tokenizer.pt \ # We use the SOAR tokenizer for PushT world model
  dynamics_ckpt=path/to/dynamics.pt \
  dataset.data_dir=/path/to/pusht_data_sharded
```
Running above will pops up an openCV windown showing the world-model predicted frames as you command it with your joystick. 

**Requirement**: The small world models (110M parameters) run in real-time on a single 4090 GPU and require only 8GB of VRAM. The information for the large model will be provided upon its release in the future. 

**Note:** The demo uses `NUM_INIT_FRAMES=8`, `CONTEXT_LEN=32`, `NUM_FORWARD_STEPS=1000` in the script. We’ll expose these as Hydra overrides in a follow-up cleanup.

**ToDo**: We will release a VR control mode using a Meta Quest3 for this environment soon.

## Training
### Dataset Preparation
Our scripts work on a custom HDF5-based dataset that need to be prepared beforehand. Our dataset is defined as a directory with the following files:
```
<output_dir>/
  metadata.json
  shard_0000.h5
  shard_0001.h5
  ...
```
Each shard file contains:
- images: `(num_episodes, max_len, H, W, C)` uint8
- actions: `(num_episodes, max_len, action_dim)` float32
- episode_lengths: `(num_episodes,)` int32
- HDF5 attrs: `num_episodes, max_length`

Training scripts support distributed execution and can be launched with torchrun.


We include preprocessing scripts to convert common dataset formats into the format this repo expects:

#### From HuggingFace Datasets
```bash
python scripts/preprocessing/huggingface_to_hdf5.py \
  --repo_id <HF_REPO_ID> \           # HuggingFace / LeRobot dataset repo_id (e.g., "lerobot/soar")
  --output_dir <OUTPUT_DIR> \        # Output folder where shards + metadata.json will be written
  --local_dir <LOCAL_CACHE_DIR> \    # Optional: use a local dataset cache instead of downloading
  --episodes_per_shard 500 \         # Episodes per HDF5 file (I/O parallelism vs overhead tradeoff)
  --target_fps 10 \                  # Subsample to this FPS (integer subsampling in current script)
  --width 256 --height 256 \         # Resize frames to (height, width)
  --min_episode_length 1000 \        # Drop episodes shorter than this threshold
  --rectangular \                    # Pad to square before resizing (note: flag behavior should be verified in script)
  --relative_actions \               # Store action deltas (a_t - a_{t-1}) instead of absolute actions
  --image_key observation.images.camera_0 \  # Key for image in the LeRobot sample dict
  --action_key action                # Key for action in the LeRobot sample dict
```
**Note**: `--relative_actions stores` $a_t - a_{t-1}$ with first action delta set to $0$ per episode.

#### From RLDS Datasets
```bash
python scripts/preprocessing/rlds_to_hdf5.py \
  --rlds_dir <RLDS_DIR> \            # Path to the RLDS dataset directory (tfds.builder_from_directory reads this)
  --output_dir <OUTPUT_DIR> \        # Output folder where shard_XXXX.h5 and metadata.json will be written
  --split success+failure \          # Which RLDS split to convert (example: success+failure)
  --episodes_per_shard 500           # Number of episodes stored per shard file (controls shard size / I/O parallelism)
```

### Tokenizer Training
```
torchrun --standalone --nproc_per_node=1 scripts/train_tokenizer.py \
  --config-path scripts/config --config-name tokenizer/soar.yaml \
  dataset.data_dir=/path/to/soar_data_sharded \
  train.batch_per_gpu=1 \
  train.grad_accum_steps=1
```
### Dynamics Training

```bash
torchrun --standalone --nproc_per_node=1 scripts/train_dynamics.py \
  --config-path scripts/config --config-name dynamics/soar.yaml \ 
     dataset.data_dir=/path/to/soar_data_sharded \
     train.batch_per_gpu=1 \
    train.grad_accum_steps=1
```

- `dataset.data_dir` should point to the sharded dataset folder containing `metadata.json` and `shard_*.h5`.
- `--nproc_per_node` controls the number of GPUs. Increase to use multiple GPUs on one machine.
- For multi-node runs, rely on your cluster launcher (SLURM) + torchrun env vars; we’ll document an example later.

**Requirement**: The small dynamic model (110M parameters) was trained on 4xH200 GPUs for 72 hours (SOAR dataset) while PushT model was trained on a single RTX6000 Pro (97GB) for same ammount of time. The tokenizer was trained on 24 H200 GPUs for 48 hours. With change of batch size and context length all our small models can be trained on a single RTX6000 Pro. 

## Repository Structure

### Main Components
- **Causal Tokenizer**
  - patchify images → mask/drop tokens → encode to latents → decode back to images
- **Interactive Dynamics (Denoiser)**
  - action-conditioned denoiser / dynamics model for rollout in latent space
- **Training**
  - distributed training via **DDP/FSDP**
- **Data**
  - preprocessing utilities to convert standard datasets to the format expected here
- **Interactive play**
  - `play-*.py` scripts for controller-driven interaction + imagination rollouts


### Structure
```
dreamer-v4/
├── checkpoints/
│   ├── dynamics_ckpts/        # pretrained dynamics weights
│   └── tokenizer_ckpts/       # pretrained tokenizer weights
├── dreamerv4/
│   ├── datasets.py            # dataset loading
│   ├── loss.py                # training losses
│   ├── sampling.py            # rollout / sampling utilities
│   ├── models/
│   │   ├── blocks.py          # attention, RoPE, KV-cache, transformer blocks
│   │   ├── tokenizer.py       # causal tokenizer (enc/dec + patchify + heads)
│   │   ├── dynamics.py        # denoiser / dynamics model
│   │   ├── utils.py
│   │   └── init.py
│   └── utils/
│       ├── distributed.py     # DDP/FSDP helpers
│       └── joy.py             # joystick/controller I/O
└── scripts/
├── config/                # hydra configs
├── preprocessing/         # dataset conversion scripts
├── train_tokenizer.py
├── train_dynamics.py
├── play-pushT.py
└── play-soar.py
```

## Citation
If you use our codebase for your projects and research, we would appreciate giving us a **star** here and citing the original dreamer paper:

```
@article{hafner2025training,
  title={Training agents inside of scalable world models},
  author={Hafner, Danijar and Yan, Wilson and Lillicrap, Timothy},
  journal={arXiv preprint arXiv:2509.24527},
  year={2025}
}
```

## Acknowledgement 
We would like to acknowledge the computational resources allocated to our project by [NYU Torch](https://www.nyu.edu/life/information-technology/research-computing-services/high-performance-computing/high-performance-computing-nyu-it.html), LAAS-Gepetto, ANITI, and [Jean Zay](http://www.idris.fr/eng/jean-zay/jean-zay-presentation-eng.html). This project is developed at [Machines in Motion Lab (MiM)](https://www.machinesinmotion.org/) at [NYU Center for Robotics and Embodied Intelligence (CREO)](https://engineering.nyu.edu/research/centers/nyu-center-robotics-and-embodied-intelligence-creo) with equal contribution from [Joseph Amigo](https://scholar.google.com/citations?user=-PPor9IAAAAJ&hl=en) and [Rooholla Khorrambakht](https://scholar.google.com/citations?user=VdgZUjoAAAAJ&hl=en). We would also like to acknowledge the use of AI (Perplexity, ChatGPT 5.2, and Gemini) in accelerating the development of this project.