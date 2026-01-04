#!/bin/bash
#SBATCH --job-name=dreamer_v4_eval_dynamics
#SBATCH --nodes=1                          # 2 nodes
#SBATCH --ntasks-per-node=1                # 8 tasks (GPUs) per node
#SBATCH --gres=gpu:1                       # 8 GPUs per node
#SBATCH --constraint='h100'
#SBATCH --cpus-per-task=24                 # CPU cores per GPU
#SBATCH --time=01:00:00                    # Max runtime
#SBATCH --output=logs/dreamer_%j.out       # Output log
#SBATCH --error=logs/dreamer_%j.err        # Error log
#SBATCH --hint=nomultithread
#SBATCH --account=ugb@h100 # H100 accounting
#SBATCH --qos=qos_gpu_h100-dev

# --qos=qos_gpu_h100-dev
# srun --pty --ntasks=1 --cpus-per-task=4 --hint=nomultithread --account=ugb@h100 --gres=gpu:1 --constraint='h100' --qos=qos_gpu_h100-dev bash

# Create logs directory
mkdir -p logs

export WANDB_MODE=offline

module purge
module load arch/h100
module load miniforge/24.9.0
conda activate sheeprl_h100

python eval_dynamics.py --tokenizer_ckpt="./logs/dreamer_v4_tokenizer/2025-11-18_08-12-02/latest.pt" --dynamics_ckpt="./logs/dreamer_v4_dynamics/2025-11-25_00-41-11/latest.pt"
