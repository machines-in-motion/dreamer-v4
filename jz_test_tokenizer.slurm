#!/bin/bash
#SBATCH --job-name=dreamer_v4_multinode
#SBATCH --nodes=6                          # 2 nodes
#SBATCH --ntasks-per-node=4                # 8 tasks (GPUs) per node
#SBATCH --gres=gpu:4                       # 8 GPUs per node
#SBATCH --constraint='h100'
#SBATCH --cpus-per-task=24                 # CPU cores per GPU
#SBATCH --time=02:00:00                    # Max runtime
#SBATCH --output=logs/dreamer_%j.out       # Output log
#SBATCH --error=logs/dreamer_%j.err        # Error log
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --account=ugb@h100 # H100 accounting
#SBATCH --qos=qos_gpu_h100-dev

# --qos=qos_gpu_h100-dev
# srun --pty --ntasks=1 --cpus-per-task=4 --hint=nomultithread --account=ugb@h100 --gres=gpu:1 --constraint='h100' --qos=qos_gpu_h100-dev bash

# Create logs directory
mkdir -p logs

export WANDB_MODE=offline

# Print job information
echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: 8"
echo "Total GPUs: $(($SLURM_JOB_NUM_NODES * 8))"
echo "Master node: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Network interface: $NCCL_SOCKET_IFNAME"
echo "=================================================="

module purge
module load arch/h100
module load miniforge/24.9.0
conda activate sheeprl_h100

#export TORCHDYNAMO_VERBOSE=1
#export TORCH_LOGS="aot_graphs"

srun python debug_train_dreamer_fsdp.py

echo "Training completed!"
