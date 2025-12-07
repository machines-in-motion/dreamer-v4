#!/bin/bash
#SBATCH --job-name=dreamer_v4_single
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1           
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs_single/%x_%j.out
#SBATCH --error=logs_single/%x_%j.err
#SBATCH --account=torch_pr_155_tandon_advanced
#SBATCH --constraint='h200'

mkdir -p logs_single

module purge
# Run once per node
singularity exec --nv \
    --overlay /scratch/rk4342/dreamer-v4/hpc/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
    bash -c "
        source /ext3/env.sh; conda activate dreamerv4; python tokenizer_train_single.py
    "
