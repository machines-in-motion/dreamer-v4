#!/bin/bash
#SBATCH --job-name=dreamer_v4_eval
#SBATCH --nodes=1                          # 1 nodes
#SBATCH --ntasks-per-node=1                # 1 tasks (GPUs) per node
#SBATCH --gres=gpu:1                       # 1 GPUs per node
#SBATCH --constraint='h200'
#SBATCH --cpus-per-task=16                 # CPU cores per GPU
#SBATCH --mem=64GB                         # Request all memory on node
#SBATCH --time=64:00:00                    # Max runtime
#SBATCH --output=logs/dreamer_eval_%j.out       # Output log
#SBATCH --error=logs/dreamer_eval_%j.err        # Error log
#SBATCH --account torch_pr_155_tandon_advanced

# Create logs directory
mkdir -p logs

module purge

# Launch training directly with srun
srun singularity exec --nv \
    --overlay /scratch/ja5009/dmo_dreamer_diffusion/overlay-25GB-500K.ext3:ro \
    /share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif \
    /bin/bash -c "
        source /ext3/env.sh
        conda activate dmo_dreamer_diffusion
        python eval_tokenizer.py
    "

echo "Training completed!"
