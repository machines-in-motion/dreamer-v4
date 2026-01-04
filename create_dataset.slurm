#!/bin/bash
#SBATCH --job-name=dreamer_v4_create_dataset
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8                  # CPU cores per GPU
#SBATCH --mem=64GB
#SBATCH --time=10:00:00                    # Max runtime
#SBATCH --output=logs/create_dataset_%j.out       # Output log
#SBATCH --error=logs/create_dataset_%j.err        # Error log
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
        python create_shard_hdf5.py
    "
