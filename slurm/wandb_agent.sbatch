#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2GB

echo ${HOME}
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate workshop

wandb agent ${wandb_sweep_id} --count 1