#!/usr/bin/env bash
#SBATCH --output=/Midgard/home/%u/rpl-workshop/runs/%J_slurm.out
#SBATCH --error=/Midgard/home/%u/rpl-workshop/runs/%J_slurm.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2GB

echo ${HOME}
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate workshop

wandb agent ${wandb_sweep_id} --count 1

