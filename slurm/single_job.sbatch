#!/usr/bin/env bash

SOURCE_PATH="${HOME}/rpl-workshop"                                  # TODO: path to the repository
RUNS_PATH="${HOME}/rpl-workshop/runs"                               # TODO: path to the log directory
DATA_PATH="/local_storage/datasets/CUB_20"                          # TODO: path to the dataset

#SBATCH --output=/Midgard/home/%u/rpl-workshop/runs/%J_slurm.out    # TODO: path to slurm outputs
#SBATCH --error=/Midgard/home/%u/rpl-workshop/runs/%J_slurm.err     # TODO: path to slurm errors
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=user@kth.se                                     # TODO: email where slurm sends notifications
#SBATCH --constrain="rivendell"                                     # TODO: node constrains
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2GB

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate workshop

nvidia-smi

python -m workshop.train \
    --datapath "${DATA_PATH}" \
    --batch_size 64 \
    --learning_rate .001 \
    --weight_decay .00001 \
    --number_epochs 3 \
    --number_workers 2 \
    --device 'cuda'
