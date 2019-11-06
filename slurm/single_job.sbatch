#!/usr/bin/env bash
#SBATCH --output=/Midgard/home/%u/rpl-workshop/runs/%J_slurm.out
#SBATCH --error=/Midgard/home/%u/rpl-workshop/runs/%J_slurm.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=user@kth.se
#SBATCH --constrain="khazadum|rivendell|belegost|shire|gondor"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2GB

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate workshop

nvidia-smi

SOURCE_PATH="${HOME}/rpl-workshop"
RUNS_PATH="${HOME}/rpl-workshop/runs"
DATA_PATH="/local_storage/datasets/CUB_20"

python -m workshop.train \
    --runpath "${RUNS_PATH}" \
    --datapath "${DATA_PATH}" \
    --batch_size 64 \
    --learning_rate .001 \
    --weight_decay .00001 \
    --number_epochs 3 \
    --number_workers 2 \
    --device 'cuda'
