#!/usr/bin/env bash
#SBATCH --output          # TODO: path to slurm outputs
#SBATCH --error=          # TODO: path to slurm errors
#SBATCH --mail-type FAIL
#SBATCH --mail-user       # TODO: email where slurm sends notifications
#SBATCH --constrain       # TODO: node constrains
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task   # TODO: how many CPUs do you need
#SBATCH --mem=2GB

SOURCE_PATH=""            # TODO: path to the repository
RUNS_PATH=""              # TODO: path to the log directory
DATA_PATH=""              # TODO: path to the dataset

# TODO: make sure conda is available
# TODO: activate the workshop environment

nvidia-smi

python -m workshop.train \
    --runpath "${RUNS_PATH}" \
    --datapath "${DATA_PATH}" \
    --batch_size 64 \
    --learning_rate .001 \
    --weight_decay .00001 \
    --number_epochs 3 \
    --number_workers 2 \
    --device 'cuda'
