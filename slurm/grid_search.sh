#!/usr/bin/env bash

SOURCE_PATH="${HOME}/rpl-workshop"          # TODO : add path to the repository directory
RUNS_PATH="${HOME}/rpl-workshop/runs"       # TODO : add path to the log directory
DATA_PATH="/local_storage/datasets/CUB_20"  # TODO : add path to the dataset
CONSTRAIN="rivendell"                       # TODO : add constrains on which node(s) to run
EMAIL=""                                    # TODO : add your email

SBATCH_OR_CAT=sbatch
# SBATCH_OR_CAT=cat

for learning_rate in .001 .01; do
for weight_decay in .001 .00001; do
for batch_size in 32 64; do

"${SBATCH_OR_CAT}" << HERE
#!/usr/bin/env bash
#SBATCH --output="${RUNS_PATH}/%J_slurm.out"
#SBATCH --error="${RUNS_PATH}/%J_slurm.err"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user="${EMAIL}"
#SBATCH --constrain=${CONSTRAIN} # TODO
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2GB

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate workshop

nvidia-smi

python -m workshop.train \
    --datapath "${DATA_PATH}" \
    --batch_size "${batch_size}" \
    --learning_rate "${learning_rate}" \
    --weight_decay "${weight_decay}" \
    --number_epochs 20 \
    --number_workers 2 \
    --device 'cuda'
HERE

# break 1000
done
done
done