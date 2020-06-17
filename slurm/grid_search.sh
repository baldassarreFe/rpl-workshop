#!/usr/bin/env bash

# Submit a sbatch job without a physical .sbatch file using a HERE document.
# https://en.wikipedia.org/wiki/Here_document
#
# Variables and commands in the HERE document work like this:
# - ${RUNS_PATH}     is evaluated *now* and takes the value 
#                    from the current shell (as defined below),
#                    it's useful to pass paths and thyperparameters
# - \${SLURM_JOB_ID} is evaluated when the job starts, therefore
#                    you can access variables set by slurm
# - $(date)          is evaluated *now* and takes the value 
#                    from the current shell (as defined above)
# - \$(date)         is evaluated when the job starts, therefore
#                    you can run commands on the node
# 
# Before submitting you can replace `sbatch` with `cat` to check that 
# all variables and commands work as expected, you can also uncomment 
# `break 1000` below to break the for loops after the first iteration

SOURCE_PATH="${HOME}/rpl-workshop"
RUNS_PATH="${HOME}/rpl-workshop/runs"
DATA_PATH="/local_storage/datasets/CUB_20"

for learning_rate in .001 .01; do
for weight_decay in .001 .00001; do
for batch_size in 32 64; do

sbatch << HERE
#!/usr/bin/env bash
#SBATCH --output="${RUNS_PATH}/%J_slurm.out"
#SBATCH --error="${RUNS_PATH}/%J_slurm.err"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user="${USER}@kth.se"
#SBATCH --constrain="khazadum|rivendell|belegost|shire|gondor"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2GB

# Check job environment
echo "JOB: \${SLURM_JOB_ID}"
echo "HOST: \$(hostname)"
echo "SUBMITTED: $(date)"
echo "STARTED: \$(date)"
echo ""
nvidia-smi

# Activate conda
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate workshop

# Train and save the exit code of the python script
python -m workshop.train \
    --runpath "${RUNS_PATH}" \
    --datapath "${DATA_PATH}" \
    --batch_size "${batch_size}" \
    --learning_rate "${learning_rate}" \
    --weight_decay "${weight_decay}" \
    --number_epochs 20 \
    --number_workers 2 \
    --device 'cuda'
EXIT_CODE="\${?}"

# Perform post-train steps but return the exit code of the python script
if [ "\${EXIT_CODE}" -eq 0 ]; then
  # Success
else
  # Error (cleanup?)
fi
exit "\${EXIT_CODE}"
HERE

# break 1000
done
done
done