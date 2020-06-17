#!/usr/bin/env bash

# Submit a job array without a physical .sbatch file using config files a HERE document.
# https://en.wikipedia.org/wiki/Here_document
# https://slurm.schedmd.com/job_array.html
# 
# Before submitting prepare a `queue` folder where each file corresponds to one config.
# Each file is called `array.<date>.<id>.yaml`. Files corresponding to succesful runs
# are deleted. If the run fails the config file is moved to an `error` folder.
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

RUNS_PATH="${HOME}/rpl-workshop/runs"
DATA_PATH="/local_storage/datasets/CUB_20"
RUN_CONFIG_PREFIX="array.$(date +'%F_%T.%N')"
SLURM_MAX_TASKS=2
mkdir -p "${RUNS_PATH}/queue" "${RUNS_PATH}/error"

SLURM_ARRAY_TASK_ID=0
for learning_rate in .001 .01; do
for weight_decay in .001 .00001; do
for batch_size in 32 64; do

let "SLURM_ARRAY_TASK_ID++"

cat << HERE > "${RUNS_PATH}/queue/${RUN_CONFIG_PREFIX}.${SLURM_ARRAY_TASK_ID}.yaml"
paths:
    runs: ${RUNS_PATH}
    data: ${DATA_PATH}
dataloader:
    number_workers: 2
    batch_size: ${batch_size}
optimizer:
    learning_rate: ${learning_rate}
    weight_decay: ${weight_decay}
    number_epochs: 20
session:
    device: cuda
HERE

done
done
done

sbatch << HERE
#!/usr/bin/env bash
#SBATCH --output="${RUNS_PATH}/%A_%a_slurm.out"
#SBATCH --error="${RUNS_PATH}/%A_%a_slurm.err"
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user="${USER}@kth.se"
#SBATCH --constrain="khazadum|rivendell|belegost|shire|gondor"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=2GB
#SBATCH --job-name=${RUN_CONFIG_PREFIX}
#SBATCH --array=1-${SLURM_ARRAY_TASK_ID}%${SLURM_MAX_TASKS}

# Check job environment
echo "JOB: \${SLURM_ARRAY_JOB_ID}"
echo "TASK: \${SLURM_ARRAY_TASK_ID}"
echo "HOST: \$(hostname)"
echo "SUBMITTED: $(date)"
echo "STARTED: \$(date)"
echo ""
nvidia-smi

# Activate conda
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate workshop

# Train and save the exit code of the python script
python -m workshop.train_yaml \
    "${RUNS_PATH}/queue/${RUN_CONFIG_PREFIX}.\${SLURM_ARRAY_TASK_ID}.yaml"
EXIT_CODE="\${?}"

# Perform post-train steps but return the exit code of the python script
if [ "\${EXIT_CODE}" -eq 0 ]; then
  # Success
  rm "${RUNS_PATH}/queue/${RUN_CONFIG_PREFIX}.\${SLURM_ARRAY_TASK_ID}.yaml"
else
  # Error
  mv "${RUNS_PATH}/queue/${RUN_CONFIG_PREFIX}.\${SLURM_ARRAY_TASK_ID}.yaml" \
     "${RUNS_PATH}/error/${RUN_CONFIG_PREFIX}.\${SLURM_ARRAY_TASK_ID}.yaml"
fi
exit "\${EXIT_CODE}"
HERE
