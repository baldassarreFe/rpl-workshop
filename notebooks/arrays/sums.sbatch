#!/bin/bash
#SBATCH --output %A_%a.out

echo "This is job ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} on ${SLURMD_NODENAME}"

ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${ARGS_FILE}")
echo $ARGS
python3 sums.py $ARGS
