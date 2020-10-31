#!/bin/bash
#SBATCH --output %A_%a.out

echo "This is job ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} on ${SLURMD_NODENAME}"

ARGS="${ARGS_PREFIX}_${SLURM_ARRAY_TASK_ID}.json"
cat "$ARGS"
python3 prods.py "$ARGS"
