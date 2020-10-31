#!/bin/bash
#SBATCH --output limits.out
#SBATCH --nodes  2

srun --nodes 1 --ntasks 1 python3 limits.py
sleep 2
srun --nodes 2 --ntasks 2 python3 limits.py
sleep 2
srun --nodes 2 --ntasks 4 --ntasks-per-node 2 --cpus-per-task 1 --mem-per-cpu 256M python3 limits.py
