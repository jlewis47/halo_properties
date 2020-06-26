#!/bin/bash
#SBATCH -A ybr@gpu
#SBATCH --nodes=1
#SBATCH  --ntasks=1
#SBATCH  --gres=gpu:0
#SBATCH  --cpus-per-task=32
#SBATCH  --time=01:00:00


cd ${SLURM_SUBMIT_DIR} 


module purge 
module load python

conda activate healpy_loc_env

ulimit -c 0
set -x


srun  python run.py >run_log


conda deactivate

