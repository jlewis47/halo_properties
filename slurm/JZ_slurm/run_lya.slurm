#!/bin/bash
#SBATCH -A ybr@gpu
#SBATCH --qos=qos_gpu-dev
#SBATCH --nodes=1
#SBATCH  --ntasks=1
#SBATCH  --gres=gpu:0
#SBATCH  --cpus-per-task=32
#SBATCH  --time=02:00:00


cd ${SLURM_SUBMIT_DIR} 


module purge 
module load python



ulimit -c 0
set -x


srun  python run_lya.py >run_lya_log


