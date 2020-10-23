#!/bin/bash
#SBATCH -A ybr@gpu
#SBATCH --qos=qos_gpu-dev
#SBATCH --nodes=1
#SBATCH  --ntasks=1
#SBATCH  --gres=gpu:0
#SBATCH  --cpus-per-task=32
#SBATCH  --time=02:00:00





module purge 
module load python



ulimit -c 0
set -x

cd /gpfsscratch/rech/xpu/uoj51ok/1024/16Mpc/Metals_He_YDdust_BPASSV221_fesc0.5_Tsf3e4K_eps0.04_dtmmax0.5_lowTcooling_properHeCool/ 
srun amr2cube -inp output_00021 -out Z_00021 -typ 6

cd ${SLURM_SUBMIT_DIR}  

srun  python run.py >run_log


