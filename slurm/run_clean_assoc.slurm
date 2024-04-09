#! /bin/bash -l
#SBATCH -A AST031
#SBATCH -J clean_stellar_association_stpeak
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task 1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time 48:00:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

module purge
module load gcc
module load openmpi
module load hdf5
module load python

eval "$(conda shell.bash hook)"

source activate /ccs/proj/ast031/jlewis/conda_envs/andes/mpi4py

#34 42 52 65 82 106

export SLURM_CPU_BIND=none


mpirun -np $SLURM_NTASKS python -u run_clean_assoc.py $1 --rtwo_fact=1 --assoc_mthd='stellar_peak'  --ll=0.15


echo 'job done'
