#! /bin/bash -l
#SBATCH -A AST031
#SBATCH -J get_fescs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
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



#for snap in $1; do 
snap=$1

mpiexec -np $SLURM_NTASKS --bind-to core --report-bindings  python -u run_stellar_part_fesc.py $snap --sub_nb=3 --overwrite --overd_fact=50
wait

echo 'job done'