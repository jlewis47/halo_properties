#! /bin/bash -l
#SBATCH -A AST031
#SBATCH -J get_fescs
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
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
dilate=4 #sums and means resample the grid 2**dilate times to get <rtwo_fact * r200 cell values -- higher numbers mean closer to truth, but anything greater than 3-4 is pretty slow


#for snap in $1; do 
snap=$1
mpiexec -np $SLURM_NTASKS --bind-to core --report-bindings  python -u halo_fesc_latest_big_boxes.py $snap --assoc_mthd="stellar_peak" --ll=0.2 --rtwo_fact=1.0 --dilate $dilate #--overwrite
wait
mpiexec -np $SLURM_NTASKS --bind-to core --report-bindings  python -u halo_fesc_latest_big_boxes.py $snap --assoc_mthd="stellar_peak" --ll=0.1 --rtwo_fact=1.0 --dilate $dilate #--overwrite
wait
mpiexec -np $SLURM_NTASKS --bind-to core --report-bindings  python -u halo_fesc_latest_big_boxes.py $snap --assoc_mthd="stellar_peak" --ll=0.15 --rtwo_fact=1.0 --dilate $dilate #--overwrite
wait
mpiexec -np $SLURM_NTASKS --bind-to core --report-bindings  python -u halo_fesc_latest_big_boxes.py $snap --assoc_mthd="stellar_peak" --ll=0.2 --rtwo_fact=2.0 --dilate $dilate #--overwrite
wait
mpiexec -np $SLURM_NTASKS --bind-to core --report-bindings  python -u halo_fesc_latest_big_boxes.py $snap --assoc_mthd="stellar_peak" --ll=0.1 --rtwo_fact=2.0 --dilate $dilate #--overwrite
wait
mpiexec -np $SLURM_NTASKS --bind-to core --report-bindings  python -u halo_fesc_latest_big_boxes.py $snap --assoc_mthd="stellar_peak" --ll=0.15 --rtwo_fact=2.0 --dilate $dilate #--overwrite
wait
#done


echo 'job done'