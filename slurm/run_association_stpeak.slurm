#! /bin/bash -l
#SBATCH -A AST031
#SBATCH -J stellar_association_stpeak
#SBATCH --nodes=1
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


# for i in  14 23 42 52 65 82 106; do

#     srun --exclusive python star_halo_latest.py $i 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII /lustre/orion/proj-shared/ast031/conaboy/fof/ll_0p200 --rtwo_fact=1 --assoc_mthd='stellar_peak'

#     srun --exclusive python  get_halo_mags.py $i 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII --ll=0.2 --rtwo_fact=1.0 --assoc_mthd='stellar_peak'

# done;

# for i in  14 23 42 52 65 82 106; do

#     srun --exclusive python star_halo_latest.py $i 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII /lustre/orion/proj-shared/ast031/conaboy/fof/ll_0p200 --rtwo_fact=1 --assoc_mthd=''

#     srun --exclusive python  get_halo_mags.py $i 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII --ll=0.2 --rtwo_fact=1.0 --assoc_mthd=''

# done;

# for i in  14 23 42 52 65 82 106; do

#     srun --exclusive python star_halo_latest.py $i 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII /lustre/orion/proj-shared/ast031/conaboy/fof/ll_0p200 --rtwo_fact=1 --assoc_mthd='stellar_barycentre'

#     srun --exclusive python  get_halo_mags.py $i 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII --ll=0.2 --rtwo_fact=1.0 --assoc_mthd='stellar_barycentre'

# done;

# for i in  14 23 42 52 65 82 106; do

#     srun --exclusive python star_halo_latest.py $i 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII /lustre/orion/proj-shared/ast031/conaboy/fof/ll_0p100 --rtwo_fact=1 --assoc_mthd='stellar_peak'

#     srun --exclusive python  get_halo_mags.py $i 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII --ll=0.1 --rtwo_fact=1.0 --assoc_mthd='stellar_peak'

# done;


# for i in  14 23 42 52 65 82 106; do

#     srun --exclusive python star_halo_latest.py $i 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII /lustre/orion/proj-shared/ast031/conaboy/fof/ll_0p100 --rtwo_fact=1 --assoc_mthd=''

#     srun --exclusive python  get_halo_mags.py $i 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII --ll=0.1 --rtwo_fact=1.0 --assoc_mthd=''

# done;

# #14 23 42 52 65 82 106
# #for i in 14 23 42 52 65 82 106; do
# for i in 101; do

# #srun --exclusive python -u star_halo_latest.py $i 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII /lustre/orion/proj-shared/ast031/conaboy/fof/ll_0p200 --rtwo_fact=1 --assoc_mthd='stellar_peak'
# #mpiexec -np $SLURM_NTASKS python -u star_halo_latest.py $i 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII /lustre/orion/proj-shared/ast031/conaboy/fof/ll_0p200 --rtwo_fact=2 --assoc_mthd='stellar_peak'
# mpiexec -np $SLURM_NTASKS python -u star_halo_latest.py $i 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII /lustre/orion/proj-shared/ast031/conaboy/fof/ll_0p100 --rtwo_fact=1 --assoc_mthd='stellar_peak'
# mpiexec -np $SLURM_NTASKS python -u star_halo_latest.py $i 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII /lustre/orion/proj-shared/ast031/conaboy/fof/ll_0p100 --rtwo_fact=2 --assoc_mthd='stellar_peak'

# #srun --exclusive python -u get_halo_mags.py $i 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII --ll=0.2 --rtwo_fact=1.0 --assoc_mthd='stellar_peak'

# done


# for i in 14 23 42 52 65 82 101 106; do
# for i in 116; do
# for i in 34; do

# #srun --exclusive python -u star_halo_latest.py $i 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII /lustre/orion/proj-shared/ast031/conaboy/fof/ll_0p200 --rtwo_fact=1 --assoc_mthd='stellar_peak'
# mpiexec -np $SLURM_NTASKS python -u star_halo_latest.py $i 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII /lustre/orion/proj-shared/ast031/conaboy/fof/ll_0p100 --rtwo_fact=1 --assoc_mthd='stellar_peak'
# mpiexec -np $SLURM_NTASKS python -u star_halo_latest.py $i 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII /lustre/orion/proj-shared/ast031/conaboy/fof/ll_0p100 --rtwo_fact=2 --assoc_mthd='stellar_peak'
# mpiexec -np $SLURM_NTASKS python -u star_halo_latest.py $i 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII /lustre/orion/proj-shared/ast031/conaboy/fof/ll_0p200 --rtwo_fact=2 --assoc_mthd='stellar_peak'
# mpiexec -np $SLURM_NTASKS python -u star_halo_latest.py $i 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII /lustre/orion/proj-shared/ast031/conaboy/fof/ll_0p200 --rtwo_fact=1 --assoc_mthd='stellar_peak'

# # srun --exclusive python -u get_halo_mags.py $i 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII --ll=0.2 --rtwo_fact=1.0 --assoc_mthd='stellar_peak'
# # srun --exclusive python -u get_halo_mags.py $i 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII --ll=0.1 --rtwo_fact=2.0 --assoc_mthd='stellar_peak'
# # srun --exclusive python -u get_halo_mags.py $i 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII --ll=0.2 --rtwo_fact=2.0 --assoc_mthd='stellar_peak'

# done


# srun --exclusive python -u star_halo_latest.py $i 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII /lustre/orion/proj-shared/ast031/conaboy/fof/ll_0p150 --rtwo_fact=1 --assoc_mthd='stellar_peak'
# srun --exclusive python -u star_halo_latest.py $i 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII /lustre/orion/proj-shared/ast031/conaboy/fof/ll_0p200 --rtwo_fact=1 --assoc_mthd='stellar_peak'
# mpiexec -np $SLURM_NTASKS python -u star_halo_latest.py $1 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII /lustre/orion/proj-shared/ast031/conaboy/fof/ll_0p200 --rtwo_fact=1 --assoc_mthd='stellar_peak'
# mpiexec -np $SLURM_NTASKS python -u star_halo_latest.py $1 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII /lustre/orion/proj-shared/ast031/conaboy/fof/ll_0p150 --rtwo_fact=1 --assoc_mthd='stellar_peak'
# mpiexec -np $SLURM_NTASKS python -u star_halo_latest.py $1 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII /lustre/orion/proj-shared/ast031/conaboy/fof/ll_0p100 --rtwo_fact=1 --assoc_mthd='stellar_peak' 
# mpiexec -np $SLURM_NTASKS python -u star_halo_latest.py $1 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII /lustre/orion/proj-shared/ast031/conaboy/fof/ll_0p150 --rtwo_fact=2 --assoc_mthd='stellar_peak'
# mpiexec -np $SLURM_NTASKS python -u star_halo_latest.py $1 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII /lustre/orion/proj-shared/ast031/conaboy/fof/ll_0p100 --rtwo_fact=2 --assoc_mthd='stellar_peak'
# mpiexec -np $SLURM_NTASKS python -u star_halo_latest.py $1 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII /lustre/orion/proj-shared/ast031/conaboy/fof/ll_0p200 --rtwo_fact=2 --assoc_mthd='stellar_peak'

srun -n $SLURM_NTASKS python -u run_association.py $1 --rtwo_fact=1.0 --assoc_mthd='stellar_peak' --ll=0.15

#with MP's segmentation
# mpirun -np $SLURM_NTASKS python -u run_association.py $1 --rtwo_fact=1 --assoc_mthd='stellar_peak' --mp --overwrite
# mpirun -np $SLURM_NTASKS python -u run_association.py $1 --rtwo_fact=1 --assoc_mthd='stellar_peak' --mp --overwrite
# mpiexec -np $SLURM_NTASKS python -u run_association.py $1 --rtwo_fact=2 --assoc_mthd='stellar_peak' --mp
# mpiexec -np $SLURM_NTASKS python -u run_association.py $1 --rtwo_fact=0.5 --assoc_mthd='stellar_peak' --mp 
# srun --exclusive python -u get_halo_mags.py $i 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII --ll=0.2 --rtwo_fact=1.0 --assoc_mthd='stellar_peak'
# srun --exclusive python -u get_halo_mags.py $i 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII --ll=0.1 --rtwo_fact=2.0 --assoc_mthd='stellar_peak'
# srun --exclusive python -u get_halo_mags.py $i 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII --ll=0.2 --rtwo_fact=2.0 --assoc_mthd='stellar_peak'



echo 'job done'
