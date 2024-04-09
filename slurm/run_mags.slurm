#! /bin/bash -l
#SBATCH -A AST031
#SBATCH -J get_mags
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=16
#SBATCH --time 48:00:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH \-\-mem=0

module load python


#for i in  14 23 42 65 82 106; do
for i in  34; do

mpiexec -np $SLURM_NTASKS python -u get_halo_mags.py $i 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII --ll=0.2 --rtwo_fact=1.0 --assoc_mthd='stellar_peak'

done

echo 'job done'
