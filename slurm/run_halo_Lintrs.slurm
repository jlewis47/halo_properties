#! /bin/bash -l
#SBATCH -A AST031
#SBATCH -J stellar_association
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time 48:00:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH \-\-mem=0

module load python




for i in 14 23 42 52 65 82 106; do

    srun --exclusive python  get_halo_Lintr.py $i 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII    

    srun --exclusive python  get_halo_Lintr.py $i 8192 /lustre/orion/proj-shared/ast031/pocvirk/CoDaIII/prod_sr/ CoDaIII --rtwo_fact=2.0    

done;


echo 'job done'
