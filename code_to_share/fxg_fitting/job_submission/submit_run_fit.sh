#!/bin/bash
#SBATCH --account=ctb-vkaspi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # number of MPI processes
#SBATCH --cpus-per-task=40 # number of OpenMP processes
#SBATCH --mem=191000M # memory per node
#SBATCH --time=47:55:00

module load python/3.10
module load scipy-stack
source ~/ENV2/bin/activate

python run_fit.py

echo "Finished"