#!/bin/bash
#SBATCH --account=ctb-vkaspi
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 # number of MPI processes
#SBATCH --cpus-per-task=40 # number of OpenMP processes
#SBATCH --mem=191000M # memory per node
#SBATCH --time=11:55:00

module --force purge
module load StdEnv/2023
module load python/3.11.5
unset PYTHONPATH
source ~/ENV3/bin/activate

python curve_fit.py

echo "Finished"