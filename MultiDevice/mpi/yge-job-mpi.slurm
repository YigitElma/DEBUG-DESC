#!/bin/bash
#SBATCH --job-name=mpi-test        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=3               # total number of tasks
#SBATCH --cpus-per-task=1        # cpu-cores per task
#SBATCH --mem=12G         # memory per cpu-core (4G is default)
#SBATCH --time=00:15:00          # total run time limit (HH:MM:SS)

module purge
module load anaconda3/2024.6
module load openmpi/gcc/4.1.2
conda activate desc-mpi

mpirun -n 3 python mpi-script.py
# python dummy-test-case-mpi.py

