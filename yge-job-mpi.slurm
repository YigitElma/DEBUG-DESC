#!/bin/bash
#SBATCH --job-name=mpi-test        # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=2               # total number of tasks
#SBATCH --cpus-per-task=16        # cpu-cores per task
#SBATCH --mem=12G         # memory per cpu-core (4G is default)
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:2

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

module purge
module load anaconda3/2024.6
# module load openmpi/gcc/4.1.2
# conda activate desc-mpi

module load nvhpc/25.5 openmpi/cuda-12.9/nvhpc-25.5/4.1.8 cudatoolkit/12.9
conda activate mpi-cuda

# srun -n 2 python docs/notebooks/tutorials/mpi-tutorials/mpi-eq-solve.py
# srun -n 2 python CUDA-mpi-jax.py
srun -n 2 --map-by ppr:1:gpu python CUDA-mpi-jax.py
