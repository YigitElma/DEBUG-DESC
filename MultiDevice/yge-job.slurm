#!/bin/bash
#SBATCH --job-name=fb-parallel         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH -n 1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G         # memory per cpu-core (4G is default)
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --constraint=gpu80
#SBATCH --gres=gpu:2

module purge
module load anaconda3/2024.6
conda activate desc-env

python multidevice-desc.py

