#!/bin/bash
#SBATCH --job-name=trace-w7-x
#SBATCH --nodes=1                # node count
#SBATCH -n 8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G         # memory per cpu-core (4G is default)
#SBATCH --time=03:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig

module purge
module load anaconda3/2024.10
conda activate desc-env

python profile-tracing-scale.py "w7-x"
