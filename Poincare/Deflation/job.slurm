#!/bin/bash
#SBATCH --job-name=def-poin
#SBATCH --nodes=1                # node count
#SBATCH -n 8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G         # memory per cpu-core (4G is default)
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig

module purge
module load anaconda3/2024.10
conda activate desc-env

python  deflation-for-poincare.py "precise_QA" 12 25
