#!/bin/bash
#SBATCH --job-name=poincare         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH -n 1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=12G         # memory per cpu-core (4G is default)
#SBATCH --time=01:30:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send mail when process begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=ye2698@princeton.edu
#SBATCH --constraint=gpu80
#SBATCH --gres=gpu:1

module purge
module load anaconda3/2021.11
conda activate desc-env

python all_solve_poincare.py
