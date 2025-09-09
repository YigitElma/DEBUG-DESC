#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=2:00:00
#SBATCH --constraint=cpu
#SBATCH --qos=premium
#SBATCH --account=m4680 # Change to your account number

module load python cray-hdf5/1.14.3.1 cray-netcdf/4.9.0.13
conda activate firm3d # Change to the name of your environment
srun -n 32 -c 1 python -u firm3d_scan.py
