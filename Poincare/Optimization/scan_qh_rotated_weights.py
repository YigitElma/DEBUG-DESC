import os
import sys
import subprocess
import numpy as np


def get_queued_job_names():
    # Get the current user's username
    user = "ye2698"

    # Command: squeue -u $USER -t PD,R -h -o "%j"
    # -u $USER: Filter by current user
    # -t PD,R:  Filter by state (PENDING or RUNNING)
    # -h:       No header in the output
    # -o "%j":  Output *only* the job name
    result = subprocess.run(
        ["squeue", "-u", user, "-t", "PD,R", "-h", "-o", "%j"],
        capture_output=True,
        text=True,
        check=True,
    )

    # Split the output by newlines and put non-empty names into a set
    job_names = set(filter(None, result.stdout.split("\n")))
    print(f"Found {len(job_names)} jobs already in queue (PD/R).")
    return job_names


# --- GET QUEUED JOBS ONCE AT THE START ---
queued_jobs = get_queued_job_names()

weights_qs = [1.0, 10.0]
weights_ar = [1.0, 10.0, 50.0, 100.0]
weights_vol = [5.0, 10.0, 50.0, 100.0]

for w_qs in weights_qs:
    for w_ar in weights_ar:
        for w_vol in weights_vol:
            job_name = f"wqs{w_qs}_war{w_ar}_wvol{w_vol}"
            file_identifier = f"rotated_poincare_optimize_QH_{job_name}"

            if os.path.exists(f"./scan_rotated_QH/eqfam_{file_identifier}.h5"):
                print(
                    f"Skipping weights wqs={w_qs}, war={w_ar}, wvol={w_vol} since output file exists."
                )
                continue

            if job_name in queued_jobs:
                print(f"Skipping (job in queue/running): {job_name}")
                continue

            slurm_str = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes=1                # node count
#SBATCH -n 8        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=32G         # memory per cpu-core (4G is default)
#SBATCH --time=00:45:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1
#SBATCH --constraint=nomig
#SBATCH -o ./scan_rotated_QH/slurm_{file_identifier}.out

module purge
module load anaconda3/2024.10
conda activate desc-env

python  run_qh_weights.py {w_qs} {w_ar} {w_vol}
"""
            with open("job.slurm_scan", "w+") as f:
                f.write(slurm_str)
            result = subprocess.run(
                ["sbatch", "job.slurm_scan"],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                # The sbatch command failed!
                print("\n" + "=" * 30)
                print(f"!! ERROR submitting job: {job_name}")
                print("!! Slurm Error Message:")
                print(result.stderr)
                print("=" * 30)
                print("\nStopping submission script. You likely hit a job quota.")
                sys.exit(1)  # Exit the entire script
            else:
                # Success, print the sbatch output (e.g., "Submitted batch job 12345")
                print(result.stdout.strip())
