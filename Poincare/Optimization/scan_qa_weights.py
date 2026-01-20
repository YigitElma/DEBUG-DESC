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

weights_qs = [1.0]
weights_ar = [1.0, 10.0, 50.0, 100.0]
weights_vol = [1.0, 10.0, 50.0, 100.0]
weights_iota = [10.0, 20.0, 50.0, 100.0]

for w_qs in weights_qs:
    for w_ar in weights_ar:
        for w_vol in weights_vol:
            for w_iota in weights_iota:
                job_name = f"wqs{w_qs}_war{w_ar}_wvol{w_vol}_wiota{w_iota}"
                file_identifier = f"poincare_optimize_QA_wqs{w_qs}_war{w_ar}_wvol{w_vol}_wiota{w_iota}"

                if os.path.exists(f"./scan_results/eqfam_{file_identifier}.h5"):
                    print(
                        f"Skipping weights wqs={w_qs}, war={w_ar}, wvol={w_vol}, wiota={w_iota} since output file exists."
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
#SBATCH -o ./scan_results/slurm_{file_identifier}.out

module purge
module load anaconda3/2024.10
conda activate desc-env

python  run_qa_weights.py {w_qs} {w_ar} {w_vol} {w_iota}
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
