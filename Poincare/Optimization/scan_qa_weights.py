import os
import subprocess
import numpy as np

weights_qs = [1.0, 10.0, 50.0]
weights_ar = [1.0, 10.0, 50.0, 100.0]
weights_vol = [1.0, 10.0, 50.0, 100.0]
weights_iota = [10.0, 20.0, 50.0, 100.0]

for w_qs in weights_qs:
    for w_ar in weights_ar:
        for w_vol in weights_vol:
            for w_iota in weights_iota:
                file_identifier = f"poincare_optimize_QA_wqs{w_qs}_war{w_ar}_wvol{w_vol}_wiota{w_iota}"

                if os.path.exists(f"./scan_results/eqfam_{file_identifier}.h5"):
                    print(
                        f"Skipping weights wqs={w_qs}, war={w_ar}, wvol={w_vol}, wiota={w_iota} since output file exists."
                    )
                    continue

                slurm_str = f"""#!/bin/bash
                #SBATCH --job-name={f"wqs{w_qs}_war{w_ar}_wvol{w_vol}_wiota{w_iota}"}
                #SBATCH --nodes=1                # node count
                #SBATCH -n 8        # cpu-cores per task (>1 if multi-threaded tasks)
                #SBATCH --mem=32G         # memory per cpu-core (4G is default)
                #SBATCH --time=00:45:00          # total run time limit (HH:MM:SS)
                #SBATCH --gres=gpu:1
                #SBATCH --constraint=nomig
                #SBATCH -o slurm_{file_identifier}.out

                module purge
                module load anaconda3/2024.10 
                conda activate desc-env

                python  run_qa_weights.py {w_qs} {w_ar} {w_vol} {w_iota}
                """
                with open("job.slurm_scan", "w+") as f:
                    f.write(slurm_str)
                result = subprocess.run(
                    ["sbatch", "job.slurm_scan"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
