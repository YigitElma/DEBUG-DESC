import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../../"))
import numpy as np
import jax

jax.config.update("jax_compilation_cache_dir", "./jax-caches")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import multiprocessing
from desc.objectives import ForceBalance, ObjectiveFunction
from desc.grid import LinearGrid
from desc.examples import get

def worker_task(flat_obj, structure, x, queue):
    """Worker function that reconstructs ObjectiveFunction and computes scalar value."""
    # Reconstruct ObjectiveFunction using tree_unflatten
    print(f"Process {multiprocessing.current_process().name} computing")
    obj = jax.tree_util.tree_unflatten(structure, flat_obj)
    # Compute result
    result = obj.compute_scalar(x)
    print(f"Process {multiprocessing.current_process().name} computed!")
    queue.put((multiprocessing.current_process().name, result))



if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)  # Use 'forkserver' method

    eq = get("HELIOTRON")
    num_process = 4
    processes = []
    rhos = np.linspace(0.1, 0.8, num_process)
    queue = multiprocessing.Queue()

    objs = []
    for rho in range(num_process):
        rho = rhos[rho]
        # Define different grid instances
        grid1 = LinearGrid(rho=rho, M=3, N=3, NFP=eq.NFP)
        # Define and build ObjectiveFunctions
        obj1 = ForceBalance(eq, grid=grid1)
        objs.append(obj1)
    objective = ObjectiveFunction(objs)
    objective.build()
    x = objective.x(eq)
    for rho in range(num_process):
        obji = objective.objectives[rho]
        # Flatten ObjectiveFunctions before passing to workers
        flat_obj1, structure1 = jax.tree_util.tree_flatten(obji)
        # Define worker processes
        processes.append(
            multiprocessing.Process(target=worker_task, args=(flat_obj1, structure1, x, queue), name=f"Worker-{rho}")
            )
        
    print("Number of processes: ", len(processes))
    for p in processes:
        print(f"++++++++   Starting process {p.name}")
        p.start()

    results = []
    for _ in processes:
        worker_name, worker_results = queue.get()  # Blocks until a result is available
        results.append((worker_name, worker_results))

    for p in processes:
        p.join()

    for worker_name, worker_results in results:
        print(f"\nResults from {worker_name}:")
        print(f"  Result: {worker_results}")

    print("All processes completed.")



