import multiprocessing
import numpy as np

import jax

jax.config.update("jax_compilation_cache_dir", "./jax-caches")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


import jax.numpy as jnp
# jnp = np

def task_1(x, A):
    print(f"Process {multiprocessing.current_process().name} computing task_1({x}) = {x * 2}")
    q, r = jnp.linalg.qr(A)
    print(jnp.linalg.norm(q @ r - A))

def task_2(y, A):
    print(f"Process {multiprocessing.current_process().name} computing task_2({y}) = {y + 10}")
    q, r = jnp.linalg.qr(A)
    print(jnp.linalg.norm(q @ r - A))

def task_3(z, A):
    print(f"Process {multiprocessing.current_process().name} computing task_3({z}) = {z ** 2}")
    q, r = jnp.linalg.qr(A)
    print(jnp.linalg.norm(q @ r - A))

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)  # Use 'spawn' method
    key = jax.random.PRNGKey(758493)  # Random seed is explicit in JAX
    A = jax.random.uniform(key, shape=(1000, 100))
    # A = jax.random.rand(3000, 300)

    # Define processes with different functions and inputs
    p1 = multiprocessing.Process(target=task_1, args=(5,A), name="Worker-1")
    p2 = multiprocessing.Process(target=task_2, args=(3,A), name="Worker-2")
    p3 = multiprocessing.Process(target=task_3, args=(7,A), name="Worker-3")

    # Start processes
    p1.start()
    p2.start()
    p3.start()

    # Wait for processes to complete
    p1.join()
    p2.join()
    p3.join()

    print("All processes completed.")
