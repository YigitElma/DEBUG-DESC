from mpi4py import MPI
import numpy as np
import jax
import jax.numpy as jnp

@jax.jit
def fun(x):
    return x**2

@jax.jit
def jac(x):
    return jax.jacfwd(fun)(x)

@jax.jit
def fun2(x):
    y = jnp.empty(x.size * 2)
    y = y.at[:x.size].set(x**2)
    y = y.at[x.size:].set(x**2)
    return y

@jax.jit
def jac2(x):
    return jax.jacfwd(fun2)(x)

def compute(x):
    """Start MPI workers, compute, then pause them."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    assert size == 2

    # Wake up all ranks
    comm.Barrier()

    if rank == 1:
        # Compute on worker rank
        jac2arr = jac2(x).block_until_ready()
        jac2arr = np.array(jac2arr, dtype=np.float64)
        comm.Send(jac2arr, dest=0, tag=107)

    elif rank == 0:
        # Compute on rank 0
        jac1arr = jac(x).block_until_ready()
        jac1arr = np.array(jac1arr, dtype=np.float64)
        
        # Receive from rank 1
        jac2arr = np.empty((2 * jac1arr.shape[0], jac1arr.shape[1]), dtype=np.float64)
        comm.Recv(jac2arr, source=1, tag=107)

    # Pause all ranks until next call
    comm.Barrier()

    return jac1arr, jac2arr if rank == 0 else None

# --- MAIN PROGRAM ---
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    print("Running serial computations before MPI")

    for val in [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]:
        x = jnp.array(val)
        
        # Call compute
        jac1arr, jac2arr = compute(x)

        # Serial work continues
        jacobian = np.concatenate((jac1arr, jac2arr), axis=0)
        print(f"jac = \n{jacobian}")

    print("Serial computations after MPI")

# Rank 1 just waits inside `compute()` when not needed
comm.Barrier()
