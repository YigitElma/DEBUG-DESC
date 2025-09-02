# test_jax_mpi.py
from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# --- Option: force each rank to only see its own GPU ---
# Uncomment this block to restrict visible devices:
# os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

import jax
import jax.numpy as jnp
import numpy as np

# Each rank picks a device (default is all visible)
devices = jax.devices("gpu")
if not devices:
    raise RuntimeError("No GPUs visible to rank %d" % rank)

print(f"Rank {rank} sees devices {devices}")

device = devices[0]
print(f"Rank {rank} using device {device}")

# Allocate a JAX array on that device
N = 5
sendbuf = jnp.ones(N, dtype=jnp.float32) * (rank + 1)
sendbuf.block_until_ready()

if rank == 0:
    # Rank 0 receives from all others
    for src in range(1, size):
        recvbuf = jnp.empty_like(sendbuf, device=device)
        recvbuf.block_until_ready()
        comm.Recv(
            [recvbuf.__array_interface__["data"][0], MPI.FLOAT], source=src, tag=src
        )
        print(f"Rank 0 received from rank {src}: {np.array(recvbuf)}")
else:
    comm.Send([sendbuf.__array_interface__["data"][0], MPI.FLOAT], dest=0, tag=rank)
    print(f"Rank {rank} sent {np.array(sendbuf)} to rank 0")
