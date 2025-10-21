# test_jax_mpi.py
from mpi4py import MPI
import os

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# --- Option: force each rank to only see its own GPU ---
# Uncomment this block to restrict visible devices:
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

import jax
import jax.numpy as jnp

# Each rank picks a device (default is all visible)
devices = jax.devices("gpu")
if not devices:
    raise RuntimeError("No GPUs visible to rank %d" % rank)

print(f"Total of {size} ranks")
print(f"Rank {rank} sees devices {devices}")

if len(devices) == size:
    # If multiple devices are visible, pick one based on rank
    device = devices[rank]
else:
    device = devices[0]
print(f"Rank {rank} using device {device}")

comm.Barrier()

# Allocate a JAX array on that device
N = 5

if rank == 0:
    sendbuf = jnp.ones(N, dtype=jnp.float32, device=device) * (rank + 1)
    sendbuf.block_until_ready()
    comm.Bcast(sendbuf, root=0)
else:
    sendbuf = jnp.empty(N, dtype=jnp.float32, device=device)
    comm.Bcast(sendbuf, root=0)
