import jax
import jax.numpy as jnp
import numpy as np
from mpi4py import MPI

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    type = np.float64

    def Nrank(rank):
        return 2 * rank + 3

    N = Nrank(rank)
    M = 10
    x = np.ones((N, M), dtype=type) * rank
    x = x.T
    x[0] += 5

    # x = np.ones((M, N), dtype=type) * rank
    # x[0] += 5

    if rank == 0:
        print(x.ravel())
        print("Local array shape:", x.shape)
        print("Contents:")
        print(x)

    Ns = [Nrank(i) for i in range(size)]
    sizes = [Nrank(i) * M for i in range(size)]
    displs = [sum(sizes[:i]) for i in range(size)]

    Ns = np.array(Ns, dtype=np.int32)
    sizes = np.array(sizes, dtype=np.int32)
    displs = np.array(displs, dtype=np.int32)

    # Allocate receive buffer on root
    if rank == 0:
        recvbuf = np.empty((int(sum(Ns)), M), dtype=type)
    else:
        recvbuf = None

    # Gather arrays to root
    comm.Gatherv(x, (recvbuf, sizes, displs, MPI.DOUBLE), root=0)

    if rank == 0:
        print("Gatherv array shape:", recvbuf.shape)
        print("Contents:")
        print(recvbuf)
