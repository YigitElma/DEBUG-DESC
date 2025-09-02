import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../"))

from mpi4py import MPI

from desc import _set_cpu_count, set_device

num_device = 3
_set_cpu_count(num_device)
set_device(kind="cpu", num_device=num_device, mpi=MPI)

from desc.grid import LinearGrid
from desc.objectives import ObjectiveFunction, ForceBalance
from desc.examples import get
import numpy as np

np.set_printoptions(
    precision=1,
    suppress=False,
    formatter={"float": "{:3.1f}".format},
    linewidth=np.inf,
    threshold=sys.maxsize,
)

rank = MPI.COMM_WORLD.Get_rank()
eq = get("precise_QH")
eq.change_resolution(L=1, M=1, N=0, L_grid=1, M_grid=1, N_grid=0)

gM = eq.M_grid
gN = eq.N_grid
grid1 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.2], sym=True)
grid2 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.6, 0.8], sym=True)
grid3 = LinearGrid(M=gM, N=gN, NFP=eq.NFP, rho=[0.2], sym=True)

obj2 = ObjectiveFunction(
    [
        ForceBalance(eq, grid=grid1, device_id=0),
        ForceBalance(eq, grid=grid2, device_id=1),
        ForceBalance(eq, grid=grid3, device_id=2),
    ],
    mpi=MPI,
)
obj2.build(verbose=0)

with obj2:
    if rank == 0:
        print(obj2._f_sizes)
        print(obj2._f_displs)
        f2 = obj2.jac_unscaled(obj2.x(eq))
        print(f2.shape)
        print(f2)
