import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../../"))

from desc import set_device, _set_cpu_count
num_device = 4
_set_cpu_count(num_device)
set_device("cpu", num_device=num_device)

from mpi4py import MPI
import numpy as np

from desc.objectives import ForceBalance, ObjectiveFunction
from desc.objectives.getters import get_parallel_forcebalance, get_fixed_boundary_constraints
from desc.grid import LinearGrid
from desc.examples import get

from desc.backend import jax

if __name__ == "__main__":
    rank = MPI.COMM_WORLD.Get_rank()
    # eq = get("HELIOTRON")
    # num_process = num_device
    # processes = []
    # rhos = np.linspace(0.1, 0.8, num_process)

    # objs = []
    # for i in range(num_process):
    #     rho = rhos[i]
    #     # Define different grid instances
    #     grid = LinearGrid(rho=rho, M=3, N=3, NFP=eq.NFP)
    #     # Define and build ObjectiveFunctions
    #     obj = ForceBalance(eq, grid=grid, device_id=i)
    #     if rank == 0:
    #         obj.build(verbose=3)
    #     else:
    #         obj.build(verbose=0)
    #     obj = jax.device_put(obj, device=obj._device)
    #     obj._things[0] = eq
    #     objs.append(obj)
    # objective = ObjectiveFunction(objs, mpi=MPI)
    # objective.build()

    # with objective as objective:
    #     if rank == 0:
    #         for _ in range(10):
    #             J = objective.jac_scaled_error(objective.x(eq))
    #             f = objective.compute_scaled_error(objective.x(eq))
    #         print(J.shape)
    #         print(f.shape)


    # ************* TEST FOR EQ SOLVE *************
    eq = get("HELIOTRON")
    eq.change_resolution(6,6,6,12,12,12)

    obj = get_parallel_forcebalance(eq, num_device=num_device, mpi=MPI, verbose=1)
    cons = get_fixed_boundary_constraints(eq)
    with obj as obj:
        if rank == 0:
            eq.solve(objective=obj, constraints=cons, maxiter=1, ftol=0, gtol=0, xtol=0, verbose=3)