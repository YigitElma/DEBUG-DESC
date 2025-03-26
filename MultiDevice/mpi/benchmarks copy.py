from desc import set_device
import os

num_device = 3
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(1 / (num_device + 2))
set_device("gpu", num_device=num_device)


import numpy as np
from mpi4py import MPI
from desc.backend import print_backend_info, jnp
from desc.objectives import ObjectiveFunction
from desc.examples import get
from desc.grid import LinearGrid
from desc.objectives import QuasisymmetryTwoTerm, AspectRatio, EffectiveRipple
from timeit import timeit

if __name__ == "__main__":
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()
    if rank == 0:
        print(f"====== TOTAL OF {size} RANKS ======")
        print_backend_info()

    eq = get("W7-X")

    # create two grids with different rho values, this will effectively separate
    # the quasisymmetry objective into two parts
    grid1 = LinearGrid(
        M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=jnp.linspace(0.2, 1.0, 8), sym=True
    )
    ripple_grid = LinearGrid(
        rho=np.linspace(0.2, 1, 3), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False
    )

    # when using parallel objectives, the user needs to supply the device_id
    obj1 = QuasisymmetryTwoTerm(eq=eq, helicity=(1, eq.NFP), grid=grid1, device_id=0)
    obj2 = EffectiveRipple(
        eq,
        grid=ripple_grid,
        X=16,
        Y=32,
        Y_B=133,
        num_transit=10,
        num_well=25 * 10,
        num_quad=32,
        num_pitch=45,
        device_id=1,
    )
    obj3 = AspectRatio(eq=eq, target=8, weight=100, device_id=2)
    objs = [obj1, obj2, obj3]

    # Parallel objective function needs the MPI communicator
    # If you don't specify `deriv_mode=blocked`, you will get a warning and DESC will
    # automatically switch to `blocked`.
    objective = ObjectiveFunction(objs, deriv_mode="blocked", mpi=MPI)
    if rank == 0:
        objective.build(verbose=3)
    else:
        objective.build(verbose=0)
    with objective as objective:
        if rank == 0:
            num_runs = 100
            time_taken = timeit(
                lambda: objective.jac_scaled_error(objective.x(eq)), number=num_runs
            )
            print(
                f"Average time taken for jac_scaled_error over {num_runs} runs: {time_taken / num_runs} seconds"
            )
