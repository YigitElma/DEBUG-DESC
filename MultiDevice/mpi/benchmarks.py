from desc import set_device

set_device("gpu")

import numpy as np
from desc.backend import print_backend_info, jnp
from desc.objectives import ObjectiveFunction
from desc.examples import get
from desc.grid import LinearGrid
from desc.objectives import QuasisymmetryTwoTerm, AspectRatio, EffectiveRipple
from timeit import timeit

if __name__ == "__main__":
    print_backend_info()

    eq = get("W7-X")

    # create two grids with different rho values, this will effectively separate
    # the quasisymmetry objective into two parts
    grid1 = LinearGrid(
        M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, rho=jnp.linspace(0.1, 1.0, 20), sym=True
    )
    ripple_grid = LinearGrid(
        rho=np.linspace(0.2, 1, 10), M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False
    )

    obj1 = QuasisymmetryTwoTerm(eq=eq, helicity=(1, eq.NFP), grid=grid1)
    obj2 = EffectiveRipple(
        eq,
        grid=ripple_grid,
        X=32,
        Y=64,
        Y_B=133,
        num_transit=10,
        num_well=30 * 20,
        num_quad=64,
        num_pitch=45,
    )
    obj3 = AspectRatio(eq=eq, target=8, weight=100)
    objs = [obj1, obj2, obj3]
    objective = ObjectiveFunction(objs)
    objective.build(verbose=3)
    num_runs = 100
    time_taken = timeit(
        lambda: objective.jac_scaled_error(objective.x(eq)).block_until_ready(),
        number=num_runs,
    )
    print(
        f"Average time taken for jac_scaled_error over {num_runs} runs: {time_taken / num_runs} seconds"
    )
