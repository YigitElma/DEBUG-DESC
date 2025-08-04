import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../"))


from desc.backend import jax, jnp, np
import time
import gc
from desc.grid import LinearGrid
from desc.objectives import (
    ObjectiveFunction,
    EffectiveRipple,
)
from desc.integrals import Bounce2D
from desc.examples import get

X = 32
Y = 64
Y_B = 128
num_transit = 10
num_well = 20 * num_transit
num_quad = 64

eq = get("W7-X")
rho = np.linspace(0, 1, 10)
grid = LinearGrid(rho=rho, M=eq.M_grid, N=eq.N_grid, NFP=eq.NFP, sym=False)

obj = ObjectiveFunction(
    EffectiveRipple(
        eq,
        grid=grid,
        X=X,
        Y=Y,
        Y_B=Y_B,
        num_transit=num_transit,
        num_well=num_well,
        num_quad=num_quad,
        jac_chunk_size=1,  # to reduce the memory usage
    )
)
obj.build()


def compute_fun():
    jax.clear_caches()
    data = eq.compute(
        "effective ripple 3/2",
        grid=grid,
        theta=Bounce2D.compute_theta(eq, X=X, Y=Y, rho=rho),
        Y_B=128,
        num_transit=num_transit,
        num_well=num_well,
        surf_batch_size=2,
    )
    eps = data["effective ripple 3/2"].block_until_ready()
    return eps


def obj_fun():
    jax.clear_caches()
    eps = obj.compute_scaled_error(obj.x(eq)).block_until_ready()
    return eps


for i in range(3):
    print(f"Iteration {i+1} of compute_fun()")
    compute_fun()
    time.sleep(2)

gc.collect()
print("Sleep to cool down...")
time.sleep(15)


for i in range(3):
    print(f"Iteration {i+1} of obj_fun()")
    obj_fun()
    time.sleep(2)
