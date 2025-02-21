import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../"))

num_device = 2
from desc import set_device, _set_cpu_count

_set_cpu_count(num_device)
set_device("cpu", num_device=num_device)

import jax

jax.config.update("jax_compilation_cache_dir", "./jax-caches")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


import numpy as np

from desc import config as desc_config
from desc.examples import get
from desc.objectives import *
from desc.objectives.getters import *
from desc.grid import LinearGrid
from desc.backend import jnp
from desc.plotting import plot_grid
from desc.backend import jax
from desc.optimize import Optimizer

if __name__ == "__main__":
    eq = get("HELIOTRON")
    eq.change_resolution(3, 3, 3, 6, 6, 6)

    obj = get_parallel_forcebalance(eq, num_device=num_device)
    cons = get_fixed_boundary_constraints(eq)
    for obji in obj.objectives:
        print(jax.devices(desc_config["kind"])[obji._device_id])

    eq.solve(objective=obj, constraints=cons, maxiter=1, ftol=0, gtol=0, xtol=0, verbose=3)