import jax
import jax.numpy as jnp

jax.config.update("jax_compilation_cache_dir", "./jax-caches")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../"))
sys.path.append(os.path.abspath("../../../"))

from desc import set_device

set_device("gpu")


import desc

from desc.basis import *
from desc.backend import *
from desc.compute import *
from desc.coils import *
from desc.equilibrium import *
from desc.examples import *
from desc.grid import *
from desc.geometry import *
from desc.io import *

from desc.objectives import *
from desc.objectives.objective_funs import *
from desc.objectives.getters import *
from desc.objectives.normalization import compute_scaling_factors
from desc.objectives.utils import *
from desc.optimize._constraint_wrappers import *

from desc.transform import Transform
from desc.plotting import *
from desc.optimize import *
from desc.perturbations import *
from desc.profiles import *
from desc.compat import *
from desc.utils import *
from desc.magnetic_fields import *
from desc.particles import *
from diffrax import *

from desc.__main__ import main
from desc.vmec_utils import vmec_boundary_subspace
from desc.input_reader import InputReader
from desc.continuation import solve_continuation_automatic
from desc.compute.data_index import register_compute_fun
from desc.optimize.utils import solve_triangular_regularized

print_backend_info()

eq0 = load("./equilibria/desc-eq-HBT_105995_06-rev-curr-stage2-fb.h5")
field0 = load("./coils/coils-eq-HBT_105995_06-rev-curr-stage2-fb.h5")
all_HBT_coils_wo_tf = field0[1:]
all_HBT_coils_wo_tf = MixedCoilSet(all_HBT_coils_wo_tf)
tf = ToroidalMagneticField(B0=0.32, R0=0.92)
field = [tf, all_HBT_coils_wo_tf]

eq = eq0.copy()
eq.change_resolution(L=16, M=16, L_grid=24, M_grid=24, N=3, N_grid=6, NFP=1, sym=True)
# eq.change_resolution(N=2, N_grid=4, NFP=1, sym=True)
coil_grid = LinearGrid(N=50)
eval_grid = LinearGrid(rho=np.array([1.0]), M=32, N=64, NFP=1, sym=False)
source_grid = LinearGrid(rho=np.array([1.0]), M=32, N=64, NFP=1, sym=False)

eq.solve(maxiter=300, ftol=1e-4, gtol=0, xtol=0, verbose=3)

objective = ObjectiveFunction(
    BoundaryError(
        eq=eq,
        field=field,
        field_fixed=True,
        field_grid=coil_grid,
        source_grid=source_grid,
        eval_grid=eval_grid,
        b_plasma_chunk_size=200,
        bs_chunk_size=200,
    ),
    jac_chunk_size=1,
)
constraints = (
    ForceBalance(eq=eq, grid=QuadratureGrid(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid)),
    FixPressure(eq=eq),
    FixPsi(eq=eq),
)
if eq.current is not None:
    constraints += (FixCurrent(eq),)
else:
    constraints += (FixIota(eq),)


k = 4
R_modes = np.vstack(
    (
        # [0, 0, 0],  # kind of cheating...
        eq.surface.R_basis.modes[np.max(np.abs(eq.surface.R_basis.modes), 1) > k, :],
    )
)
Z_modes = eq.surface.Z_basis.modes[np.max(np.abs(eq.surface.Z_basis.modes), 1) > k, :]
bdry_constraints = (
    FixBoundaryR(eq=eq, modes=R_modes),
    FixBoundaryZ(eq=eq, modes=Z_modes),
)
eq, out = eq.optimize(
    objective,
    constraints + bdry_constraints,
    optimizer="proximal-lsq-exact",
    x_scale="ess",
    verbose=3,
    maxiter=30,
    ftol=1e-3,
    options={"solve_options": {"ftol": 1e-3, "gtol": 0, "xtol": 0, "verbose": 0}},
)
eq0_lowres_fb = eq.copy()
eq0_lowres_fb.save(
    f"./equilibria/desc-eq-HBT_105995_06-rev-curr-stage2-fb-L{eq.L}M{eq.M}N{eq.N}-fb-k4.h5"
)
