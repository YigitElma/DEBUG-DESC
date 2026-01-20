import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../../"))

from desc import set_device

set_device("gpu")

import jax

jax.config.update("jax_compilation_cache_dir", "../jax-caches")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import numpy as np
import matplotlib.pyplot as plt

from desc.io import load
from desc.equilibrium import EquilibriaFamily
from desc.optimize import Optimizer
from desc.grid import LinearGrid
from desc.objectives import (
    ObjectiveFunction,
    ForceBalance,
    FixCurrent,
    FixSectionR,
    FixSectionZ,
    FixPressure,
    FixPsi,
    QuasisymmetryTwoTerm,
    AspectRatio,
    Volume,
    RotationalTransform,
    get_fixed_xsection_constraints,
)
from desc.examples import get
from desc.compat import rotate_zeta
from desc.plotting import plot_boozer_surface, plot_comparison, plot_1d, plot_section
from desc.backend import print_backend_info

print_backend_info()

# load initial equilibrium
try:
    eq = load("poincare_precise_QA_initial_eq_using_v16_updated.h5")
    eq.xsection = eq.get_surface_at(zeta=0)
    eq.surface = eq.get_surface_at(rho=1)
except FileNotFoundError:
    eq = load("../../../desc/examples/precise_QA_output.h5")[0]
    eq.xsection = eq.get_surface_at(zeta=0)
    eq.surface = eq.get_surface_at(rho=1)
    constraints = get_fixed_xsection_constraints(eq=eq)
    objective = ObjectiveFunction(ForceBalance(eq))

    # before optimization make sure that the initial equilibrium
    # is in force balance in terms of poincare constraints
    eq.solve(
        verbose=0,
        objective=objective,
        constraints=constraints,
        maxiter=100,
        ftol=1e-3,
    )
    eq.xsection = eq.get_surface_at(zeta=0)
    eq.surface = eq.get_surface_at(rho=1)
    eq.save("poincare_precise_QA_initial_eq_using_v16_updated.h5")

eqfam = EquilibriaFamily(eq)
eq00 = get("precise_QA")
V = eq.compute("V")["V"]
Vorg = eq00.compute("V")["V"]

# grid for computing quasisymmetry objective
grid = LinearGrid(M=eq.M, N=eq.N, NFP=eq.NFP, rho=np.array([0.6, 0.8, 1.0]), sym=True)

# weights for objectives
w_qs = float(sys.argv[1])
w_ar = float(sys.argv[2])
w_vol = float(sys.argv[3])
w_iota = float(sys.argv[4])

file_identifier = f"poincare_optimize_QA_wqs{w_qs}_war{w_ar}_wvol{w_vol}_wiota{w_iota}"


def run_step(n, eqfam, ftol=1e-2, **kwargs):
    objective = ObjectiveFunction(
        (
            QuasisymmetryTwoTerm(
                eq=eqfam[-1],
                helicity=(1, 0),
                grid=grid,
                normalize=False,
                weight=w_qs,
            ),
            AspectRatio(eq=eqfam[-1], target=6, weight=w_ar, normalize=False),
            Volume(
                eq=eqfam[-1], target=Vorg, weight=w_vol, normalize=False
            ),  # giving Vorg is kind of cheating
            RotationalTransform(
                eq=eqfam[-1], target=0.42, weight=w_iota, normalize=False
            ),
        ),
    )
    # modes to fix
    bc_surf = eqfam[-1].xsection
    R_modes = bc_surf.R_basis.modes[np.max(np.abs(bc_surf.R_basis.modes), 1) > n, :]
    Z_modes = bc_surf.Z_basis.modes[np.max(np.abs(bc_surf.Z_basis.modes), 1) > n, :]
    constraints = (
        ForceBalance(eq=eqfam[-1]),
        FixSectionR(eq=eqfam[-1], modes=R_modes),
        FixSectionZ(eq=eqfam[-1], modes=Z_modes),
        FixPressure(eq=eqfam[-1]),
        FixCurrent(eq=eqfam[-1]),
        FixPsi(eq=eqfam[-1]),
    )

    optimizer = Optimizer("proximal-lsq-exact")
    eq_new, _ = eqfam[-1].optimize(
        objective=objective,
        constraints=constraints,
        optimizer=optimizer,
        maxiter=250,
        verbose=3,
        ftol=ftol,
        copy=True,
        options={
            "perturb_options": {"verbose": 0},
            "solve_options": {"verbose": 0, "ftol": 1e-3, "maxiter": 250},
            **kwargs,
        },
    )
    # to make sure the surfaces are updated properly
    eq_new.xsection = eq_new.get_surface_at(zeta=0)
    eq_new.surface = eq_new.get_surface_at(rho=1)
    eqfam.append(eq_new)
    return eqfam


eqfam = run_step(8, eqfam, ftol=1e-3)
plot_boozer_surface(eqfam[-1])
plt.savefig(f"./scan_results/boozer_{file_identifier}.png", dpi=300)


eq_rotated = eqfam[-1].copy()
eq_rotated = rotate_zeta(eq_rotated, np.pi / 2)
plot_comparison(
    eqs=[eqfam[0], eq_rotated, eq00],
    labels=["Initial", "Poincare", "precise_QA"],
    color=["black", "blue", "red"],
)
plt.savefig(f"./scan_results/surfaces_{file_identifier}.png", dpi=500)

fig, ax = plot_1d(eqfam[-1], "iota", label="Poincare Optimized", color="blue")
plot_1d(eq00, "iota", ax=ax, label="precise_QA", color="red")
plt.legend()
plt.savefig(f"./scan_results/iota_{file_identifier}.png", dpi=300)

plot_section(eqfam[-1], "|F|_normalized", log=True)
plt.savefig(f"./scan_results/force_{file_identifier}.png", dpi=500)

eqfam.save(f"./scan_results/eqfam_{file_identifier}.h5")
