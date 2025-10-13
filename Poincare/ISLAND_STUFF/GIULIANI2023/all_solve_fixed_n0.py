import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../../../"))


from desc import set_device

set_device("gpu")


import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import functools
import scipy

import desc

from desc.basis import *
from desc.backend import *
from desc.compute import *
from desc.coils import *
from desc.equilibrium import *
from desc.examples import *
from desc.grid import *
from desc.geometry import *

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

from desc.__main__ import main
from desc.vmec_utils import vmec_boundary_subspace
from desc.input_reader import InputReader
from desc.continuation import solve_continuation_automatic

print_backend_info()

paper = "giuliani2023"
mode = "fixed-n0"  # "poincare"


def plot_field_lines(field, equ, ntransit=200, nrho=9, size=0.4, outside=False):
    # for starting locations we'll pick positions on flux surfaces on the outboard midplane
    if outside:
        grid_trace = LinearGrid(rho=np.linspace(0, 1, 2))
        r0 = equ.compute("R", grid=grid_trace)["R"]
        z0 = equ.compute("Z", grid=grid_trace)["Z"]
        rmax = np.max(r0)
        rmin = np.min(r0)
        r0 = np.linspace(rmin, rmax + (rmax - rmin) * 0.05, nrho)
        z0 = np.zeros_like(r0)
    else:
        grid_trace = LinearGrid(rho=np.linspace(0, 1, nrho))
        r0 = equ.compute("R", grid=grid_trace)["R"]
        z0 = equ.compute("Z", grid=grid_trace)["Z"]
    fig, ax = plot_surfaces(equ)
    fig, ax = poincare_plot(
        field,
        r0,
        z0,
        NFP=equ.NFP,
        ax=ax,
        color="k",
        size=size,
        ntransit=ntransit,
        bounds_R=(0.5, 1.5),
        bounds_Z=(-0.7, 0.7),
    )
    return fig, ax


def optimize_coils_regcoil(surf, equ, num_coils=16, return_k=False):
    # create the FourierCurrentPotentialField object from the constant offset surface we found in the previous cell
    surface_current_field = FourierCurrentPotentialField.from_surface(
        surf,
        I=0,
        # manually setting G to value needed to provide the equilibrium's toroidal flux,
        # though this is not necessary as it gets set automatically inside the solve_regularized_surface_current function
        G=np.asarray(
            [
                -equ.compute("G", grid=LinearGrid(rho=np.array(1.0)))["G"][0]
                / mu_0
                * 2
                * np.pi
            ]
        ),
        # set symmetry of the current potential, "sin" is usually expected for stellarator-symmetric surfaces and equilibria
        sym_Phi="sin",
    )

    surface_current_field.change_Phi_resolution(M=12, N=12)

    # create the evaluation grid (where Bn will be minimized on plasma surface)
    # and source grid (discretizes the source K for Biot-Savart and where |K| will be penalized on winding surface)
    Megrid = 20
    Negrid = 20
    Msgrid = 20
    Nsgrid = 20

    eval_grid = LinearGrid(M=Megrid, N=Negrid, NFP=equ.NFP, sym=False)
    # ensure that sym=False for source grid so the field evaluated from the surface current is accurate
    # (i.e. must evaluate source over whole surface, not just the symmetric part)
    # NFP>1 is ok, as we internally will rotate the source through the field periods to sample entire winding surface
    sgrid = LinearGrid(M=Msgrid, N=Nsgrid, NFP=equ.NFP, sym=False)

    lambda_regularization = np.append(np.array([0]), np.logspace(-30, 1, 20))

    # solve_regularized_surface_current method runs the REGCOIL algorithm
    fields, data = solve_regularized_surface_current(
        surface_current_field,  # the surface current field whose geometry and Phi resolution will be used
        eq=equ,  # the Equilibrium object to minimize Bn on the surface of
        source_grid=sgrid,  # source grid
        eval_grid=eval_grid,  # evaluation grid
        current_helicity=(
            1 * surface_current_field.NFP,
            -1,
        ),  # pair of integers (M_coil, N_coil), determines topology of contours (almost like  QS helicity),
        #  M_coil is the number of times the coil transits poloidally before closing back on itself
        # and N_coil is the toroidal analog (if M_coil!=0 and N_coil=0, we have modular coils, if both M_coil
        # and N_coil are nonzero, we have helical coils)
        # we pass in an array to perform scan over the regularization parameter (which we call lambda_regularization)
        # to see tradeoff between Bn and current complexity
        lambda_regularization=lambda_regularization,
        # lambda_regularization can also be just a single number in which case no scan is performed
        vacuum=True,  # this is a vacuum equilibrium, so no need to calculate the Bn contribution from the plasma currents
        regularization_type="regcoil",
        chunk_size=40,
    )
    surface_current_field = fields[
        0
    ]  # fields is a list of FourierCurrentPotentialField objects

    if return_k:
        return surface_current_field
    else:
        coilset = surface_current_field.to_CoilSet(num_coils=num_coils, stell_sym=True)
        return coilset


def solve_poincare(eq2solve, fix_lambda=True, **kwargs):
    jac_chunk_size = kwargs.pop("jac_chunk_size", None)
    constraints = get_fixed_xsection_constraints(eq2solve, fix_lambda=fix_lambda)
    objective = ObjectiveFunction(ForceBalance(eq2solve, jac_chunk_size=jac_chunk_size))
    eq2solve.solve(constraints=constraints, objective=objective, verbose=3, **kwargs)


def solve_n0_fixed(eq2solve, **kwargs):
    jac_chunk_size = kwargs.pop("jac_chunk_size", None)
    R_modes = eq2solve.R_basis.modes[eq2solve.R_basis.modes[:, 2] == 0]
    Z_modes = eq2solve.Z_basis.modes[eq2solve.Z_basis.modes[:, 2] == 0]
    cons = (
        FixModeR(eq2solve, modes=R_modes),
        FixModeZ(eq2solve, modes=Z_modes),
        FixPressure(eq2solve),
        FixPsi(eq2solve),
        FixCurrent(eq2solve),
        FixSheetCurrent(eq2solve),
        FixLambdaGauge(eq2solve),
    )
    cons = maybe_add_self_consistency(eq2solve, cons)
    obj = ObjectiveFunction(ForceBalance(eq2solve, jac_chunk_size=jac_chunk_size))
    eq2solve.solve(
        constraints=cons,
        objective=obj,
        verbose=3,
        **kwargs,
    )


def all_above(equ):
    print(f"\n\nSOLVING {mode.upper()} FOR L={equ.L} M={equ.M} N={equ.N}\n\n")
    equ.xsection = equ.get_surface_at(zeta=0)
    equ.surface = equ.get_surface_at(rho=1)
    eqp = equ.copy()
    if mode == "fixed-n0":
        solve_n0_fixed(eqp, maxiter=1000, ftol=5e-4, xtol=0, gtol=0)
    elif mode == "poincare":
        solve_poincare(eqp, maxiter=1000, ftol=5e-4, xtol=0, gtol=0)
    eqp.surface = eqp.get_surface_at(rho=1)
    eqp.xsection = eqp.get_surface_at(zeta=0)
    eqp.save(f"eq-{mode}/eq-{mode}-{paper}-island-L{eqp.L}M{eqp.M}N{eqp.N}.h5")
    fig, ax = plot_comparison(
        eqs=[equ, eqp],
        labels=[
            f"original L{equ.L}M{equ.M}N{equ.N}",
            f"resolve {mode} L{eqp.L}M{eqp.M}N{eqp.N}",
        ],
    )
    fig.savefig(
        f"eq-{mode}/plot-{paper}-surface-compare-{mode}-L{eqp.L}M{eqp.M}N{eqp.N}.png",
        dpi=1000,
    )
    plt.close()
    regcoil = 1
    return_k = 1
    if regcoil:
        # create the constant offset surface
        surf2 = eqp.surface.constant_offset_surface(
            offset=0.25,  # desired offset
            M=16,  # Poloidal resolution of desired offset surface
            N=16,  # Toroidal resolution of desired offset surface
            grid=LinearGrid(M=32, N=32, NFP=eqp.NFP),
        )  # grid of points on base surface to evaluate unit normal and find points on offset surface,
        # generally should be twice the desired resolution
        optimized_coilset2 = optimize_coils_regcoil(
            surf2, eqp, num_coils=8, return_k=return_k
        )
    optimized_coilset2.save(
        f"eq-{mode}/surface-K-{paper}-{mode}-L{eqp.L}M{eqp.M}N{eqp.N}.h5"
    )
    fig, ax = plot_field_lines(
        optimized_coilset2, eqp, nrho=18, ntransit=200, size=0.2, outside=False
    )
    fig.suptitle(f"Field Line Trace after {mode} L={eqp.L} M={eqp.M} N={eqp.N}")
    fig.savefig(
        f"eq-{mode}/plot-{paper}-{mode}-L{eqp.L}M{eqp.M}N{eqp.N}.png", dpi=1000
    )
    plt.close()

    fig, ax = plot_1d(equ, "iota", label="Original", linecolor="r")
    fig, ax = plot_1d(eqp, "iota", label=f"{mode}", ax=ax, linecolor="b")
    fig.suptitle(f"L={eqp.L}M={eqp.M}N={eqp.N}")
    fig.savefig(
        f"eq-{mode}/iota-{paper}-{mode}-L{eqp.L}M{eqp.M}N{eqp.N}.png", dpi=500
    )
    plt.close()


import glob

pwd = os.getcwd()

eq = desc.io.load(f"eq-org/eq-org-{paper}-island-LMN10.h5")

coil_grid = LinearGrid(N=50)
plasma_grid = LinearGrid(M=25, N=25, NFP=eq.NFP, sym=eq.sym)

for fname in glob.glob(pwd + "/eq-org/*.h5"):
    foutputname = fname.split(".")[0].split("/")[-1]
    print(f"Plotting the output file {foutputname}")
    eqi = desc.io.load(fname)
    all_above(eqi)
