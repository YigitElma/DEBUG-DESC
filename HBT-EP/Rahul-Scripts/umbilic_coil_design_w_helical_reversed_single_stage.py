#!/usr/bin/env python3
from desc import set_device

set_device("gpu")
import shutil

import pdb
import numpy as np

from desc.coils import (
    CoilSet,
    FourierPlanarCoil,
    FourierRZCoil,
    FourierXYZCoil,
    MixedCoilSet,
    SplineXYZCoil,
)
from desc.compute import rpz2xyz_vec, xyz2rpz, xyz2rpz_vec, rpz2xyz
from desc.examples import get
from desc.geometry import FourierRZCurve, FourierRZToroidalSurface
from desc.grid import LinearGrid
from desc.magnetic_fields import SumMagneticField, VerticalMagneticField
from desc.backend import jnp

from desc.coils import CoilSet

from desc.equilibrium import Equilibrium
from desc.profiles import PowerSeriesProfile

import numpy as np

# import pytest
from scipy.constants import mu_0

from desc.backend import jit, jnp
from desc.basis import DoubleFourierSeries
from desc.compute import rpz2xyz_vec, xyz2rpz_vec
from desc.examples import get
from desc.geometry import FourierRZToroidalSurface, FourierXYZCurve
from desc.grid import LinearGrid
from desc.io import load
from desc.magnetic_fields import (
    CurrentPotentialField,
    DommaschkPotentialField,
    FourierCurrentPotentialField,
    MagneticFieldFromUser,
    OmnigenousField,
    PoloidalMagneticField,
    ScalarPotentialField,
    SplineMagneticField,
    ToroidalMagneticField,
    VerticalMagneticField,
    field_line_integrate,
    read_BNORM_file,
)
from desc.utils import dot
from desc.plotting import *
from desc.coils import CoilSet, FourierPlanarCoil

# from desc.objectives._reconstruction import FluxLoop, RogowskiLoop
from desc.examples import get
from desc.magnetic_fields import ToroidalMagneticField
from desc.objectives import (
    BoundaryError,
    FixBoundaryR,
    FixBoundaryZ,
    FixCurrent,
    FixPressure,
    FixSumCoilCurrent,
    FixPsi,
    ObjectiveFunction,
    ForceBalance,
    QuadraticFlux,
    FixParameters,
    FixIota,
    CoilSetMinDistance,
    CoilLength,
    CoilSetMinDistance,
    CoilBounds,
    PlasmaCoilSetMinDistance,
    PlasmaCoilSetDistanceBound,
    LinkingCurrentConsistency,
)
from desc.optimize import Optimizer


# eq = load("beak_equilibrium_medium_beak_beta1p0.h5")
# eq = load("beak_equilibrium_medium_beak_beta1p0_current.h5")
# eq = load("eq_final.h5")
eq = load("eq_final2.h5")
# eq = load("eq_final3.h5")

# eq = load("beak_equilibrium_beta1p0.h5")
print(eq.L_grid, eq.M_grid, eq.N_grid)

## Converting iota profile from spline to power series
# grid0 = LinearGrid(L=100)
# rho_profile = np.linspace(0, 1, 101)
# current = eq.compute("current", grid=grid0)["current"]
# current_sign = np.sign(np.mean(eq.compute("current", grid=grid0)["current"]))
# current_profile = PowerSeriesProfile.from_values(rho_profile, current, order=2*eq.L, sym=True)
# current_profile.params[0] = 0.
# eq.current = current_profile

# eq.iota = None

# eq, _ = eq.solve(objective="force", ftol=1e-4, xtol=1e-6, gtol=1e-6, maxiter=100)
# eq.save("beak_equilibrium_medium_beak_beta1p0_current.h5")

grid_at_surf = LinearGrid(rho=1.0, M=eq.M_grid, N=eq.N_grid)
G_tot = 2 * jnp.pi * eq.compute("G", grid=grid_at_surf)["G"][0] / mu_0

# field = ToroidalMagneticField(R0=1, B0=mu_0*G_tot/2/jnp.pi) + VerticalMagneticField(mu_0*G_tot/2/jnp.pi)

sign0 = -1

coil1 = FourierPlanarCoil(
    current=-sign0 * 8e3, center=[0, 0, 0.8], normal=[0, 0, 1], r_n=1.4
)  # Outer VF
coil2 = FourierPlanarCoil(
    current=-sign0 * 8e3, center=[0, 0, -0.8], normal=[0, 0, 1], r_n=1.4
)

coil5 = FourierPlanarCoil(
    current=sign0 * 2e4, center=[0, 0, 0.35], normal=[0, 0, 1], r_n=0.3
)  # Inner VF
coil6 = FourierPlanarCoil(
    current=sign0 * 2e4, center=[0, 0, -0.35], normal=[0, 0, 1], r_n=0.3
)

n_coils = 20
TFcoil = FourierPlanarCoil(
    current=-sign0 * G_tot / n_coils, center=[1.1, 0, 0], normal=[0, 1, 0], r_n=0.65
)
TFcoil.change_resolution(N=0)
coil3 = CoilSet(TFcoil, NFP=n_coils)

minor_radius = eq.compute("a")["a"]
offset = 1.8 * minor_radius
num_coils = 1  # coils per half field period per coilset
zeta = np.linspace(0, 2 * np.pi, 41)
grid = LinearGrid(rho=[0.0], zeta=zeta, NFP=eq.NFP)
data = eq.axis.compute(["x", "x_s"], grid=grid, basis="xyz")
print(np.mean(data["x"], axis=0))
helical_offset = -0.0
# R0 = 0.94
R0 = 1.0
# import pdb
# pdb.set_trace()

# R = R0 + offset* (1-0.1*np.cos(zeta)) * np.cos(zeta - helical_offset)
# Z = offset* (1-0.1*np.cos(zeta)) * np.sin(zeta - helical_offset)

R = R0 + offset * np.cos(zeta - helical_offset)
Z = offset * np.sin(zeta - helical_offset)


data = jnp.vstack([R, zeta, Z]).T
coil4 = FourierRZCoil.from_values(
    current=-2.1e3,
    # current=0.,
    coords=data,
    N=10,
    basis="rpz",  # we are giving the center and normal in cylindrical
)


field = MixedCoilSet(coil1, coil2, coil3, coil4, coil5, coil6)
# field = CoilSet.load("field_opt_final3.h5")

coil_grid = LinearGrid(N=50)


# field = CoilSet.load("field_opt_final.h5")
field = CoilSet.load("field_opt_final2.h5")

k = 7

modes_R = np.vstack(
    (
        [0, 0, 0],
        eq.surface.R_basis.modes[np.max(np.abs(eq.surface.R_basis.modes), 1) > k, :],
    )
)
modes_Z = eq.surface.Z_basis.modes[np.max(np.abs(eq.surface.Z_basis.modes), 1) > k, :]
bdry_constraints = (
    FixBoundaryR(eq=eq, modes=modes_R),
    FixBoundaryZ(eq=eq, modes=modes_Z),
)

## first coil opt to find correct toroidal field needed
# R_modes = eq.surface.R_basis.modes[np.max(np.abs(eq.surface.R_basis.modes), 1) > 0, :]
# Z_modes = eq.surface.Z_basis.modes[np.max(np.abs(eq.surface.Z_basis.modes), 1) > 0, :]
# bdry_constraints = (
#    FixBoundaryR(eq=eq, modes=R_modes),
#    FixBoundaryZ(eq=eq, modes=Z_modes),
# )


constraints = (
    ForceBalance(eq=eq),
    FixPsi(eq=eq),
    FixPressure(eq=eq),
    FixCurrent(eq=eq),
    # FixIota(eq=eq),
    # FixParameters(eq),
    # FixParameters(field,[{"r_n":False,"center": [0, 1],"normal":False}, {"r_n":False,"center": [0, 1],"normal":False},{"normal":True, "center":True}, {},\
    FixParameters(
        field,
        [
            {"r_n": True, "center": [0, 1, 2], "normal": False},
            {"r_n": True, "center": [0, 1, 2], "normal": False},
            {"normal": True, "center": True},
            {},
            {"r_n": True, "center": [0, 1, 2], "normal": False, "current": True},
            {"r_n": True, "center": [0, 1, 2], "normal": False, "current": True},
        ],
    ),
    # FixParameters(field,[{"normal":True}, {}]),
    # FixSumCoilCurrent(fielHYd,indices=[True,True],target = G_tot/n_coils),
    FixSumCoilCurrent(
        field,
        indices=[False, False, True, False, False, False],
        target=-sign0 * (G_tot) / n_coils,
    ),
    FixSumCoilCurrent(
        field, indices=[True, True, False, False, False, False], target=-sign0 * 65000
    ),
    # FixSumCoilCurrent(field,indices=[False, False, False, True, False, False],target = sign0 * 5000),
)
# source_grid = LinearGrid(rho=np.array([1.0]), M=int(1.5*eq.M_grid), N=int(1.5*eq.N_grid), NFP=eq.NFP, sym=False)
# objective = ObjectiveFunction((BoundaryError(eq, field, source_grid=source_grid, eval_grid=eval_grid, deriv_mode="fwd", weight=50),
# objective = ObjectiveFunction((BoundaryError(eq, field, deriv_mode="fwd", weight=40, field_fixed=False),
#                               CoilSetMinDistance(field,bounds=(0.02,100),grid=LinearGrid(N=100), weight=10, deriv_mode="fwd"),
#                               CoilLength(field, bounds=(0, 2 * np.pi * (1.35)), normalize_target=True, weight=5, grid=coil_grid, deriv_mode="fwd"),
#                               PlasmaCoilSetMinDistance(eq, field, bounds=(0.1, np.inf), normalize_target=True, coil_grid=coil_grid, eq_fixed=True, weight=1000),
#                               LinkingCurrentConsistency(eq=eq, coil=field, eq_fixed=False),
#                              ), deriv_mode="batched")
# LinkingCurrentConsistency(eq=eq, coil=field, eq_fixed=True),

# Fails giving NaN optimality
# objective = ObjectiveFunction((BoundaryError(eq, field, weight=40),
#                               CoilSetMinDistance(field,bounds=(0.08,100),grid=LinearGrid(N=100), weight=10),
#                               CoilLength(field, bounds=(0, 2 * np.pi * (1.3)), normalize_target=True, weight=5, grid=coil_grid),
#                               PlasmaCoilSetMinDistance(eq, field, bounds=(0.05, np.inf), normalize_target=True, coil_grid=coil_grid, eq_fixed=True, weight=100),
#                              ), deriv_mode="batched")

# Passes!
len1 = 2 * np.pi * 1.45
len2 = 2 * np.pi * 1.05

dist1 = np.inf
dist2 = 0.1

objective = ObjectiveFunction(
    (
        BoundaryError(
            eq,
            field,
            deriv_mode="fwd",
            weight=60,
            bs_chunk_size=10,
            B_plasma_chunk_size=10,
        ),
        CoilSetMinDistance(
            field,
            bounds=(0.1, 100),
            grid=LinearGrid(N=100),
            weight=20,
            deriv_mode="fwd",
        ),
        CoilLength(
            field,
            bounds=(
                0,
                [len1, len1, len1, len2, len1, len1],
            ),
            normalize_target=True,
            weight=5,
            grid=coil_grid,
            deriv_mode="fwd",
        ),
        PlasmaCoilSetMinDistance(
            eq,
            field,
            bounds=(0.04, np.inf),
            normalize_target=True,
            coil_grid=coil_grid,
            eq_fixed=True,
            weight=500,
        ),
        # PlasmaCoilSetDistanceBound(eq, field, mode="bound", bounds=(0.04, [dist1, dist1, dist1, dist2, dist1, dist1],), normalize_target=True, coil_grid=coil_grid, eq_fixed=True, weight=500),
        # CoilBounds(field, bounds=(0.04, [dist1, dist1, dist1, dist1, dist1, dist1],), normalize_target=True, grid=coil_grid, weight=[0, 0, 0, 1, 0, 0]),
        CoilBounds(
            field,
            bounds=(0.0, 0.01),
            normalize_target=True,
            grid=coil_grid,
            weight=[0, 0, 0, 10, 0, 0],
        ),
    ),
    deriv_mode="batched",
)

optimizer = Optimizer("proximal-lsq-exact")
# optimizer = Optimizer("lsq-exact")
# optimizer = Optimizer("proximal-lsq-auglag")
[eq, field_opt], out = optimizer.optimize(
    [eq, field],
    objective,
    constraints + bdry_constraints,
    verbose=3,
    options={},
    copy=True,
    ftol=1e-10,
    xtol=1e-10,
    gtol=1e-10,
    maxiter=80,
)

print(
    field_opt[0].current,
    field_opt[1].current,
    field_opt[2].current,
    field_opt[3].current,
    field_opt[4].current,
)
eq.save("eq_final3_5kA.h5")
field_opt.save("field_opt_final3_5kA.h5")
