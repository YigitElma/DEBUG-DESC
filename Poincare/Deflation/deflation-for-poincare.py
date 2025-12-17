"""Solve the deflated eq problem with Poincare to find if the original equilibrium is one of the deflated local minimum."""

import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../../"))

from desc import set_device

set_device("gpu")

import numpy as np
import matplotlib.pyplot as plt

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
from desc.plotting import *
from desc.optimize import *
from desc.compat import *
from desc.utils import *

from desc.objectives.objective_funs import _Objective
from desc.compute.utils import _compute as compute_fun

print_backend_info()


name = str(sys.argv[1])
LMN = int(sys.argv[2])
num_deflations = int(sys.argv[3])
sigma = 100
power = 2


class ForceBalanceDeflated(_Objective):
    """Radial and helical MHD force balance.

    Given force densities:

    Fᵨ = √g (J^θ B^ζ - J^ζ B^θ) - ∇ p

    Fₕₑₗᵢ √g J^ρ

    and helical basis vector:

    𝐞ʰᵉˡⁱ = B^ζ ∇ θ - B^θ ∇ ζ

    Minimizes the magnitude of the forces:

    fᵨ = Fᵨ ||∇ ρ|| dV  (N)

    fₕₑₗᵢ = Fₕₑₗᵢ ||𝐞ʰᵉˡⁱ|| dV  (N)

    Parameters
    ----------
    eq : Equilibrium
        Equilibrium that will be optimized to satisfy the Objective.
    eqs: list of Equilibrium
        list of Equilibrium objects to use in deflation operator.
    sigma: float, optional
        sigma term in deflation operator
    power: float, optional
        power parameter in deflation operator.
    grid : Grid, optional
        Collocation grid containing the nodes to evaluate at.
        Defaults to ``ConcentricGrid(eq.L_grid, eq.M_grid, eq.N_grid)``
    """

    __doc__ = __doc__.rstrip() + collect_docs(
        target_default="``target=0``.", bounds_default="``target=0``."
    )
    _static_attrs = _Objective._static_attrs + ["_params_to_deflate_with"]

    _equilibrium = True
    _coordinates = "rtz"
    _units = "(N)"
    _print_value_fmt = "Force error: "

    def __init__(
        self,
        eq,
        eqs,
        sigma=0.05,
        power=2,
        target=None,
        bounds=None,
        weight=1,
        normalize=True,
        normalize_target=True,
        loss_function=None,
        deriv_mode="auto",
        grid=None,
        name="force",
        jac_chunk_size=None,
    ):
        if target is None and bounds is None:
            target = 0
        self._grid = grid
        self._eqs = eqs
        self._sigma = sigma
        self._power = power
        super().__init__(
            things=eq,
            target=target,
            bounds=bounds,
            weight=weight,
            normalize=normalize,
            normalize_target=normalize_target,
            loss_function=loss_function,
            deriv_mode=deriv_mode,
            name=name,
            jac_chunk_size=jac_chunk_size,
        )

    def build(self, use_jit=True, verbose=1):
        """Build constant arrays.

        Parameters
        ----------
        use_jit : bool, optional
            Whether to just-in-time compile the objective and derivatives.
        verbose : int, optional
            Level of output.

        """
        eq = self.things[0]
        if self._grid is None:
            grid = ConcentricGrid(
                L=eq.L_grid,
                M=eq.M_grid,
                N=eq.N_grid,
                NFP=eq.NFP,
                sym=eq.sym,
                axis=False,
            )
        else:
            grid = self._grid

        self._dim_f = 2 * grid.num_nodes
        self._data_keys = [
            "F_rho",
            "|grad(rho)|",
            "sqrt(g)",
            "F_helical",
            "|e^helical*sqrt(g)|",
        ]

        timer = Timer()
        if verbose > 0:
            print("Precomputing transforms")
        timer.start("Precomputing transforms")

        profiles = get_profiles(self._data_keys, obj=eq, grid=grid)
        transforms = get_transforms(self._data_keys, obj=eq, grid=grid)

        self._constants = {
            "transforms": transforms,
            "profiles": profiles,
        }
        print("Building R to Rb matrix")
        Robj = BoundaryRSelfConsistency(eq)
        Robj.build()
        self.Ar = Robj._A

        print("Building Z to Zb matrix")
        Zobj = BoundaryZSelfConsistency(eq)
        Zobj.build()
        self.Az = Zobj._A

        timer.stop("Precomputing transforms")
        if verbose > 1:
            timer.disp("Precomputing transforms")

        if self._normalize:
            scales = compute_scaling_factors(eq)
            self._normalization = scales["f"]

        super().build(use_jit=use_jit, verbose=verbose)

    def compute(self, params, constants=None):
        """Compute MHD force balance errors.

        Parameters
        ----------
        params : dict
            Dictionary of equilibrium degrees of freedom, eg Equilibrium.params_dict
        constants : dict
            Dictionary of constant data, eg transforms, profiles etc. Defaults to
            self.constants

        Returns
        -------
        f : ndarray
            MHD force balance error at each node (N).

        """
        if constants is None:
            constants = self.constants
        data = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self._data_keys,
            params=params,
            transforms=constants["transforms"],
            profiles=constants["profiles"],
        )
        fr = data["F_rho"] * data["|grad(rho)|"] * data["sqrt(g)"]
        fb = data["F_helical"] * data["|e^helical*sqrt(g)|"]
        keys = ["R_lmn", "Z_lmn"]
        As = [self.Ar, self.Az]
        diffs = [
            jnp.concatenate(
                [A @ (params[key] - eq.params_dict[key]) for A, key in zip(As, keys)]
            )
            for eq in self._eqs
        ]
        diffs = jnp.vstack(diffs)
        deflation_parameter = jnp.prod(
            1 / jnp.linalg.norm(diffs, axis=1) ** self._power + self._sigma
        )
        return jnp.concatenate([fr, fb]) * deflation_parameter


def set_poincare_equilibrium(eq):
    eq_poincare = Equilibrium(
        xsection=eq.get_surface_at(zeta=0),
        pressure=eq.pressure,
        iota=eq.iota,
        Psi=eq.Psi,  # flux (in Webers) within the last closed flux surface
        NFP=eq.NFP,  # number of field periods
        L=eq.L,  # radial spectral resolution
        M=eq.M,  # poloidal spectral resolution
        N=eq.N,  # toroidal spectral resolution
        L_grid=eq.L_grid,  # real space radial resolution, slightly oversampled
        M_grid=eq.M_grid,  # real space poloidal resolution, slightly oversampled
        N_grid=eq.N_grid,  # real space toroidal resolution
        sym=eq.sym,  # explicitly enforce stellarator symmetry
        spectral_indexing=eq._spectral_indexing,
    )

    eq_poincare.change_resolution(eq.L, eq.M, eq.N)
    eq_poincare.axis = eq_poincare.get_axis()
    eq_poincare.surface = eq_poincare.get_surface_at(rho=1)
    return eq_poincare


try:
    eq0h = desc.io.load(f"{name}_LMN_{LMN}.h5")
except:
    eq0 = get(name)
    eq0h = eq0.copy()
    eq0h.change_resolution(
        L=LMN, M=LMN, N=LMN, L_grid=LMN * 2, M_grid=LMN * 2, N_grid=LMN * 2
    )
    eq0h.solve(
        maxiter=500,
        ftol=1e-3,
        xtol=0,
        gtol=0,
        verbose=3,
        x_scale="ess",
    )
    eq0h.xsection = eq0h.get_surface_at(zeta=0)
    eq0h.save(f"{name}_LMN_{LMN}.h5")
    plot_section(eq0h, "|F|_normalized", log=True)
    plt.savefig(f"f_error_{name}_LMN_{LMN}.png", dpi=500)


try:
    eqp0 = desc.io.load(f"{name}_LMN_{LMN}_poincare.h5")
except:
    eqp0 = eq0h.copy()
    constraints = get_fixed_xsection_constraints(eq=eqp0, fix_lambda=True)
    objective = ObjectiveFunction(ForceBalance(eqp0))

    eqp0.solve(
        verbose=3,
        objective=objective,
        constraints=constraints,
        maxiter=1000,
        ftol=1e-3,
        x_scale="ess",
    )

    eqp0.surface = eqp0.get_surface_at(rho=1)
    eqp0.save(f"{name}_LMN_{LMN}_poincare.h5")
    plot_section(eqp0, "|F|_normalized", log=True)
    plt.savefig(f"f_error_{name}_LMN_{LMN}_poincare.png", dpi=500)

try:
    eqp0_as = desc.io.load(f"{name}_LMN_{LMN}_poincare_as.h5")
except:
    eqp0_as = set_poincare_equilibrium(eq0h)
    constraints = get_fixed_xsection_constraints(eq=eqp0_as, fix_lambda=True)
    objective = ObjectiveFunction(ForceBalance(eqp0_as))

    eqp0_as.solve(
        verbose=3,
        objective=objective,
        constraints=constraints,
        maxiter=1000,
        ftol=1e-3,
        x_scale="ess",
    )

    eqp0_as.surface = eqp0_as.get_surface_at(rho=1)
    eqp0_as.save(f"{name}_LMN_{LMN}_poincare_as.h5")
    plot_section(eqp0_as, "|F|_normalized", log=True)
    plt.savefig(f"f_error_{name}_LMN_{LMN}_poincare_as.png", dpi=500)

dont_goto = [eqp0_as]

for i in range(1, num_deflations + 1):
    try:
        eqd = desc.io.load(f"{name}_LMN_{LMN}_poincare_as_def{i}_sigma_{sigma}.h5")
    except:
        eqd = set_poincare_equilibrium(eq0h)
        constraints = get_fixed_xsection_constraints(eq=eqd, fix_lambda=True)
        objective = ObjectiveFunction(
            ForceBalanceDeflated(eqd, eqs=dont_goto, sigma=sigma, power=power)
        )
        # find some deflated result
        eqd.solve(
            verbose=3,
            objective=objective,
            constraints=constraints,
            maxiter=1000,
            ftol=1e-3,
            x_scale="ess",
        )

        eqd.surface = eqd.get_surface_at(rho=1)
        constraints = get_fixed_xsection_constraints(eq=eqd, fix_lambda=True)
        objective = ObjectiveFunction(ForceBalance(eqd))
        # make sure that it is a Poincare equilibrium
        eqd.solve(
            verbose=3,
            objective=objective,
            constraints=constraints,
            maxiter=1000,
            ftol=1e-3,
            x_scale="ess",
        )
        eqd.surface = eqd.get_surface_at(rho=1)
        eqd.save(f"{name}_LMN_{LMN}_poincare_as_def{i}_sigma_{sigma}.h5")
        plot_section(eqd, "|F|_normalized", log=True)
        plt.savefig(
            f"f_error_{name}_LMN_{LMN}_poincare_as_def{i}_sigma_{sigma}.png", dpi=500
        )

    dont_goto += [eqd]
