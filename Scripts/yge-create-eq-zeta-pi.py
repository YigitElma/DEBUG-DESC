"""multiple equilibruim solver script for cluster."""
from desc import set_device
set_device("gpu")
from desc.basis import (
    FourierZernike_to_FourierZernike_no_N_modes,
)
from desc.equilibrium import Equilibrium
from desc.examples import get
from desc.objectives import ForceBalance, ObjectiveFunction
from desc.objectives.getters import get_fixed_boundary_constraints
from desc.plotting import plot_comparison
import datetime
import matplotlib.pyplot as plt
import os
import numpy as np


filenames = ["WISTELL-A", "W7X", "NCSX", "ATF", "ARIES-CS", "W7-X", "HELIOTRON"]
# Get the name of the current script
script_name = os.path.basename(__file__)

# Remove the file extension
name = os.path.splitext(script_name)[0]
date = datetime.date.today()

for filename in filenames:
    eq = get(filename)

    n = eq.N
    k = eq.N_grid
    zeta = np.pi
    maxiter = 5000

    eq_poin = eq.set_poincare_equilibrium(zeta=zeta)
    eq_poin.change_resolution(eq_poin.L, eq_poin.M, n)
    eq_poin.N_grid = k
    constraints = get_fixed_boundary_constraints(eq=eq_poin, poincare_lambda=True, zeta=zeta)
    objective = ObjectiveFunction(ForceBalance(eq_poin))

    eq_poin.solve(
        verbose=3,
        ftol=0,
        objective=objective,
        constraints=constraints,
        maxiter=maxiter,
        xtol=0,
        zeta=zeta,
    )
    plot_comparison(
        eqs=[eq, eq_poin], labels=["LCFS", f"Poincare BC N={n} N_grid={k}"]
    )

    img_name = f"yge/figure/{name}-{date}-{filename}-zeta-{zeta}-maxiter-{maxiter}.pdf"
    out_name = f"yge/output/{name}-{date}-{filename}-zeta-{zeta}-maxiter-{maxiter}.h5"

    plt.savefig(img_name, format="pdf", dpi=1200)
    eq_poin.save(out_name)
