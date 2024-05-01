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
from desc.objectives.linear_objectives import (
    SecondBoundaryRSelfConsistency,
    SecondBoundaryZSelfConsistency,
    SecondBoundaryLambdaSelfConsistency,
)

import datetime
import matplotlib.pyplot as plt
import os
import numpy as np

filenames = ["NCSX", "ATF", "ARIES-CS", "HELIOTRON"]
# Get the name of the current script
script_name = os.path.basename(__file__)

# Remove the file extension
name = os.path.splitext(script_name)[0]
date = datetime.date.today()

for filename in filenames:
    eq = get(filename)

    zeta = 0
    zeta2 = np.pi
    maxiter = 5000

    # Take second surface to be fixed
    surface2 = eq.get_surface_at(zeta=zeta2/eq.NFP)
    surface2.change_resolution(eq.L,eq.M,eq.N)

    eq_poin = eq.set_poincare_equilibrium(zeta=zeta)
    eq_poin.change_resolution(eq_poin.L,eq_poin.M,eq.N) 

    Lb2_lmn, Lb2_basis = FourierZernike_to_FourierZernike_no_N_modes(eq.L_lmn, eq.L_basis, zeta=zeta2)

    constraints = get_fixed_boundary_constraints(eq=eq_poin,poincare_lambda=True, zeta=zeta)
    constraints += (
        SecondBoundaryRSelfConsistency(eq=eq_poin, zeta=zeta2, surface=surface2), 
        SecondBoundaryZSelfConsistency(eq=eq_poin, zeta=zeta2, surface=surface2),
        SecondBoundaryLambdaSelfConsistency(eq=eq_poin, zeta=zeta2, Lb2_lmn=Lb2_lmn, Lb2_basis=Lb2_basis),
        )
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
    plot_comparison(eqs=[eq,eq_poin],labels=['LCFS',f'2 BC Poincare']);

    img_name = f"yge/figure/{name}-{date}-{filename}-maxiter-{maxiter}.pdf"
    out_name = f"yge/output/{name}-{date}-{filename}-maxiter-{maxiter}.h5"

    plt.savefig(img_name, format="pdf", dpi=1200)
    eq_poin.save(out_name)
