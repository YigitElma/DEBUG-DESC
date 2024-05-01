from desc.objectives import (
    ObjectiveFunction,
    ForceBalance
)
from desc.plotting import plot_comparison
from desc.examples import get
from desc.objectives.getters import get_fixed_boundary_constraints


eq = get("HELIOTRON")
for n in range(3,4):
    k = 6
    eq_poin = eq.set_poincare_equilibrium()
    eq_poin.change_resolution(eq_poin.L,eq_poin.M,n) 
    eq_poin.N_grid=k
    constraints = get_fixed_boundary_constraints(eq=eq_poin,poincare_lambda=True)
    objective = ObjectiveFunction(ForceBalance(eq_poin,weight=1))
    
    eq_poin.solve(verbose=3, ftol=0,objective=objective,constraints=constraints,maxiter=400,xtol=0)
    plot_comparison(eqs=[eq,eq_poin],labels=['LCFS',f'poincare BC R* Z* N={n} N_grid={k}']);