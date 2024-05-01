"""multiple equilibruim solver script for cluster."""
from desc import set_device
set_device("gpu")
from desc.basis import FourierZernike_to_FourierZernike_no_N_modes
from desc.equilibrium import Equilibrium
from desc.examples import get
from desc.geometry import ZernikeRZToroidalSection
from desc.objectives import ForceBalance, ObjectiveFunction
from desc.objectives.getters import get_fixed_boundary_constraints
from desc.plotting import plot_comparison
import datetime
import matplotlib.pyplot as plt
import os

def get_eq_poin(eq, zeta=0):
    surface = eq.get_surface_at(zeta=zeta/eq.NFP)
    Lb_lmn, Lb_basis = FourierZernike_to_FourierZernike_no_N_modes(eq.L_lmn, eq.L_basis, zeta)

    eq_poin = Equilibrium(surface=surface,
                 pressure=eq.pressure,
                 iota=eq.iota,
                 Psi=eq.Psi, # flux (in Webers) within the last closed flux surface
                 NFP=eq.NFP, # number of field periods
                 L=eq.L, # radial spectral resolution
                 M=eq.M, # poloidal spectral resolution
                 N=eq.N, # toroidal spectral resolution 
                 L_grid=eq.L_grid, # real space radial resolution, slightly oversampled
                 M_grid=eq.M_grid, # real space poloidal resolution, slightly oversampled
                 N_grid=eq.N_grid, # real space toroidal resolution
                 sym=True, # explicitly enforce stellarator symmetry
                 bdry_mode='poincare',
                 spectral_indexing=eq._spectral_indexing
                )
    eq_poin.L_lmn = (
        Lb_lmn  # initialize the poincare eq with the lambda of the original eq
    )
    eq_poin.axis = eq_poin.get_axis()
    return eq_poin


def get_poin_GS(eq):
    eq_poin = get_eq_poin(eq)
    eq_poin.change_resolution(eq_poin.L,eq_poin.M,eq.N) # must be certain bases are the same before assigning eq.X_lmn
    eq_poin.N_grid=eq_poin.N_grid
    surf=eq_poin.get_surface_at(rho=1)
    
    eq_GS = Equilibrium(surface=surf,
                 pressure=eq_poin.pressure,
                 iota=eq_poin.iota,
                 Psi=eq_poin.Psi, # flux (in Webers) within the last closed flux surface
                 NFP=eq_poin.NFP, # number of field periods
                 L=eq_poin.L, # radial spectral resolution
                 M=eq_poin.M, # poloidal spectral resolution
                 N=eq_poin.N, # toroidal spectral resolution 
                 L_grid=eq_poin.L_grid, # real space radial resolution, slightly oversampled
                 M_grid=eq_poin.M_grid, # real space poloidal resolution, slightly oversampled
                 N_grid=eq_poin.N_grid, # real space toroidal resolution
                 sym=True, # explicitly enforce stellarator symmetry
                 bdry_mode='lcfs',
                 spectral_indexing=eq_poin._spectral_indexing,  
                )
    
    return eq_GS

def get_perturbed_GS_Poincare(eq, eq_GS, eq_poin, step, numstep):
    surface_Poincare = eq_poin.get_surface_at(zeta=0)
    surface_GS = eq_GS.get_surface_at(zeta=0)
    
    Rb_lmn = surface_GS.R_lmn + (surface_Poincare.R_lmn - surface_GS.R_lmn)*step/numstep
    Zb_lmn = surface_GS.Z_lmn + (surface_Poincare.Z_lmn - surface_GS.Z_lmn)*step/numstep
    
    Rb_basis = surface_GS.R_basis
    Zb_basis = surface_GS.Z_basis
    
    L_GS_lmn, L_GS_basis = FourierZernike_to_FourierZernike_no_N_modes(eq_GS.L_lmn, eq_GS.L_basis)
    L_Poin_lmn, L_Poin_basis = FourierZernike_to_FourierZernike_no_N_modes(eq_poin.L_lmn, eq_poin.L_basis)

    Lb_lmn = L_GS_lmn + (L_Poin_lmn - L_GS_lmn)*step/numstep
    # Lb_basis = L_GS_basis
    
    surf = ZernikeRZToroidalSection(
        R_lmn=Rb_lmn,
        modes_R=Rb_basis.modes[:, :2].astype(int),
        Z_lmn=Zb_lmn,
        modes_Z=Zb_basis.modes[:, :2].astype(int),
        spectral_indexing=eq_poin._spectral_indexing,
    )

    eq.bdry_mode = 'poincare'
    eq.surface = surf
    eq.L_lmn = Lb_lmn
    
    return eq

# Get the name of the current script
script_name = os.path.basename(__file__)

# Remove the file extension
name = os.path.splitext(script_name)[0]

filename = "HELIOTRON"
NumStep = 10
maxiter = 500

eq = get(filename)
eq_GS = get_poin_GS(eq)
eq_poin = get_eq_poin(eq)

constraints = get_fixed_boundary_constraints(eq=eq_GS)
objective = ObjectiveFunction(ForceBalance(eq_GS))
eq_GS.solve(verbose=3, objective=objective, constraints=constraints, maxiter=maxiter, ftol=0, xtol=0, gtol=0)

eq_step = eq_GS.copy()
for step in range(1,NumStep+1):
    eq_prev = eq_step.copy()
    eq_step = get_perturbed_GS_Poincare(eq=eq_step,eq_GS=eq_GS,eq_poin=eq_poin,step=step,numstep=NumStep)
    constraints = get_fixed_boundary_constraints(eq=eq_step, poincare_lambda=True)
    objective = ObjectiveFunction(ForceBalance(eq_step))

    eq_step.solve(verbose=3, objective=objective, constraints=constraints, maxiter=maxiter, ftol=0, xtol=0, gtol=0)
plot_comparison(eqs=[eq,eq_prev,eq_step],labels=['LCFS',f'GS-Continuation step: {step-1}',f'GS-Continuation step: {step}']);

date = datetime.date.today()
img_name = f"yge/figure/{name}-{date}-{filename}-maxiter-{maxiter}-step-{NumStep}.pdf"
out_name = f"yge/output/{name}-{date}-{filename}-maxiter-{maxiter}-step-{NumStep}.h5"

plt.savefig(img_name, format="pdf", dpi=1200)
eq_poin.save(out_name)
