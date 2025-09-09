import sys
import numpy as np
import time
from boozer_to_vmec import boozer_to_vmec

from simsopt.field.boozermagneticfield import (
    BoozerRadialInterpolant,
    InterpolatedBoozerField,
)
from simsopt.field.tracing import (
    trace_particles_boozer,
    MaxToroidalFluxStoppingCriterion,
)
from simsopt.util.constants import (
    ALPHA_PARTICLE_MASS,
    ALPHA_PARTICLE_CHARGE,
    FUSION_ALPHA_PARTICLE_ENERGY,
)
from simsopt.util.functions import proc0_print

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None

boozmn_filename = "../../booz_xform/boozmn_LandremanPaul2021_QA_reactorScale_lowres_reference.nc"
wout_filename = "../../booz_xform/wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc"
order = 3  # Order for radial interpolation
degree = 3  # Degree for 3d interpolation
Ekin = FUSION_ALPHA_PARTICLE_ENERGY
mass = ALPHA_PARTICLE_MASS
charge = ALPHA_PARTICLE_CHARGE
tmax = 1e-3

# Initialize single trapped particle 
v0 = np.sqrt(2 * Ekin / mass)  # Initial speed
vpar_init = [0.0 * v0]
s0 = np.loadtxt('../../initial_condition/s_booz.txt')
theta0 = np.loadtxt('../../initial_condition/vartheta_booz.txt')
zeta0 = np.loadtxt('../../initial_condition/zeta_booz.txt')
points = np.zeros((1, 3))
points[0, 0] = s0  # s values
points[0, 1] = theta0  # theta values
points[0, 2] = zeta0  # zeta values

## Setup radial interpolation
bri = BoozerRadialInterpolant(boozmn_filename, order, no_K=True, comm=comm)
nfp = bri.nfp

sys.stdout = open(f"stdout_trajectory_resolution_scan_trapped.txt", "a", buffering=1)

for resolution in [64, 80, 96]:

    time1 = time.time()

    ns_interp = resolution
    ntheta_interp = resolution
    nzeta_interp = resolution

    ## Setup 3d interpolation
    field = InterpolatedBoozerField(
        bri,
        degree,
        ns_interp=ns_interp,
        ntheta_interp=ntheta_interp,
        nzeta_interp=nzeta_interp,
    )

    time2 = time.time()
    proc0_print('Time to set up field: ', time2 - time1, ' seconds')

    for tol in [1e-6, 1e-7, 1e-8, 1e-9, 1e-10]:

        reltol = tol
        abstol = tol

        proc0_print('Running firm3d with tol =', tol, ', tmax =', tmax, ', resolution =', resolution)

        time2 = time.time()

        ## Trace alpha particle in Boozer coordinates until it hits the s = 1 surface
        ## Set forget_exact_path=False to save the trajectory information.
        ## Set the dt_save parameter to the time interval for trajectory data
        ## to be saved.
        traj_booz, res_hits = trace_particles_boozer(
            field,
            points,
            vpar_init,
            tmax=tmax,
            mass=mass,
            charge=charge,
            Ekin=Ekin,
            stopping_criteria=[MaxToroidalFluxStoppingCriterion(1.0)],
            forget_exact_path=False,
            dt_save=1e-7,
            abstol=abstol,
            reltol=reltol,
        )

        time3 = time.time()

        proc0_print('Time for integration: ', time3 - time2, ' seconds')

        np.savetxt(f'trajectory_data_tol_{tol}_resolution_{resolution}_tmax_{tmax}_trapped.txt', traj_booz[0])

        theta_vmec, phi_vmec = boozer_to_vmec(wout_filename, field, traj_booz[0][:, 1], traj_booz[0][:, 2], traj_booz[0][:, 3])

        np.savetxt(f'trajectory_data_vmec_tol_{tol}_resolution_{resolution}_tmax_{tmax}_trapped.txt', np.column_stack((traj_booz[0][:, 0], traj_booz[0][:,1], theta_vmec, phi_vmec, traj_booz[0][:, 4])))

