
import os
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])


import numpy as np
from simsopt.field.boozermagneticfield import BoozerRadialInterpolant

def boozer_to_cylindrical(field, s, theta, zeta):
    if not isinstance(s, np.ndarray):
        s = np.asarray(s)
    if not isinstance(theta, np.ndarray):
        theta = np.asarray(theta)
    if not isinstance(zeta, np.ndarray):
        zeta = np.asarray(zeta)

    # Handle scalar inputs - return scalars if any input is a scalar
    input_scalar = np.isscalar(s) or np.isscalar(theta) or np.isscalar(zeta)
    npoints = s.size

    points = np.zeros((npoints, 3))
    points[:, 0] = s.flatten()
    points[:, 1] = theta.flatten()
    points[:, 2] = zeta.flatten()

    field.set_points(points)

    R = field.R()[:, 0]
    Z = field.Z()[:, 0]
    nu = field.nu()[:, 0]
    phi = zeta - nu

    # Return scalars for scalar inputs, arrays for array inputs
    if input_scalar:
        return R[0], phi[0], Z[0]
    else:
        return R, phi, Z
    

boozmn_filename = "booz_xform/boozmn_LandremanPaul2021_QH_reactorScale_lowres_reference.nc"
field = BoozerRadialInterpolant(boozmn_filename, order=3, no_K=True)

N = int(10000/20) # number of points per job
start = int(idx*N)
end = int((idx+1)*N)
traj_booz = np.loadtxt("run_firm3d/precise_QH/trajectory_data_tol_1e-10_resolution_96_tmax_0.001_trapped.txt")
R, phi, Z = boozer_to_cylindrical(field, traj_booz[start:end, 1], traj_booz[start:end, 2], traj_booz[start:end, 3])

np.savetxt(f"converted/R_job{idx}_booz.txt", R)
np.savetxt(f"converted/Z_job{idx}_booz.txt", Z)
np.savetxt(f"converted/phi_job{idx}_booz.txt", phi)
