from simsopt.field.boozermagneticfield import BoozerRadialInterpolant
from scipy.optimize import root
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy as np 
from scipy.io import netcdf_file

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    verbose = comm.rank == 0
    comm_size = comm.size
except ImportError:
    comm = None
    verbose = True
    comm_size = 1

boozmn_filename = '../booz_xform/boozmn_LandremanPaul2021_QA_reactorScale_lowres_reference.nc'
wout_filename = '../booz_xform/wout_LandremanPaul2021_QA_reactorScale_lowres_reference.nc'
order = 3

## Initial condition in VMEC coordinates
s_init = np.array([0.5])
theta_init = np.array([0.5])
phi_init = np.array([0.5])
npoints = len(s_init)

'''Load VMEC and booz_xform data'''
f = netcdf_file(wout_filename, mmap=False)
lmns = f.variables['lmns'][()]
mnmax = f.variables['mnmax'][()]
ns = f.variables['ns'][()]
xm = f.variables['xm'][()]
xn = f.variables['xn'][()]
f.close()
s_full_grid = np.linspace(0, 1, ns)
s_half_grid = (s_full_grid[0:-1] + s_full_grid[1::]) / 2.0

lmns_splines = []
for jmn in range(mnmax):
    lmns_splines.append(InterpolatedUnivariateSpline(s_half_grid, lmns[1::, jmn]))

bri = BoozerRadialInterpolant(boozmn_filename,order=3,no_K=True)

'''Compute vartheta from VMEC data'''
lmns = np.zeros((ns, mnmax))
for jmn in range(mnmax):
    lmns[:, jmn] = lmns_splines[jmn](s_init)

angle = xm[:, None] * theta_init[None, :] - xn[:, None] * phi_init[None, :]
sinangle = np.sin(angle)

lambd = np.einsum('ij,ji->i', lmns, sinangle)
vartheta = theta_init + lambd

'''Perform root solve to obtain start.dat data in Boozer coordinates'''
def vartheta_phi_vmec(s,theta_b,zeta_b):
    points = np.zeros((1,3))
    points[:,0] = s
    points[:,1] = theta_b
    points[:,2] = zeta_b
    bri.set_points(points)
    nu = bri.nu()[0,0]
    iota = bri.iota()[0,0]
    vartheta = theta_b - iota*nu
    phi = zeta_b - nu
    return vartheta, phi

def func_root(x,s,vartheta_target,phi_target):
    theta_b = x[0]
    zeta_b = x[1]
    vartheta, phi = vartheta_phi_vmec(s,theta_b,zeta_b)
    return [vartheta - vartheta_target,phi - phi_target]

theta_b = []
zeta_b = []
for i in range(npoints):
    sol = root(func_root, [vartheta[i],phi_init[i]], args=(s_init[i],vartheta[i],phi_init[i]))
    if (sol.success):
        theta_b.append(sol.x[0])
        zeta_b.append(sol.x[1])
    else:
        theta_b_out = sol.x[0]
        zeta_b_out = sol.x[1]
        vartheta_out, phi_out = vartheta_phi_vmec(s_init[i],theta_b_out,zeta_b_out)
        print('vartheta: ',vartheta[i])
        print('vartheta_out: ',vartheta_out)
        print('phi: ',phi_init[i])
        print('phi_out: ',phi_out)

np.savetxt('s_booz.txt', np.array(s_init))
np.savetxt('theta_vmec.txt', np.array(theta_init))
np.savetxt('phi_vmec.txt', np.array(phi_init))
np.savetxt('vartheta_booz.txt', np.array(theta_b))
np.savetxt('zeta_booz.txt', np.array(zeta_b))
