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

def vmec_to_boozer(wout_filename, field, s_vmec, theta_vmec, phi_vmec):

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

    def vartheta_vmec(s, theta_vmec, phi_vmec):
        '''Compute vartheta from VMEC data'''
        lmns = np.zeros((ns, mnmax))
        for jmn in range(mnmax):
            lmns[:, jmn] = lmns_splines[jmn](s_vmec)

        angle = xm * theta_vmec - xn * phi_vmec
        sinangle = np.sin(angle)

        lambd = np.sum(lmns * sinangle)
        vartheta = theta_vmec + lambd
        return vartheta 

    '''Perform root solve to obtain start.dat data in Boozer coordinates'''
    def vartheta_phi_vmec(s,theta_b,zeta_b):
        points = np.zeros((1,3))
        points[:,0] = s
        points[:,1] = theta_b
        points[:,2] = zeta_b
        field.set_points(points)
        nu = field.nu()[0,0]
        iota = field.iota()[0,0]
        vartheta = theta_b - iota*nu
        phi = zeta_b - nu
        return vartheta, phi

    def func_root(x,s,vartheta_target,phi_target):
        theta_b = x[0]
        zeta_b = x[1]
        vartheta, phi = vartheta_phi_vmec(s,theta_b,zeta_b)
        return [vartheta - vartheta_target, phi - phi_target]

    theta_b = []
    zeta_b = []
    for i in range(len(s_vmec)):
        vartheta = vartheta_vmec(s_vmec[i], theta_vmec[i], phi_vmec[i])
        print('vartheta')
        sol = root(func_root, [vartheta, phi_vmec[i]], args=(s_vmec[i], vartheta, phi_vmec[i]), method='hybr')
        theta_b.append(sol.x[0])
        zeta_b.append(sol.x[1])
    return theta_b, zeta_b  # Returns (theta_b, zeta_b) in Boozer coordinates

# Returns (theta_vmec, phi_vmec) from Boozer coordinates (s, theta_b, zeta_b)
def boozer_to_vmec(wout_filename,field,s,theta_b,zeta_b):

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

    def vartheta_vmec(s, theta_vmec, phi_vmec):
        '''Compute vartheta from VMEC data'''
        lmns = np.zeros((ns, mnmax))
        for jmn in range(mnmax):
            lmns[:, jmn] = lmns_splines[jmn](s)

        angle = xm * theta_vmec - xn * phi_vmec
        sinangle = np.sin(angle)

        lambd = np.sum(lmns * sinangle)
        vartheta = theta_vmec + lambd
        return vartheta 

    # Compute PEST angles from Boozer coordinates
    def vartheta_phi_vmec(s, theta_b, zeta_b):
        points = np.zeros((1, 3))
        points[:, 0] = s
        points[:, 1] = theta_b
        points[:, 2] = zeta_b
        field.set_points(points)
        nu = field.nu()[0, 0]
        iota = field.iota()[0, 0]
        vartheta = theta_b - iota * nu
        phi = zeta_b - nu
        return vartheta, phi

    def func_root(x, s, vartheta_boozer, zeta_boozer):
        theta_vmec = x[0]
        # Compute PEST angles from desired Boozer coordinates
        vartheta_target, phi_target = vartheta_phi_vmec(s, vartheta_boozer, zeta_boozer)
        # Compute PEST angles from VMEC coordinates
        vartheta = vartheta_vmec(s, theta_vmec, phi_target)
        return [vartheta - vartheta_target]

    theta_vmec = np.zeros_like(s)
    phi_vmec = np.zeros_like(s)
    for i in range(len(s)):
        theta_vmec[i] = root(func_root, [theta_b[i]], args=(s[i], theta_b[i], zeta_b[i]), method='hybr').x[0]
        vartheta, phi_vmec[i] = vartheta_phi_vmec(s[i], theta_b[i], zeta_b[i])
    return theta_vmec, phi_vmec