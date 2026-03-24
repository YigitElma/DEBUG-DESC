import desc.io
import sys
from scipy.interpolate import interp1d
import numpy as np
from desc.grid import Grid
from scipy.constants import mu_0
import toml


def get_gx_arrays(
    zeta, bmag, grho, gradpar, gds2, gds21, gds22, gbdrift, gbdrift0, cvdrift, cvdrift0
):
    dzeta = zeta[1] - zeta[0]
    dzeta_pi = np.pi / nzgrid
    index_of_middle = nzgrid

    gradpar_half_grid = np.zeros(2 * nzgrid)
    temp_grid = np.zeros(2 * nzgrid + 1)
    z_on_theta_grid = np.zeros(2 * nzgrid + 1)
    uniform_zgrid = np.zeros(2 * nzgrid + 1)

    gradpar_temp = np.copy(gradpar)
    for i in range(2 * nzgrid - 1):
        gradpar_half_grid[i] = 0.5 * (
            np.abs(gradpar_temp[i]) + np.abs(gradpar_temp[i + 1])
        )
    gradpar_half_grid[2 * nzgrid - 1] = gradpar_half_grid[0]

    for i in range(2 * nzgrid):
        temp_grid[i + 1] = temp_grid[i] + dzeta * (1 / np.abs(gradpar_half_grid[i]))

    for i in range(2 * nzgrid + 1):
        z_on_theta_grid[i] = temp_grid[i] - temp_grid[index_of_middle]
    desired_gradpar = np.pi / np.abs(z_on_theta_grid[0])

    for i in range(2 * nzgrid + 1):
        z_on_theta_grid[i] = z_on_theta_grid[i] * desired_gradpar
        gradpar_temp[i] = desired_gradpar

    for i in range(2 * nzgrid + 1):
        uniform_zgrid[i] = z_on_theta_grid[0] + i * dzeta_pi

    final_theta_grid = uniform_zgrid

    bmag_gx = interp_to_new_grid(bmag, z_on_theta_grid, uniform_zgrid)
    grho_gx = interp_to_new_grid(grho, z_on_theta_grid, uniform_zgrid)
    gds2_gx = interp_to_new_grid(gds2, z_on_theta_grid, uniform_zgrid)
    gds21_gx = interp_to_new_grid(gds21, z_on_theta_grid, uniform_zgrid)
    gds22_gx = interp_to_new_grid(gds22, z_on_theta_grid, uniform_zgrid)
    gbdrift_gx = interp_to_new_grid(gbdrift, z_on_theta_grid, uniform_zgrid)
    gbdrift0_gx = interp_to_new_grid(gbdrift0, z_on_theta_grid, uniform_zgrid)
    cvdrift_gx = interp_to_new_grid(cvdrift, z_on_theta_grid, uniform_zgrid)
    cvdrift0_gx = interp_to_new_grid(cvdrift0, z_on_theta_grid, uniform_zgrid)
    gradpar_gx = gradpar_temp

    return (
        uniform_zgrid,
        bmag_gx,
        grho_gx,
        gradpar_gx,
        gds2_gx,
        gds21_gx,
        gds22_gx,
        gbdrift_gx,
        gbdrift0_gx,
        cvdrift_gx,
        cvdrift0_gx,
    )


def interp_to_new_grid(geo_array, zgrid, uniform_grid):
    geo_array_gx = np.zeros(len(geo_array))
    f = interp1d(zgrid, geo_array, kind="cubic")
    for i in range(len(uniform_grid) - 1):
        if uniform_grid[i] > zgrid[-1]:
            geo_array_gx[i] = geo_array_gx[i - 1]
        else:
            geo_array_gx[i] = f(np.round(uniform_grid[i], 5))

    geo_array_gx[-1] = geo_array[-1]

    return geo_array_gx


# read parameters from input file
input_file = sys.argv[1]
if len(sys.argv) > 2:
    stem = input_file.split(".")[0]
    eikfile = sys.argv[2]
    eiknc = eikfile[-8:] + ".eiknc.nc"
else:
    stem = input_file.split(".")[0]
    eikfile = stem + ".eik.out"
    eiknc = stem + ".eiknc.nc"

f = toml.load(input_file)

# note: this script assumes irho=2, so rhoc = r/a
nzgrid = int(f["Dimensions"]["ntheta"] / 2)
npol = f["Geometry"]["npol"]
rho = f["Geometry"]["rhotor"]
path = f["Geometry"]["geo_file"]

eq = desc.io.load(path)  # loads desc output
if hasattr(eq, "__len__"):
    eq = eq[-1]

eq_keys = ["iota", "iota_r", "a", "rho", "psi"]

flux_tube_keys = [
    "B",
    "|B|",
    "lambda",
    "lambda_r",
    "lambda_t",
    "lambda_z",
    "|grad(rho)|",
    "g^rr",
    "g^tt",
    "g^zz",
    "g^rt",
    "g^rz",
    "g^tz",
    "g_tz",
    "g_tt",
    "g_zz",
    "B_theta",
    "B_zeta",
    "B_rho",
    "|B|_t",
    "|B|_z",
    "|B|_r",
    "B^theta",
    "B^zeta_r",
    "B^theta_r",
    "B^zeta",
    "e_theta",
    "e_theta_r",
    "e_zeta_r",
    "e_zeta",
    "p_r",
    "grad(psi)",
]

data_eq = eq.compute(eq_keys)

psi = rho**2

fi = interp1d(data_eq["rho"], data_eq["iota"])
fs = interp1d(data_eq["rho"], data_eq["iota_r"])

iotas = fi(rho)
shears = fs(rho)

zeta_center = f["Geometry"].get("zeta_center", 0.0)
alpha = f["Geometry"].get("alpha", -iotas * zeta_center)
shift_grad_alpha = f["Geometry"].get("shift_grad_alpha", True)

if not shift_grad_alpha:
    zeta_center = 0.0

zeta = np.linspace(
    (-np.pi * npol - alpha) / np.abs(iotas),
    (np.pi * npol - alpha) / np.abs(iotas),
    2 * nzgrid + 1,
)
iota = iotas * np.ones(len(zeta))
shear = shears * np.ones(len(zeta))
thetas = iotas / np.abs(iotas) * alpha * np.ones(len(zeta)) + iota * zeta

rhoa = rho * np.ones(len(zeta))
c = np.vstack([rhoa, thetas, zeta]).T
coords = eq.compute_theta_coords(c, tol=1e-10, maxiter=50)
grid = Grid(coords)

data = eq.compute(flux_tube_keys, grid=grid)

psib = data_eq["psi"][-1]
# normalizations
Lref = data_eq["a"]
Bref = 2 * np.abs(psib) / Lref**2
# calculate bmag
modB = data["|B|"]
bmag = modB / Bref

# calculate gradpar and grho
gradpar = Lref * data["B^zeta"] / modB
grho = data["|grad(rho)|"] * Lref

# calculate grad_psi and grad_alpha
grad_psi = 2 * psib * rho
lmbda = data["lambda"]
lmbda_r = data["lambda_r"]
lmbda_t = data["lambda_t"]
lmbda_z = data["lambda_z"]


grad_alpha_r = lmbda_r - (zeta - zeta_center) * shear
grad_alpha_t = 1 + lmbda_t
grad_alpha_z = -iota + lmbda_z

grad_alpha = np.sqrt(
    grad_alpha_r**2 * data["g^rr"]
    + grad_alpha_t**2 * data["g^tt"]
    + grad_alpha_z**2 * data["g^zz"]
    + 2 * grad_alpha_r * grad_alpha_t * data["g^rt"]
    + 2 * grad_alpha_r * grad_alpha_z * data["g^rz"]
    + 2 * grad_alpha_t * grad_alpha_z * data["g^tz"]
)

grad_psi_dot_grad_alpha = (
    grad_psi * grad_alpha_r * data["g^rr"]
    + grad_psi * grad_alpha_t * data["g^rt"]
    + grad_psi * grad_alpha_z * data["g^rz"]
)

# calculate gds*
shat = -rho / iotas * shear[0]
gds2 = grad_alpha**2 * Lref**2 * psi
gds21 = shat / Bref * grad_psi_dot_grad_alpha
gds22 = (shat / (Lref * Bref)) ** 2 / psi * grad_psi**2 * data["g^rr"]

# calculate gbdrift0 and cvdrift0
B_t = data["B_theta"]
B_z = data["B_zeta"]
dB_t = data["|B|_t"]
dB_z = data["|B|_z"]
jac = data["sqrt(g)"]
gbdrift0 = (
    psib
    / np.abs(psib)
    * shat
    * 2
    / modB**3
    / rho
    * (B_t * dB_z - B_z * dB_t)
    * psib
    / jac
    * 2
    * rho
)
cvdrift0 = gbdrift0

# calculate gbdrift and cvdrift
B_r = data["B_rho"]
dB_r = data["|B|_r"]

# iota = iota_data['iota'][0]
gbdrift_norm = 2 * Bref * Lref**2 / modB**3 * rho
gbdrift = (
    psib
    / np.abs(psib)
    * gbdrift_norm
    / jac
    * (
        B_r * dB_t * (lmbda_z - iota)
        + B_t * dB_z * (lmbda_r - (zeta - zeta_center) * shear[0])
        + B_z * dB_r * (1 + lmbda_t)
        - B_z * dB_t * (lmbda_r - (zeta - zeta_center) * shear[0])
        - B_t * dB_r * (lmbda_z - iota)
        - B_r * dB_z * (1 + lmbda_t)
    )
)
Bsa = 1 / jac * (B_z * (1 + lmbda_t) - B_t * (lmbda_z - iota))
p_r = data["p_r"]
cvdrift = gbdrift + 2 * Bref * Lref**2 / modB**2 * rho * mu_0 / modB**2 * p_r * Bsa

Lref = Lref
shat = shat
iota = iota


(
    uniform_zgrid,
    bmag_gx,
    grho_gx,
    gradpar_gx,
    gds2_gx,
    gds21_gx,
    gds22_gx,
    gbdrift_gx,
    gbdrift0_gx,
    cvdrift_gx,
    cvdrift0_gx,
) = get_gx_arrays(
    zeta, bmag, grho, gradpar, gds2, gds21, gds22, gbdrift, gbdrift0, cvdrift, cvdrift0
)

path_geo = eikfile

nperiod = 1
kxfac = 1.0
f = open(path_geo, "w")
f.write("ntgrid nperiod ntheta drhodpsi rmaj shat kxfac q scale")
f.write(
    "\n"
    + str(nzgrid)
    + " "
    + str(nperiod)
    + " "
    + str(2 * nzgrid)
    + " "
    + str(1.0)
    + " "
    + str(1 / Lref)
    + " "
    + str(shat)
    + " "
    + str(kxfac)
    + " "
    + str(1 / iota[0])
    + " "
    + str(2 * npol - 1)
)

f.write("\ngbdrift gradpar grho tgrid")
for i in range(len(uniform_zgrid)):
    f.write(
        "\n"
        + str(gbdrift_gx[i])
        + " "
        + str(gradpar_gx[i])
        + " "
        + str(grho_gx[i])
        + " "
        + str(uniform_zgrid[i])
    )

f.write("\ncvdrift gds2 bmag tgrid")
for i in range(len(uniform_zgrid)):
    f.write(
        "\n"
        + str(cvdrift_gx[i])
        + " "
        + str(gds2_gx[i])
        + " "
        + str(bmag_gx[i])
        + " "
        + str(uniform_zgrid[i])
    )

f.write("\ngds21 gds22 tgrid")
for i in range(len(uniform_zgrid)):
    f.write(
        "\n" + str(gds21_gx[i]) + " " + str(gds22_gx[i]) + " " + str(uniform_zgrid[i])
    )

f.write("\ncvdrift0 gbdrift0 tgrid")
for i in range(len(uniform_zgrid)):
    f.write(
        "\n"
        + str(cvdrift0_gx[i])
        + " "
        + str(gbdrift0_gx[i])
        + " "
        + str(uniform_zgrid[i])
    )

f.close()
