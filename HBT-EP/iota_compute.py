import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../"))

from desc import set_device

set_device("gpu")

from desc.grid import LinearGrid
from desc.coils import FourierRZCoil
from desc.magnetic_fields import (
    field_line_integrate,
    ToroidalMagneticField,
    SumMagneticField,
)
from desc.backend import jnp
import matplotlib.pyplot as plt

plot_grid = LinearGrid(rho=1, M=20, N=60, NFP=1, endpoint=True)
coil_grid = LinearGrid(N=30)

N = 10
R0 = 0.92
R0_shift = R0 + 0.05
B0 = 0.35
offset = 0.2
zeta = jnp.linspace(0, 2 * jnp.pi, 41)
helical_offset = 0
R = R0_shift + offset * jnp.cos(zeta - helical_offset)
Z = offset * jnp.sin(zeta - helical_offset)
helical_offset2 = jnp.pi
R2 = R0_shift + offset * jnp.cos(zeta - helical_offset2)
Z2 = offset * jnp.sin(zeta - helical_offset2)

data = jnp.vstack([R, zeta, Z]).T
umbilic_coil = FourierRZCoil.from_values(
    current=3000,
    coords=data,
    N=1,
    basis="rpz",
)
plasma_coil = FourierRZCoil(current=14000, R_n=R0, Z_n=0)
tf = ToroidalMagneticField(B0=B0, R0=R0)
field = SumMagneticField([plasma_coil, tf])
xd = 0.02

R0is = jnp.linspace(R0_shift - 0.03, R0_shift + offset - xd, N)
Z0is = jnp.zeros(N)
ntransit = 100
NFP = 1
phi = 100
nplanes = phi
phi = jnp.linspace(0, 2 * jnp.pi / NFP, phi, endpoint=False)

phis = (phi + jnp.arange(0, ntransit)[:, None] * 2 * jnp.pi / NFP).flatten()

R0s, Z0s = jnp.atleast_1d(R0is, Z0is)
print(f"Initial position R: {R0s}")
print(f"Initial position Z: {Z0s}")

fieldR, fieldZ, (_, result) = field_line_integrate(
    r0=R0s,
    z0=Z0s,
    phis=phis,
    field=field,
    source_grid=coil_grid,
    return_aux=True,
    bounds_R=(R0_shift - offset, R0_shift + offset),
    bounds_Z=(-offset, offset),
    atol=1e-6,
    rtol=1e-6,
    min_step_size=1e-8,
    max_steps=10000,
    options={"throw": False},
)
zs = fieldZ.reshape((ntransit * nplanes, -1))
rs = fieldR.reshape((ntransit * nplanes, -1))
axis_R = R0
axis_Z = 0
iotas = []
for i in range(N):
    r_traj_1 = rs[:, i]
    z_traj_1 = zs[:, i]
    R = jnp.sqrt((r_traj_1 - axis_R) ** 2 + (z_traj_1 - axis_Z) ** 2)
    # unwrap arctan2 so theta accumulates beyond +/-pi
    theta_raw = jnp.arctan2(z_traj_1 - axis_Z, r_traj_1 - axis_R)
    theta_unwrapped = jnp.unwrap(theta_raw)
    valid = ~jnp.isnan(theta_unwrapped)
    last_valid_idx = jnp.where(valid, jnp.arange(len(theta_unwrapped)), 0).max()
    total_poloidal = theta_unwrapped[last_valid_idx] - theta_unwrapped[0]
    total_toroidal = phis[last_valid_idx] - phis[0]
    iota = abs(total_poloidal / total_toroidal)
    iotas.append(iota)
    print(f"Initial position R, Z: {R0s[i]:.2f} {Z0s[i]:.2f}")
    print(f"Computed iota = {iota:.3f}")
iotas = jnp.array(iotas)
plt.plot(iotas)
plt.show()
