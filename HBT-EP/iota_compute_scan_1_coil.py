import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../"))

from desc import set_device

set_device("gpu")

import numpy as np
from desc.grid import LinearGrid
from desc.coils import FourierRZCoil
from desc.magnetic_fields import (
    field_line_integrate,
    ToroidalMagneticField,
    SumMagneticField,
)
from desc.backend import jnp
import matplotlib.pyplot as plt
from desc.plotting import poincare_plot

plot_grid = LinearGrid(rho=1, M=20, N=60, NFP=1, endpoint=True)
coil_grid = LinearGrid(N=30)


def add_coils_to_plot(ax, fields, nplanes=6):
    fields = fields if isinstance(fields, list) else [fields]
    for field in fields:
        data = field.compute(["R", "Z", "phi"], grid=LinearGrid(zeta=nplanes))
        for i in range(nplanes):
            ax.flat[i].scatter(data["R"][i], data["Z"][i], marker="*")
    return ax


def get_umbilic_coil(I, rmaj=0.99, rmin=0.2, helical_offset=0):
    zeta = jnp.linspace(0, 2 * jnp.pi, 41)
    R = rmaj + rmin * jnp.cos(zeta - helical_offset)
    Z = rmin * jnp.sin(zeta - helical_offset)

    data = jnp.vstack([R, zeta, Z]).T
    umbilic_coil = FourierRZCoil.from_values(
        current=I,
        coords=data,
        N=1,
        basis="rpz",
    )
    return umbilic_coil


def get_poincare_plot(
    field,
    r0,
    z0,
    ntransit=200,
    phi=6,
    bounds_R=(0, jnp.inf),
    bounds_Z=(-jnp.inf, jnp.inf),
    atol=1e-6,
    rtol=1e-6,
    min_step_size=1e-8,
    max_steps=10000,
    return_data=False,
):
    return poincare_plot(
        field,
        R0=r0,
        Z0=z0,
        ntransit=ntransit,
        phi=phi,
        size=0.5,
        grid=coil_grid,
        bounds_R=bounds_R,
        bounds_Z=bounds_Z,
        atol=atol,
        rtol=rtol,
        min_step_size=min_step_size,
        max_steps=max_steps,
        return_data=return_data,
    )


R0 = 0.92
B0 = 0.35
ntransit = 200
Nphi = 50
xd = 0.02
N = 10
rmaj = 0.92
rmin = 0.2
axis_R = R0
axis_Z = 0
negative = False

plasma_coil = FourierRZCoil(current=14000, R_n=R0, Z_n=0)
tf = ToroidalMagneticField(B0=B0, R0=R0)
Z0is = jnp.zeros(N)
phi = jnp.linspace(0, 2 * jnp.pi, Nphi, endpoint=False)
phis = (phi + jnp.arange(0, ntransit)[:, None] * 2 * jnp.pi).flatten()
R0is = jnp.linspace(R0 + 0.03, rmaj + rmin - xd, N)

field_just_plasma = SumMagneticField([plasma_coil, tf])
fig, ax, data_plasma = get_poincare_plot(
    field_just_plasma,
    R0is,
    Z0is,
    ntransit,
    phi=Nphi,
    bounds_R=(rmaj - rmin, rmaj + rmin),
    bounds_Z=(-rmin, rmin),
    return_data=True,
    max_steps=20000,
    atol=1e-8,
    rtol=1e-8,
)
add_coils_to_plot(ax, [plasma_coil], nplanes=Nphi)
ax.flat[0].scatter(R0is, Z0is, color="black", label="initial conditions", s=2)
keep_axes = [ax.flat[0]]

for a in ax.flat:
    if a not in keep_axes:
        a.remove()

new_gs = plt.GridSpec(2, 3, figure=fig)
for i, a in enumerate(keep_axes):
    a.set_subplotspec(new_gs[i])

fig.set_size_inches(12, 8)
fig.tight_layout()

zs = data_plasma["Z"].reshape((ntransit * Nphi, -1))
rs = data_plasma["R"].reshape((ntransit * Nphi, -1))
iotas_just_plasma = []
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
    iota = total_poloidal / total_toroidal
    if total_toroidal < ntransit / 10 * 2 * jnp.pi:
        iota = jnp.nan
    iotas_just_plasma.append(iota)
iotas_just_plasma = jnp.array(iotas_just_plasma)

name = f"1coil-R0-{R0}-B0-{B0}-rmaj-{rmaj}-rmin-{rmin}-Ip-{plasma_coil.current}"
all_data = np.vstack([R0is, iotas_just_plasma])
for Ic in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000]:
    if negative:
        Ic *= -1
    print(f"Plotting poincare plot of Ic={Ic}")
    umbilic_coil = get_umbilic_coil(I=Ic, rmaj=rmaj, rmin=rmin)
    field = SumMagneticField([plasma_coil, umbilic_coil, tf])
    fig, ax, data = get_poincare_plot(
        field,
        R0is,
        Z0is,
        ntransit,
        phi=Nphi,
        bounds_R=(rmaj - rmin, rmaj + rmin),
        bounds_Z=(-rmin, rmin),
        return_data=True,
        max_steps=20000,
        atol=1e-8,
        rtol=1e-8,
    )
    add_coils_to_plot(ax, [plasma_coil, umbilic_coil], nplanes=Nphi)
    ax.flat[0].scatter(R0is, Z0is, color="black", label="initial conditions", s=2)
    ax.flat[0].legend()

    # above plot has many phi cuts to compute iota accurately, delete some of
    # them for easy read
    indices = np.linspace(0, Nphi - 1, 6, dtype=int)
    keep_axes = [ax.flat[i] for i in indices]

    for a in ax.flat:
        if a not in keep_axes:
            a.remove()

    new_gs = plt.GridSpec(2, 3, figure=fig)
    for i, a in enumerate(keep_axes):
        a.set_subplotspec(new_gs[i])

    fig.set_size_inches(12, 8)
    fig.tight_layout()
    fig.savefig(f"./Scans/ft-{name}-Ic-{Ic}.png", dpi=500)
    zs = data["Z"].reshape((ntransit * Nphi, -1))
    rs = data["R"].reshape((ntransit * Nphi, -1))
    iotas = []
    print(f"Computing iota of Ic={Ic}")
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
        iota = total_poloidal / total_toroidal
        # if integration terminated very early, do not consider that result
        if total_toroidal < ntransit / 10 * 2 * jnp.pi:
            iota = jnp.nan
        iotas.append(iota)
    iotas = jnp.array(iotas)
    all_data = np.vstack([all_data, iotas])


def plot_iotas(data):
    r = data[0, :]
    base = data[1, :]
    indices = np.arange(len(base))[3:-1]
    currents = np.arange(1, 10)
    plt.plot(
        r[indices], np.abs(base)[indices], label="No Current", color="navy", linewidth=2
    )
    for i, di in enumerate(data[2:, :]):
        last_not_nan_idx = np.where(
            ~np.isnan(di[indices]), np.arange(len(indices)), 0
        ).max()
        percent = (
            100
            * (
                np.abs(di)[indices][last_not_nan_idx]
                - np.abs(base)[indices][last_not_nan_idx]
            )
            / np.abs(base)[indices][last_not_nan_idx]
        )
        plt.plot(
            r[indices],
            np.abs(di)[indices],
            label=f"{currents[i]} kA, ({percent:.2f}%)",
            color=plt.cm.Blues(0.8 - 0.6 * i / (len(data) - 2)),
        )

    plt.xlabel("R (m)")
    plt.ylabel("|Iota|")
    plt.title("Ip=14kA, B0=0.35T, 1 Umbilic Coil")
    plt.legend()
    plt.grid(True, alpha=0.3)


np.savetxt(
    (
        f"./Scans/data-{name}-Ic-1-9kA.txt"
        if not negative
        else f"./Scans/data-{name}-Ic-n1-9kA.txt"
    ),
    all_data,
)

plt.figure()
plot_iotas(all_data)
plt.savefig(
    (
        f"./Scans/iotas-{name}-Ic-1-9kA.png"
        if not negative
        else f"./Scans/iotas-{name}-Ic-n1-9kA.png"
    ),
    dpi=500,
)
