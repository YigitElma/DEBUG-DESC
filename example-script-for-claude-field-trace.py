import sys
import os
import time
import warnings

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../"))

from desc import set_device

set_device("gpu")

import jax
import numpy as np
from functools import partial
from scipy.constants import mu_0
from diffrax import (
    RecursiveCheckpointAdjoint,
    SaveAt,
    Event,
    PIDController,
    Tsit5,
)

from desc.grid import LinearGrid
from desc.optimizable import Optimizable
from desc.magnetic_fields import ToroidalMagneticField, SumMagneticField, _MagneticField
from desc.magnetic_fields._core import _field_line_integrate
from desc.backend import jnp, print_backend_info
from desc.coils import FourierRZCoil
from desc.utils import rpz2xyz, xyz2rpz_vec

print_backend_info()


# ============================================================
# Setup (shared by all versions)
# ============================================================


def get_umbilic_coil(I, rmaj=0.99, rmin=0.2, helical_offset=0):
    zeta = np.linspace(0, 2 * np.pi, 41)
    R = rmaj + rmin * np.cos(zeta - helical_offset)
    Z = rmin * np.sin(zeta - helical_offset)

    data = jnp.vstack([R, zeta, Z]).T
    umbilic_coil = FourierRZCoil.from_values(
        current=I,
        coords=data,
        N=1,
        basis="rpz",
    )
    return umbilic_coil


B0 = 0.35
R0 = 0.92

coil_grid = LinearGrid(N=30)

rmaj = 0.97
rmin = 0.2
umbilic_coil = get_umbilic_coil(I=3000, rmaj=rmaj, rmin=rmin)

plasma_coil = FourierRZCoil(current=14000, R_n=R0, Z_n=0)
tf = ToroidalMagneticField(B0=B0, R0=R0)
field = SumMagneticField([plasma_coil, umbilic_coil, tf])
xd = 0.02

N = 10
ntransit = 200
r0 = jnp.linspace(rmaj - 0.03, rmaj + rmin - xd, N)
z0 = jnp.zeros(N)
Nphi = 50
phi = np.linspace(0, 2 * np.pi, Nphi, endpoint=False)
phis = (phi + np.arange(0, ntransit)[:, None] * 2 * np.pi).flatten()

min_step_size = 1e-8
max_steps = 10000
bounds_R = (rmaj - rmin, rmaj + rmin)
bounds_Z = (-rmin, rmin)


# diffrax parameters
def default_terminating_event(t, y, args, **kwargs):
    R_out = jnp.logical_or(y[0] < bounds_R[0], y[0] > bounds_R[1])
    Z_out = jnp.logical_or(y[2] < bounds_Z[0], y[2] > bounds_Z[1])
    return jnp.logical_or(R_out, Z_out)


saveat = SaveAt(ts=phis)
event = Event(default_terminating_event)
adjoint = RecursiveCheckpointAdjoint()
stepsize_controller = PIDController(rtol=1e-6, atol=1e-6, dtmin=min_step_size)


# ============================================================
# PrecomputedField: general, no isinstance checks
# ============================================================


def _flatten_field_tree(field):
    """Recursively collect leaf fields from any nesting.

    Handles SumMagneticField, CoilSet, MixedCoilSet (have _fields),
    ScaledMagneticField (has _field), or any future wrapper.
    No isinstance checks - uses getattr duck typing.
    """
    # collection-like: has _fields (SumMagneticField, CoilSet, MixedCoilSet, etc.)
    children = getattr(field, "_fields", None)
    if children is not None:
        result = []
        for child in children:
            result.extend(_flatten_field_tree(child))
        return result
    # wrapper-like: has _field (ScaledMagneticField, etc.)
    child = getattr(field, "_field", None)
    if child is not None:
        return _flatten_field_tree(child)
    # leaf field
    return [field]


def _precompute_coil_data(field, source_grid):
    """Extract Biot-Savart source data from any field tree.

    Uses duck typing: if a leaf field has .current and can compute
    ["x", "x_s", "ds"], it's a coil. Everything else is analytical.
    """
    leaves = _flatten_field_tree(field)

    coil_pts_list = []
    coil_JdV_list = []
    analytical_fields = []

    for leaf in leaves:
        try:
            current = leaf.current
            data = leaf.compute(["x", "x_s", "ds"], grid=source_grid, basis="xyz")
            coil_pts_list.append(data["x"])
            coil_JdV_list.append(current * data["x_s"] * data["ds"][:, None])
        except (AttributeError, TypeError, NotImplementedError):
            analytical_fields.append(leaf)

    src_pts = jnp.concatenate(coil_pts_list, axis=0) if coil_pts_list else None
    JdV = jnp.concatenate(coil_JdV_list, axis=0) if coil_pts_list else None
    return src_pts, JdV, analytical_fields


class PrecomputedField(_MagneticField, Optimizable):
    """Lightweight field with precomputed Biot-Savart source data.

    Works with any field configuration (CoilSets, MixedCoilSets,
    SumMagneticField with arbitrary nesting, ScaledMagneticField, etc.).
    Coils are identified via duck typing (has .current + can compute geometry).
    Analytical fields are kept as-is.
    """

    _io_attrs_ = _MagneticField._io_attrs_ + [
        "_src_pts",
        "_JdV",
        "_analytical_fields",
    ]
    _static_attrs = _MagneticField._static_attrs + Optimizable._static_attrs

    def __init__(self, field, source_grid=None):
        src_pts, JdV, analytical_fields = _precompute_coil_data(field, source_grid)
        self._src_pts = src_pts
        self._JdV = JdV
        self._analytical_fields = analytical_fields

    def compute_magnetic_field(
        self,
        coords,
        params=None,
        basis="rpz",
        source_grid=None,
        transforms=None,
        chunk_size=None,
    ):
        """Compute magnetic field at a set of points."""
        coords = jnp.atleast_2d(jnp.asarray(coords))

        if basis == "rpz":
            phi = coords[:, 1]
            coords_xyz = rpz2xyz(coords)
        else:
            coords_xyz = coords

        # Biot-Savart from precomputed coil data
        B_rpz = jnp.zeros((coords.shape[0], 3))
        if self._src_pts is not None:
            dr = self._src_pts[None, :, :] - coords_xyz[:, None, :]
            num = jnp.cross(dr, self._JdV[None, :, :], axis=-1)
            den = jnp.linalg.norm(dr, axis=-1, keepdims=True) ** 3
            mask = den <= 0
            num = jnp.where(mask, 0.0, num)
            den = jnp.where(mask, 1.0, den)
            B_xyz = (num / den).sum(axis=1) * mu_0 / (4 * jnp.pi)

            if basis == "rpz":
                B_rpz = xyz2rpz_vec(B_xyz, phi=phi)
            else:
                B_rpz = B_xyz

        # Add analytical fields
        for af in self._analytical_fields:
            B_rpz = B_rpz + af.compute_magnetic_field(coords, params=None, basis=basis)

        return B_rpz

    def compute_magnetic_vector_potential(
        self,
        coords,
        params=None,
        basis="rpz",
        source_grid=None,
        transforms=None,
        chunk_size=None,
    ):
        """Not implemented."""
        raise NotImplementedError("PrecomputedField does not support vector potential.")


# ============================================================
# Build precomputed field
# ============================================================

print("Precomputing coil geometry...")
precomp_field = PrecomputedField(field, coil_grid)
print(f"  Coil source pts: {precomp_field._src_pts.shape}")
print(
    f"  Analytical subfields: "
    f"{[type(f).__name__ for f in precomp_field._analytical_fields]}"
)


# ============================================================
# Verify precomputed field matches original at random points
# ============================================================

print("\nVerifying PrecomputedField vs original at 100 random points...")
test_pts = jnp.column_stack(
    [
        jnp.linspace(bounds_R[0] + 0.01, bounds_R[1] - 0.01, 100),
        jnp.linspace(0, 2 * jnp.pi, 100),
        jnp.linspace(bounds_Z[0] + 0.01, bounds_Z[1] - 0.01, 100),
    ]
)
B_orig = field.compute_magnetic_field(test_pts, basis="rpz", source_grid=coil_grid)
B_precomp = precomp_field.compute_magnetic_field(test_pts, basis="rpz")
err = float(jnp.max(jnp.abs(B_orig - B_precomp)))
print(f"  Max |B_orig - B_precomp|: {err:.2e}")


# ============================================================
# General scan-based RK4 (works with ANY _MagneticField)
# ============================================================


@partial(jax.jit, static_argnums=(4,))
def _scan_integrate(rpz0, phis, field_args, bounds, n_substeps):
    """Integrate one field line using scan + RK4 with checkpointing.

    field_args: (field, source_grid, scale) - any _MagneticField works.
    """
    field, source_grid, scale = field_args
    R_min, R_max, Z_min, Z_max = bounds

    def _odefun(phi, rpz):
        r = rpz[0]
        br, bp, bz = (
            scale
            * field.compute_magnetic_field(
                rpz[None, :], basis="rpz", source_grid=source_grid
            ).squeeze()
        )
        return jnp.array(
            [
                r * br / bp * jnp.sign(bp),
                jnp.sign(bp),
                r * bz / bp * jnp.sign(bp),
            ]
        )

    def _rk4_substep(rpz, phi_and_dphi):
        phi, dphi = phi_and_dphi
        k1 = _odefun(phi, rpz)
        k2 = _odefun(phi + dphi / 2, rpz + dphi / 2 * k1)
        k3 = _odefun(phi + dphi / 2, rpz + dphi / 2 * k2)
        k4 = _odefun(phi + dphi, rpz + dphi * k3)
        rpz_new = rpz + (dphi / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        out = jnp.logical_or(
            jnp.logical_or(rpz_new[0] < R_min, rpz_new[0] > R_max),
            jnp.logical_or(rpz_new[2] < Z_min, rpz_new[2] > Z_max),
        )
        rpz_new = jnp.where(out, rpz, rpz_new)
        return rpz_new, None

    def _advance_one_interval(rpz, phi_pair):
        phi_start, phi_end = phi_pair
        dphi = (phi_end - phi_start) / n_substeps
        sub_phis = phi_start + jnp.arange(n_substeps) * dphi
        sub_inputs = (sub_phis, jnp.full(n_substeps, dphi))
        rpz, _ = jax.lax.scan(_rk4_substep, rpz, sub_inputs)
        return rpz, rpz

    _, trajectory = jax.lax.scan(
        jax.checkpoint(_advance_one_interval),
        rpz0,
        (phis[:-1], phis[1:]),
    )
    trajectory = jnp.concatenate([rpz0[None, :], trajectory], axis=0)
    return trajectory


def run_scan(any_field, source_grid, n_substeps=4):
    """General scan-based RK4 for any _MagneticField."""
    rshape = r0.shape
    r0_flat = r0.flatten()
    z0_flat = z0.flatten()
    x0 = jnp.array([r0_flat, phis[0] * jnp.ones_like(r0_flat), z0_flat]).T

    scale = jnp.sign(
        any_field.compute_magnetic_field(x0, source_grid=source_grid)[0, 1]
    )
    field_args = (any_field, source_grid, scale)
    bounds = (bounds_R[0], bounds_R[1], bounds_Z[0], bounds_Z[1])

    def _single_line(rpz0):
        return _scan_integrate(rpz0, phis, field_args, bounds, n_substeps)

    trajectories = jax.vmap(_single_line)(x0)
    r_out = trajectories[:, :, 0].T.reshape((phis.size, *rshape))
    z_out = trajectories[:, :, 2].T.reshape((phis.size, *rshape))
    return r_out, z_out


# ============================================================
# Benchmark functions
# ============================================================


def run_original():
    """Original: full field + diffrax."""
    return _field_line_integrate(
        r0=r0,
        z0=z0,
        phis=phis,
        field=field,
        params=None,
        source_grid=coil_grid,
        solver=Tsit5(),
        max_steps=max_steps,
        min_step_size=min_step_size,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        event=event,
        adjoint=adjoint,
        chunk_size=None,
        bs_chunk_size=None,
        options={"throw": False},
        return_aux=False,
    )


def run_precomp_diffrax():
    """PrecomputedField + diffrax."""
    return _field_line_integrate(
        r0=r0,
        z0=z0,
        phis=phis,
        field=precomp_field,
        params=None,
        source_grid=None,
        solver=Tsit5(),
        max_steps=max_steps,
        min_step_size=min_step_size,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        event=event,
        adjoint=adjoint,
        chunk_size=None,
        bs_chunk_size=None,
        options={"throw": False},
        return_aux=False,
    )


# ============================================================
# Benchmark
# ============================================================


def bench(name, fn, n_runs=3):
    """Warmup once, then time n_runs."""
    print(f"\n  [{name}] Warmup (JIT compile)...")
    r, z = fn()
    r.block_until_ready()
    z.block_until_ready()
    print(f"    shape: r={r.shape}, z={z.shape}")

    times = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        r_i, z_i = fn()
        r_i.block_until_ready()
        z_i.block_until_ready()
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"    run {i+1}: {dt:.3f}s")
    avg = np.mean(times)
    print(f"    avg: {avg:.3f}s")
    return avg, r, z


if __name__ == "__main__":
    n_runs = 3

    print("\n" + "=" * 60)
    print(f"Benchmarking field line tracing ({n_runs} timed runs each)")
    print(
        f"  N={N} field lines, ntransit={ntransit}, "
        f"Nphi={Nphi}, max_steps={max_steps}"
    )
    print("=" * 60)

    results = {}

    # 1) Original diffrax (baseline)
    avg, r_orig, z_orig = bench(
        "Original (diffrax + Biot-Savart)", run_original, n_runs
    )
    results["Original (diffrax + Biot-Savart)"] = avg

    # 2) PrecomputedField + diffrax
    avg, r_pd, z_pd = bench("PrecomputedField + diffrax", run_precomp_diffrax, n_runs)
    results["PrecomputedField + diffrax"] = avg

    # 3) scan + original field (general, no precomp)
    avg, r_so, z_so = bench(
        "scan RK4 + original field (sub=4)",
        partial(run_scan, field, coil_grid, n_substeps=4),
        n_runs,
    )
    results["scan RK4 + original field (sub=4)"] = avg

    # 4) scan + PrecomputedField (general, precomp)
    avg, r_sp4, z_sp4 = bench(
        "scan RK4 + PrecomputedField (sub=4)",
        partial(run_scan, precomp_field, None, n_substeps=4),
        n_runs,
    )
    results["scan RK4 + PrecomputedField (sub=4)"] = avg

    # 5) scan + PrecomputedField substeps=8
    avg, r_sp8, z_sp8 = bench(
        "scan RK4 + PrecomputedField (sub=8)",
        partial(run_scan, precomp_field, None, n_substeps=8),
        n_runs,
    )
    results["scan RK4 + PrecomputedField (sub=8)"] = avg

    # 6) scan + PrecomputedField substeps=16
    avg, r_sp16, z_sp16 = bench(
        "scan RK4 + PrecomputedField (sub=16)",
        partial(run_scan, precomp_field, None, n_substeps=16),
        n_runs,
    )
    results["scan RK4 + PrecomputedField (sub=16)"] = avg

    # ---- accuracy analysis ----
    print("\n" + "=" * 60)
    print("Accuracy analysis")
    print("=" * 60)

    n_early = 10 * Nphi
    print("\n  Point-wise error vs original diffrax:")
    for name, (r_t, z_t) in [
        ("PrecomputedField + diffrax", (r_pd, z_pd)),
        ("scan + original field (sub=4)", (r_so, z_so)),
        ("scan + precomp (sub=4)", (r_sp4, z_sp4)),
        ("scan + precomp (sub=8)", (r_sp8, z_sp8)),
        ("scan + precomp (sub=16)", (r_sp16, z_sp16)),
    ]:
        err_r_early = float(jnp.nanmax(jnp.abs(r_orig[:n_early] - r_t[:n_early])))
        err_z_early = float(jnp.nanmax(jnp.abs(z_orig[:n_early] - z_t[:n_early])))
        print(
            f"    {name:40s}  early(10tr): "
            f"dR={err_r_early:.2e} dZ={err_z_early:.2e}"
        )

    # Self-convergence of scan (sub=8 vs sub=16)
    err_r = float(jnp.nanmax(jnp.abs(r_sp8[:n_early] - r_sp16[:n_early])))
    err_z = float(jnp.nanmax(jnp.abs(z_sp8[:n_early] - z_sp16[:n_early])))
    print(
        f"\n  Self-convergence scan sub=8 vs 16 (early 10tr): "
        f"dR={err_r:.2e} dZ={err_z:.2e}"
    )

    # ---- Poincare section plot ----
    print("\n  Saving Poincare section comparison to poincare_comparison.png...")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = [
        ("Original (diffrax)", r_orig, z_orig),
        ("PrecomputedField + diffrax", r_pd, z_pd),
        ("scan + precomp (sub=4)", r_sp4, z_sp4),
        ("scan + precomp (sub=16)", r_sp16, z_sp16),
    ]
    fig, axes = plt.subplots(
        1, len(datasets), figsize=(6 * len(datasets), 5), sharey=True
    )
    for ax, (label, r_data, z_data) in zip(axes, datasets):
        r_poinc = np.asarray(r_data[::Nphi, :])
        z_poinc = np.asarray(z_data[::Nphi, :])
        for j in range(N):
            mask = np.isfinite(r_poinc[:, j]) & np.isfinite(z_poinc[:, j])
            ax.plot(r_poinc[mask, j], z_poinc[mask, j], ".", markersize=0.5)
        ax.set_xlabel("R")
        ax.set_title(label, fontsize=10)
        ax.set_aspect("equal")
    axes[0].set_ylabel("Z")
    plt.tight_layout()
    plt.savefig("poincare_comparison.png", dpi=150)
    plt.close()
    print("  Done.")

    # ---- summary ----
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    ref = results["Original (diffrax + Biot-Savart)"]
    for name, t in results.items():
        speedup = ref / t
        print(f"  {name:45s}  {t:.3f}s  ({speedup:.2f}x)")
