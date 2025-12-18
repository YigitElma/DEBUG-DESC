import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../../"))

from desc import set_device

set_device("gpu")

import nvtx

from desc.basis import *
from desc.backend import *
from desc.compute import *
from desc.coils import *
from desc.equilibrium import *
from desc.examples import *
from desc.grid import *
from desc.geometry import *
from desc.io import *

from desc.objectives import *
from desc.objectives.objective_funs import *
from desc.objectives.getters import *
from desc.objectives.normalization import compute_scaling_factors
from desc.objectives.utils import *
from desc.optimize._constraint_wrappers import *

from desc.transform import Transform
from desc.plotting import *
from desc.optimize import *
from desc.perturbations import *
from desc.profiles import *
from desc.compat import *
from desc.utils import *
from desc.magnetic_fields import *
from desc.particles import *
from diffrax import *

from desc.__main__ import main
from desc.vmec_utils import vmec_boundary_subspace
from desc.input_reader import InputReader
from desc.continuation import solve_continuation_automatic
from desc.compute.data_index import register_compute_fun
from desc.optimize.utils import solve_triangular_regularized

print_backend_info()


class FourierChebyshevFieldTest(IOAble):
    """Diffrax-compatible field class using strictly real arithmetic.

    Optimized to reduce memory allocation and kernel launches inside the
    Diffrax stepping loop.
    """

    _static_attrs = ["L", "M", "N", "M_fft", "N_fft", "data_keys"]

    def __init__(self, L, M, N):
        self.L = L
        self.M = M
        self.N = N

    def build(self, eq):
        """Build the constants for fit."""
        self.data_keys = ["B", "grad(|B|)", "e^rho", "e^theta*rho", "e^zeta"]
        self.l = jnp.arange(self.L)
        self.M_fft = 2 * self.M + 1
        self.N_fft = 2 * self.N + 1

        self.m = jnp.fft.fftfreq(self.M_fft) * self.M_fft
        self.n = jnp.fft.fftfreq(self.N_fft) * self.N_fft

        # Chebyshev nodes
        x = jnp.cos(jnp.pi * (2 * self.l + 1) / (2 * self.L))
        rho = (x + 1) / 2

        self.grid = LinearGrid(rho=rho, M=self.M, N=self.N, sym=False, NFP=eq.NFP)
        self.transforms = get_transforms(self.data_keys, eq, self.grid)

    def fit(self, params, profiles):
        """Fit series and prepare optimized real-valued coefficients."""
        # 1. Compute raw data
        data_raw = compute_fun(
            "desc.equilibrium.equilibrium.Equilibrium",
            self.data_keys,
            params,
            self.transforms,
            profiles,
        )
        L, M, N = self.L, self.M_fft, self.N_fft

        # 2. Stack data for batch processing
        # Order: B(3), grad|B|(3), e^rho(3), e^theta*rho(3), e^zeta_p(1) -> Total 13
        keys3d = [key for key in self.data_keys if key != "e^zeta"]
        arrays = [
            data_raw[key][:, i].reshape(N, L, M) for key in keys3d for i in [0, 1, 2]
        ]
        arrays.append(data_raw["e^zeta"][:, 1].reshape(N, L, M))

        stacked_data = jnp.stack(arrays)  # Shape (13, N, L, M)

        # 3. Perform Transforms
        # Chebyshev Transform (DCT)
        coefs = jax.scipy.fft.dct(stacked_data, axis=2, norm=None)
        coefs = coefs.at[:, :, 0, :].multiply(0.5)  # Fix 0-th mode
        coefs = coefs * (1.0 / self.L)

        # Fourier Transforms (FFT)
        coefs = jnp.fft.fft(coefs, axis=3, norm=None)  # M axis
        coefs = jnp.fft.fft(coefs, axis=1, norm=None)  # N axis

        # 4. Optimization: Pre-normalize
        # Move the division by (M*N) from evaluate() to here
        norm_factor = 1.0 / (self.M_fft * self.N_fft)
        coefs = coefs * norm_factor

        # 5. Optimization: Stack for Dot Product
        # We need Re(Field) = C_real * Basis_real - C_imag * Basis_imag
        # We store this as: Dot([C_real, -C_imag], [Basis_real, Basis_imag])
        # Result shape: (2, 13, N, L, M)
        coefs_optimized = jnp.stack([coefs.real, -coefs.imag], axis=0)

        self.params_dict = {
            "coefs_opt": coefs_optimized,
            "l": self.l,
            "m": self.m,
            "n": self.n,
        }

    # JIT this function or the function calling it!
    def evaluate(self, rho, theta, zeta, params=None):
        if params is None:
            params = self.params_dict

        # --- 1. Chebyshev Basis (L) ---
        r0p = 1 - 2 * rho
        # Shape: (L,)
        Tl = jnp.cos(params["l"] * jnp.arccos(r0p))

        # --- 2. Fourier Basis (M, N) ---
        # Map zeta to [0, 2pi/NFP]
        zeta = (zeta * self.grid.NFP) % (2 * jnp.pi)

        # Calculate trig components once
        m_theta = jnp.outer(
            theta, params["m"]
        ).flatten()  # Handle batch if needed, here assuming scalar
        n_zeta = jnp.outer(zeta, params["n"]).flatten()

        cm, sm = jnp.cos(m_theta), jnp.sin(m_theta)
        cn, sn = jnp.cos(n_zeta), jnp.sin(n_zeta)

        # Compute Basis_Real and Basis_Imag using outer products
        # Real(e^i(m+n)) = cm*cn - sm*sn
        # Imag(e^i(m+n)) = sm*cn + cm*sn
        # We use broadcasting to get shape (M, N)
        basis_real = cm[None, :] * cn[:, None] - sm[None, :] * sn[:, None]
        basis_imag = sm[None, :] * cn[:, None] + cm[None, :] * sn[:, None]

        # Stack shape: (2, N, M) - Note: transposing to match coefs layout (N, M) if needed
        # Coefs are (..., N, L, M). Let's align basis to (2, N, M)
        # basis_real is currently (M, N) -> Transpose to (N, M)
        basis_stack = jnp.stack([basis_real.T, basis_imag.T], axis=0)

        # --- 3. Fused Contraction ---
        # p: Real/Imag stack (2)
        # k: Field components (13)
        # n: Toroidal modes
        # l: Radial modes
        # m: Poloidal modes
        # We sum over p, n, l, m.
        # Result shape: (k,)

        results = jnp.einsum(
            "pknlm,l,pnm->k", params["coefs_opt"], Tl, basis_stack, optimize="optimal"
        )

        # --- 4. Pack Output ---
        # This part is cheap.
        B = results[0:3]
        B_norm = jnp.linalg.norm(B)

        return {
            "|B|": B_norm,
            "b": B / B_norm,
            "grad(|B|)": results[3:6],
            "e^rho": results[6:9],
            "e^theta*rho": results[9:12],
            "e^zeta": jnp.array([0.0, results[12], 0.0]),
        }


eq = get("precise_QA")
iota = eq.get_profile("iota")
params = eq.params_dict
params["i_l"] = iota.params

model = VacuumGuidingCenterTrajectory(frame="flux")
rhos = np.linspace(0.05, 1.0, 3)
grid = LinearGrid(rho=rhos, M=2, N=2, NFP=eq.NFP, sym=eq.sym)
particles = ManualParticleInitializerFlux(
    rho0=grid.nodes[:, 0],
    theta0=grid.nodes[:, 1],
    zeta0=grid.nodes[:, 2],
    xi0=2 * np.random.rand(grid.num_nodes) - 1,
    E=3.5e6,
)
x0, args = particles.init_particles(model=model, field=eq)

xi = x0[0]
argsi = args[0]
rho, theta, zeta, vpar = xi

xp = rho * np.cos(theta)
yp = rho * np.sin(theta)

x = jnp.array([xp, yp, zeta, vpar])
spliner = SplineFieldFlux(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid)
spliner.build(eq=eq)
spliner.fit(params=params, profiles={"current": eq.current, "iota": eq.iota})
interpolator = FourierChebyshevField(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid)
interpolator.build(eq)
interpolator.fit(params, {"iota": eq.iota, "current": eq.current})


interpolator2 = FourierChebyshevFieldTest(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid)
interpolator2.build(eq)
interpolator2.fit(params, {"iota": eq.iota, "current": eq.current})

with nvtx.annotate("eq", color="green"):
    _ = model._compute_flux_coordinates(
        x=x.squeeze(),
        eq=eq,
        params=params,
        m=argsi[0],
        q=argsi[1],
        mu=argsi[2],
        iota=iota,
    ).block_until_ready()
    for _ in range(100):
        with nvtx.annotate("eq", color="red"):
            _ = model._compute_flux_coordinates(
                x=x.squeeze(),
                eq=eq,
                params=params,
                m=argsi[0],
                q=argsi[1],
                mu=argsi[2],
                iota=iota,
            ).block_until_ready()

with nvtx.annotate("fc-org", color="red"):
    _ = model._compute_flux_coordinates_with_fit(
        x=x.squeeze(), field=interpolator, m=argsi[0], q=argsi[1], mu=argsi[2]
    ).block_until_ready()
    for _ in range(100):
        with nvtx.annotate("fc-org", color="blue"):
            _ = model._compute_flux_coordinates_with_fit(
                x=x.squeeze(), field=interpolator, m=argsi[0], q=argsi[1], mu=argsi[2]
            ).block_until_ready()

with nvtx.annotate("fc-new", color="green"):
    _ = model._compute_flux_coordinates_with_fit(
        x=x.squeeze(), field=interpolator2, m=argsi[0], q=argsi[1], mu=argsi[2]
    ).block_until_ready()
    for _ in range(100):
        with nvtx.annotate("fc-new", color="red"):
            _ = model._compute_flux_coordinates_with_fit(
                x=x.squeeze(), field=interpolator2, m=argsi[0], q=argsi[1], mu=argsi[2]
            ).block_until_ready()

with nvtx.annotate("spline", color="red"):
    _ = model._compute_flux_coordinates_with_fit(
        x=x.squeeze(), field=spliner, m=argsi[0], q=argsi[1], mu=argsi[2]
    ).block_until_ready()
    for _ in range(100):
        with nvtx.annotate("spline", color="blue"):
            _ = model._compute_flux_coordinates_with_fit(
                x=x.squeeze(), field=spliner, m=argsi[0], q=argsi[1], mu=argsi[2]
            ).block_until_ready()
