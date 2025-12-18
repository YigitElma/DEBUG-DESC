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
from desc.particles import _trace_particles

print_backend_info()

name = "precise_QA"
try:
    # if the file exists, load it
    eq = desc.io.load(f"Optimization/eqs/{name}_vacuum_scaled_solved.h5")
    eqi_scaled = eq.copy()
except:
    # else, create it from scratch
    eqi = get(name)
    eq = rescale(eq=eqi, L=("a", 1.7044), B=("<B>", 5.86), copy=True)
    eq.pressure = 0
    eq.current = 0
    eq.solve(ftol=1e-4, verbose=1)
    eqi_scaled = eq.copy()
    eq.save(f"Optimization/eqs/{name}_vacuum_scaled_solved.h5")
eq.iota = eq.get_profile("iota")

N = 10000  # number of particles traced
RHO0 = [0.2] * N
xi0 = np.linspace(0.1, 0.9, N, endpoint=True)

model = VacuumGuidingCenterTrajectory(frame="flux")
particles = ManualParticleInitializerFlux(
    rho0=RHO0,
    theta0=np.pi / 2,
    zeta0=0,
    xi0=xi0,  # add negative region too
    E=3.5e6,
)
x0, model_args = particles.init_particles(model, eq)
interpolator = FourierChebyshevField(L=eq.L_grid, M=eq.M_grid, N=eq.N_grid)
interpolator.build(eq)
interpolator.fit(eq.params_dict, {"iota": eq.iota, "current": eq.current})

res = 1
spliner = SplineFieldFlux(
    L=eq.L_grid * res, M=eq.M_grid * res, N=eq.N_grid * res, method="linear"
)
spliner.build(eq)
spliner.fit(eq.params_dict, {"iota": eq.iota, "current": eq.current})

stepsize_controller = ConstantStepSize()
ts = np.linspace(0, 1e-4, 100)
min_step_size = 1e-8
max_steps = int(ts[-1] / min_step_size)

# solver = Tsit5(scan_kind="bounded")
solver = Tsit5()
adjoint = RecursiveCheckpointAdjoint()


def default_event(t, y, args, **kwargs):
    i = jnp.sqrt(y[0] ** 2 + y[1] ** 2)
    return i > 1.0


event = Event(default_event)
particle_chunk_size = None

with nvtx.annotate("eq", color="green"):
    rtz1, _, aux1 = _trace_particles(
        field=eq,
        y0=x0,
        model=model,
        model_args=model_args,
        ts=ts,
        params=eq.params_dict,
        stepsize_controller=PIDController(
            rtol=1e-6,
            atol=1e-6,
            dtmin=min_step_size,
            pcoeff=0.3,
            icoeff=0.3,
            dcoeff=0,
        ),
        saveat=SaveAt(steps=True),
        # saveat=SaveAt(ts=ts),
        max_steps=max_steps,
        min_step_size=min_step_size,
        solver=solver,
        adjoint=adjoint,
        event=event,
        options={},
        chunk_size=particle_chunk_size,
        throw=False,
        return_aux=True,
    )

with nvtx.annotate("eq", color="green"):
    rtz1, _, aux1 = _trace_particles(
        field=eq,
        y0=x0,
        model=model,
        model_args=model_args,
        ts=ts,
        params=eq.params_dict,
        stepsize_controller=PIDController(
            rtol=1e-6,
            atol=1e-6,
            dtmin=min_step_size,
            pcoeff=0.3,
            icoeff=0.3,
            dcoeff=0,
        ),
        saveat=SaveAt(steps=True),
        # saveat=SaveAt(ts=ts),
        max_steps=max_steps,
        min_step_size=min_step_size,
        solver=solver,
        adjoint=adjoint,
        event=event,
        options={},
        chunk_size=particle_chunk_size,
        throw=False,
        return_aux=True,
    )


with nvtx.annotate("eq", color="green"):
    rtz1, _, aux1 = _trace_particles(
        field=eq,
        y0=x0,
        model=model,
        model_args=model_args,
        ts=ts,
        params=eq.params_dict,
        stepsize_controller=PIDController(
            rtol=1e-6,
            atol=1e-6,
            dtmin=min_step_size,
            pcoeff=0.3,
            icoeff=0.3,
            dcoeff=0,
        ),
        saveat=SaveAt(steps=True),
        # saveat=SaveAt(ts=ts),
        max_steps=max_steps,
        min_step_size=min_step_size,
        solver=solver,
        adjoint=adjoint,
        event=event,
        options={},
        chunk_size=particle_chunk_size,
        throw=False,
        return_aux=True,
    )

with nvtx.annotate("interpolation", color="red"):
    rtz2, _, aux2 = _trace_particles(
        field=interpolator,
        y0=x0,
        model=model,
        model_args=model_args,
        ts=ts,
        params=None,
        stepsize_controller=PIDController(
            rtol=1e-6,
            atol=1e-6,
            dtmin=min_step_size,
            pcoeff=0.3,
            icoeff=0.3,
            dcoeff=0,
        ),
        # stepsize_controller=ConstantStepSize(),
        saveat=SaveAt(steps=True),
        # saveat=SaveAt(ts=ts),
        max_steps=max_steps,
        min_step_size=min_step_size,
        solver=solver,
        adjoint=adjoint,
        event=event,
        options={},
        chunk_size=particle_chunk_size,
        throw=False,
        return_aux=True,
    )

with nvtx.annotate("interpolation", color="red"):
    rtz2, _, aux2 = _trace_particles(
        field=interpolator,
        y0=x0,
        model=model,
        model_args=model_args,
        ts=ts,
        params=None,
        stepsize_controller=PIDController(
            rtol=1e-6,
            atol=1e-6,
            dtmin=min_step_size,
            pcoeff=0.3,
            icoeff=0.3,
            dcoeff=0,
        ),
        # stepsize_controller=ConstantStepSize(),
        saveat=SaveAt(steps=True),
        # saveat=SaveAt(ts=ts),
        max_steps=max_steps,
        min_step_size=min_step_size,
        solver=solver,
        adjoint=adjoint,
        event=event,
        options={},
        chunk_size=particle_chunk_size,
        throw=False,
        return_aux=True,
    )

with nvtx.annotate("interpolation", color="red"):
    rtz2, _, aux2 = _trace_particles(
        field=interpolator,
        y0=x0,
        model=model,
        model_args=model_args,
        ts=ts,
        params=None,
        stepsize_controller=PIDController(
            rtol=1e-6,
            atol=1e-6,
            dtmin=min_step_size,
            pcoeff=0.3,
            icoeff=0.3,
            dcoeff=0,
        ),
        # stepsize_controller=ConstantStepSize(),
        saveat=SaveAt(steps=True),
        # saveat=SaveAt(ts=ts),
        max_steps=max_steps,
        min_step_size=min_step_size,
        solver=solver,
        adjoint=adjoint,
        event=event,
        options={},
        chunk_size=particle_chunk_size,
        throw=False,
        return_aux=True,
    )
