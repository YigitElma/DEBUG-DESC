import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../../"))

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.99"
from desc import set_device

set_device("gpu")

from desc.backend import print_backend_info

print_backend_info()

import timeit

import numpy as np
import matplotlib.pyplot as plt
from diffrax import *
from desc.particles import *
from desc.particles import _trace_particles
from desc.examples import get
from desc.compat import rescale
from desc.io import load

name = str(sys.argv[1])
plot = True
# these will be used as 1e-T
ts_to_profile = [5, 4, 3, 2, 1]
Ns = [1, 10, 30, 100, 300, 500, 1000, 3000, 5000, 10000]
Ns = np.array(Ns)
np.savetxt(f"{name}-Ns.txt", Ns)
repeat = 5

try:
    # if the file exists, load it
    eq = load(f"{name}_vacuum_scaled_solved.h5")
    eqi_scaled = eq.copy()
except:
    # else, create it from scratch
    eqi = get(name)
    eq = rescale(eq=eqi, L=("a", 1.7044), B=("<B>", 5.86), copy=True)
    eq.pressure = 0
    eq.current = 0
    eq.solve(ftol=1e-4, verbose=1)
    eqi_scaled = eq.copy()
    eq.save(f"{name}_vacuum_scaled_solved.h5")

eq.iota = eq.get_profile("iota")
model = VacuumGuidingCenterTrajectory(frame="flux")


def default_event(t, y, args, **kwargs):
    i = jnp.sqrt(y[0] ** 2 + y[1] ** 2)
    return jnp.logical_or(i < 0.0, i > 1.0)


event = Event(default_event)

for T in ts_to_profile:
    ts = np.linspace(0, 10 ** (-T), 100)

    @jit
    def fun(x0, args):
        rpz, _ = _trace_particles(
            field=eq,
            y0=x0,
            model=model,
            model_args=args,
            ts=ts,
            params=eq.params_dict,
            max_steps=100000,
            min_step_size=1e-8,
            stepsize_controller=PIDController(rtol=1e-4, atol=1e-4, dtmin=1e-8),
            saveat=SaveAt(ts=ts),
            solver=Tsit5(),
            adjoint=RecursiveCheckpointAdjoint(),
            event=event,
            chunk_size=None,
            options={},
            throw=False,
            return_aux=False,
        )
        return rpz

    Ts = []
    for n in Ns:
        rhos = [0.5] * n
        initializer = ManualParticleInitializerFlux(
            rho0=rhos,
            theta0=0,
            zeta0=np.random.rand(n) * 2 * np.pi,
            xi0=2 * np.random.rand(n) - 1,
            E=3.5e6,
            m=4.0,
            q=2.0,
        )
        x0, args = initializer.init_particles(model=model, field=eq)
        _ = fun(x0, args).block_until_ready()  # compile
        fun_time = lambda: fun(x0, args).block_until_ready()
        t = timeit.timeit(fun_time, number=repeat)
        print(f"N={n:^7} and Tf=1e-{T} took {t/repeat:.4f} seconds per run")
        Ts.append(t / repeat)

    Ts = np.array(Ts)
    np.savetxt(f"{name}-ts_1e-{T}.txt", Ts)

if plot:
    Ns = np.loadtxt(f"{name}-Ns.txt")
    for T in ts_to_profile:
        Tsi = np.loadtxt(f"{name}-ts_1e-{T}.txt")
        plt.semilogx(Ns, Tsi, label=f"tf = 1e-{T}")
    plt.xlabel("Number of particles")
    plt.ylabel("Time per run (seconds)")
    plt.legend()
    plt.title(f"Time to trace particles in reactor size {name}")
    plt.savefig(f"{name}_time_to_trace_particles.png", dpi=500)

print("DONE!")
