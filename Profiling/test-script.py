import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../"))


# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

# from desc import set_device

# set_device("gpu")

from desc.backend import print_backend_info
from desc.examples import get
from desc.objectives import ObjectiveFunction, ForceBalance

print_backend_info()

N = 10

eq = get("precise_QA")
eq.change_resolution(L=N, M=N, N=N, L_grid=2 * N, M_grid=2 * N, N_grid=2 * N)
eq.resolution_summary()
eq.set_initial_guess()
obj = ObjectiveFunction(ForceBalance(eq))
# obj = ObjectiveFunction(ForceBalance(eq), jac_chunk_size=200, deriv_mode="batched")
obj.build()
print(f"Objective function deriv mode: {obj._deriv_mode}")
print(f"Objective function chunk size: {obj._jac_chunk_size}")

eq.solve(
    objective=obj,
    constraints=None,
    optimizer="lsq-exact",
    ftol=1e-4,
    xtol=1e-6,
    gtol=1e-6,
    maxiter=8,
    x_scale="auto",
    verbose=2,
    copy=False,
)
