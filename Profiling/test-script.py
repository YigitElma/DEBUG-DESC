import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../../"))

if sys.argv[2] in ["GPU", "gpu"]:
    # Set the environment variable to use the GPU
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
    from desc import set_device

    set_device("gpu")

from desc.backend import print_backend_info
from desc.examples import get
from desc.objectives import ObjectiveFunction, ForceBalance

print_backend_info()

N = int(sys.argv[1])

eq = get("precise_QA")
eq.change_resolution(L=N, M=N, L_grid=2 * N, M_grid=2 * N)
eq.resolution_summary()
eq.set_initial_guess()
# obj = ObjectiveFunction(ForceBalance(eq), jac_chunk_size=None, deriv_mode="batched")
obj = ObjectiveFunction(ForceBalance(eq), jac_chunk_size=100, deriv_mode="batched")
obj.build()
print(f"Objective function deriv mode: {obj._deriv_mode}")
print(f"Objective function chunk size: {obj._jac_chunk_size}")

eq.solve(
    objective=obj,
    constraints=None,
    optimizer="lsq-exact",
    ftol=0,
    xtol=0,
    gtol=0,
    maxiter=int(sys.argv[3]),
    verbose=3,
    copy=False,
)
