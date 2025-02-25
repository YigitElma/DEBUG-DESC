import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

import functools

DEVICE_TYPE = "cpu"


def pconcat(arrays, mode="concat"):
    """Concatenate arrays that live on different devices.

    Parameters
    ----------
    arrays : list of jnp.ndarray
        Arrays to concatenate.
    mode : str
        "concat:, "hstack" or "vstack. Default is "concat"

    Returns
    -------
    out : jnp.ndarray
        Concatenated array that lives on CPU.
    """
    if DEVICE_TYPE == "gpu":
        devices = nvgpu.gpu_info()
        mem_avail = devices[0]["mem_total"] - devices[0]["mem_used"]
        # we will use either CPU or GPU[0] for the matrix decompositions, so the
        # array of float64 should fit into single device
        size = jnp.array([x.size for x in arrays])
        size = jnp.sum(size)
        if size * 8 / (1024**3) > mem_avail:
            device = jax.devices("cpu")[0]
        else:
            device = jax.devices("gpu")[0]
    else:
        device = jax.devices("cpu")[0]

    if mode == "concat":
        out = jnp.concatenate([jax.device_put(x, device=device) for x in arrays])
    elif mode == "hstack":
        out = jnp.hstack([jax.device_put(x, device=device) for x in arrays])
    elif mode == "vstack":
        out = jnp.vstack([jax.device_put(x, device=device) for x in arrays])
    return out


# we want to JIT the methods of a class on specific devices
# for convenience, we define a decorator that does this for us, this will use
# the device_id attribute of the class to determine the device to JIT on
def jit_with_device(method):
    """Decorator to Just-in-time compile a class method with a dynamic device.

    Decorates a method of a class with a dynamic device, allowing the method to be
    compiled with jax.jit for the specific device. This is needed since
    @functools.partial(jax.jit, device=jax.devices("gpu")[self._device_id]) is not
    allowed in a class definition.

    Parameters
    ----------
    method : callable
        Class method to decorate. If DESC is running on GPU, the class should have
        a device_id attribute.
    """

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        device = self._device

        # Compile the method with jax.jit for the specific device
        wrapped = jax.jit(method, device=device)
        return wrapped(self, *args, **kwargs)

    return wrapper

from jax.tree_util import register_pytree_node
import copy


class Optimizable:
    def __init__(self, N, coefs):
        self.N = N
        self.coefs = coefs

    def N(self):
        return self.N

    def coefs(self):
        return self.coefs

    def copy(self):
        return copy.copy(self)

    def __repr__(self):
        return f"Optimizable(N={self.N}, coefs={self.coefs})"


def special_flatten_opt(obj):
    """Specifies a flattening recipe."""
    children = (obj.N, obj.coefs)
    aux_data = None
    return (children, aux_data)


def special_unflatten_opt(aux_data, children):
    """Specifies an unflattening recipe."""
    obj = object.__new__(Optimizable)
    obj.N = children[0]
    obj.coefs = children[1]
    return obj


class Objective:
    def __init__(self, opt, grid, target, device_id=0):
        self.opt = opt
        self.grid = grid
        self.target = target
        self.built = False
        self._device_id = device_id
        self._device = jax.devices(DEVICE_TYPE)[self._device_id]

    def build(self):
        # the transform matrix A such that A @ coefs gives the
        # values of the function at the grid points
        self.A = jnp.vstack([jnp.cos(i * self.grid) for i in range(self.opt.N)]).T
        self.built = True

    @jit_with_device
    def compute(self, coefs, A=None):
        if A is None:
            A = self.A
        vals = A @ coefs
        return vals

    @jit_with_device
    def compute_error(self, coefs, A=None):
        if A is None:
            A = self.A
        vals = A @ coefs
        return vals - self.target

    @jit_with_device
    def jac_error(self, coefs, A=None):
        if A is None:
            A = self.A
        return jax.jacfwd(self.compute_error)(coefs, A)

    @jit_with_device
    def jac(self, coefs, A=None):
        if A is None:
            A = self.A
        return jax.jacfwd(self.compute)(coefs, A)

def special_flatten_obj(obj):
    """Specifies a flattening recipe."""
    children = (obj.opt, obj.grid, obj.target, obj.A)
    aux_data = (obj.built, obj._device_id, obj._device)
    return (children, aux_data)


def special_unflatten_obj(aux_data, children):
    """Specifies an unflattening recipe."""
    obj = object.__new__(Objective)
    obj.opt = children[0]
    obj.grid = children[1]
    obj.target = children[2]
    obj.A = children[3]
    obj.built = aux_data[0]
    obj._device_id = aux_data[1]
    obj._device = aux_data[2]
    return obj


# Global registration
register_pytree_node(Optimizable, special_flatten_opt, special_unflatten_opt)
register_pytree_node(Objective, special_flatten_obj, special_unflatten_obj)

N = 40
num_nodes = 30
coefs = np.zeros(N)
coefs[2] = 3
eq = Optimizable(N, coefs)
grid = jnp.linspace(-jnp.pi, jnp.pi, num_nodes, endpoint=False)
target = grid**2
obj = Objective(eq, grid, target)
obj.build()

plt.plot(obj.target, "or", label="target")
plt.plot(obj.compute(eq.coefs, obj.A), label=f"iter 0")
step = 0
while jnp.linalg.norm(obj.compute_error(eq.coefs, obj.A)) > 1e-3:
    J = obj.jac_error(eq.coefs, obj.A)
    f = obj.compute_error(eq.coefs, obj.A)
    eq.coefs = eq.coefs - 1e-1 * jnp.linalg.pinv(J) @ f
    step += 1
plt.plot(obj.compute(eq.coefs, obj.A), label=f"iter last")
plt.legend()
plt.title(f"Converged in {step} steps")
plt.savefig("normal.png")

class ObjectiveFunctionMPI:
    def __init__(self, objectives, mpi):
        self.objectives = objectives
        self.num_device = len(objectives)
        self.built = False
        targets = [obj.target for obj in self.objectives]
        self.target = jnp.concatenate(targets)
        self.mpi = mpi
        self.comm = self.mpi.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        # assert self.size == len(self.objectives)
        self.running = True

    def __enter__(self):
        self.worker_loop()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.comm.bcast("STOP", root=0)
        self.running = False

    def worker_loop(self):
        if self.rank == 0:
            return  # Root rank won't enter worker loop
        while self.running:
            message = self.comm.bcast(None, root=0)
            if message == "STOP":
                print(f"Rank {self.rank} STOPPING")
                break
            elif message == "jac_error":
                print(f"Rank {self.rank} computing jac_error")
                self._compute_jac_error_worker()
            elif message == "jac":
                print(f"Rank {self.rank} computing jac")
                self._compute_jac_worker()

    def build(self):
        for obj in self.objectives:
            if not obj.built:
                obj.build()
        self.A = [obj.A for obj in self.objectives]
        self.built = True

    def compute(self, coefs=None, A=None):
        if self.rank == 0:
            if A is None:
                A = self.A
            if coefs is None:
                coefs = [obj.opt.coefs for obj in self.objectives]
            fs = [
                obj.compute(jax.device_put(coefi, device=obj._device), Ai)
                for obj, coefi, Ai in zip(self.objectives, coefs, A)
            ]
            return jnp.concatenate(fs)
        else:
            return None

    def compute_error(self, coefs=None, A=None):
        if self.rank == 0:
            if A is None:
                A = self.A
            if coefs is None:
                coefs = [obj.opt.coefs for obj in self.objectives]
            fs = [
                obj.compute_error(jax.device_put(coefi, device=obj._device), Ai)
                for obj, coefi, Ai in zip(self.objectives, coefs, A)
            ]
            return jnp.concatenate(fs)
        else:
            return None

    def jac_error(self, coefs=None, A=None):
        if self.rank == 0:
            self.comm.bcast("jac_error", root=0)
        if A is None:
            A = self.A
        if coefs is None:
            coefs = [obj.opt.coefs for obj in self.objectives]
        obj = self.objectives[self.rank]
        coefi = coefs[self.rank]
        Ai = A[self.rank]
        f = obj.jac_error(jax.device_put(coefi, device=obj._device), Ai)
        f = np.asarray(f)
        gathered = self.comm.gather(f, root=0)
        if self.rank == 0:
            return jnp.concatenate(gathered, axis=0)

    def _compute_jac_error_worker(self):
        obj = self.objectives[self.rank]
        coefs = obj.opt.coefs
        Ai = obj.A
        f = obj.jac_error(jax.device_put(coefs, device=obj._device), Ai)
        f = np.asarray(f)
        self.comm.gather(f, root=0)

    def jac(self, coefs=None, A=None):
        if self.rank == 0:
            self.comm.bcast("jac", root=0)
        if A is None:
            A = self.A
        if coefs is None:
            coefs = [obj.opt.coefs for obj in self.objectives]
        obj = self.objectives[self.rank]
        coefi = coefs[self.rank]
        Ai = A[self.rank]
        f = obj.jac(jax.device_put(coefi, device=obj._device), Ai)
        f = np.asarray(f)
        gathered = self.comm.gather(f, root=0)
        if self.rank == 0:
            return jnp.concatenate(gathered, axis=0)

    def _compute_jac_worker(self):
        obj = self.objectives[self.rank]
        coefs = obj.opt.coefs
        Ai = obj.A
        f = obj.jac(jax.device_put(coefs, device=obj._device), Ai)
        f = np.asarray(f)
        self.comm.gather(f, root=0)

    def _flatten(obj):
        """Specifies a flattening recipe."""
        children = (obj.objectives, obj.target, obj.A)
        aux_data = (obj.built,)
        return (children, aux_data)

    @classmethod
    def _unflatten(cls, aux_data, children):
        """Specifies an unflattening recipe."""
        cls.objectives = children[0]
        cls.target = children[1]
        cls.A = children[2]
        cls.built = aux_data[0]
        return cls


register_pytree_node(
    ObjectiveFunctionMPI,
    ObjectiveFunctionMPI._flatten,
    ObjectiveFunctionMPI._unflatten,
)

from mpi4py import MPI

# Example usage
if __name__ == "__main__":
    processes = 4
    N = 40
    num_nodes_per_worker = 10
    num_nodes = num_nodes_per_worker * processes
    coefs = np.zeros(N)
    coefs[2] = 3
    eq = Optimizable(N, coefs)
    objs = []
    full_grid = jnp.linspace(-jnp.pi, jnp.pi, num_nodes, endpoint=False)
    for i in range(processes):
        grid = full_grid[i * num_nodes_per_worker : (i + 1) * num_nodes_per_worker]
        target = grid**2
        obj = Objective(eq, grid, target, device_id=0)
        obj.build()
        obj = jax.device_put(obj, obj._device)
        obj.opt = eq
        objs.append(obj)

    with ObjectiveFunctionMPI(objs, mpi=MPI) as objective:
        objective.build()
        if objective.rank == 0:
            plt.figure()
            plt.plot(objective.target, "or", label="target")
            plt.plot(objective.compute(), label=f"iter 0")
            step = 0
            for _ in range(30):
                J = objective.jac_error()
                f = objective.compute_error()
                eq.coefs = eq.coefs - 1e-1 * jnp.linalg.pinv(J) @ f
                step += 1
            plt.plot(objective.compute(), label=f"iter last")
            plt.legend()
            plt.title(f"Converged in {step} steps")
            plt.savefig("mpi.png")
