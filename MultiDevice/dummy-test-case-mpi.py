#!/usr/bin/env python
import re
import os
NUM_DEVICE = 2
DEVICE_TYPE = "cpu"
xla_flags = os.getenv("XLA_FLAGS", "")
xla_flags = re.sub(
    r"--xla_force_host_platform_device_count=\S+", "", xla_flags
).split()
os.environ["XLA_FLAGS"] = " ".join(
    [f"--xla_force_host_platform_device_count={NUM_DEVICE}"] + xla_flags
)


import jax
import jax.numpy as jnp
import numpy as np
import functools
from jax.tree_util import register_pytree_node
import copy


def jit_with_device(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):

        # Compile the method with jax.jit for the specific device
        wrapped = jax.jit(method, device=self._device)
        return wrapped(self, *args, **kwargs)

    return wrapper


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

def pconcat(arrays, mode="concat"):
    if DEVICE_TYPE == "cpu":
        device = jax.devices("cpu")[0]
    else:
        import nvgpu
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

    if mode == "concat":
        out = jnp.concatenate([jax.device_put(x, device=device) for x in arrays])
    elif mode == "hstack":
        out = jnp.hstack([jax.device_put(x, device=device) for x in arrays])
    elif mode == "vstack":
        out = jnp.vstack([jax.device_put(x, device=device) for x in arrays])
    return out

class ObjectiveFunctionParallel:
    def __init__(self, objectives):
        self.objectives = objectives
        self.num_device = len(objectives)
        self.built = False
        targets = [obj.target for obj in self.objectives]
        self.target = pconcat(targets)

    def build(self):
        for obj in self.objectives:
            if not obj.built:
                obj.build()

        self.A = [obj.A for obj in self.objectives]
        self.built = True

    def compute(self, coefs=None, A=None):
        if A is None:
            A = self.A
        if coefs is None:
            coefs = [obj.opt.coefs for obj in self.objectives]
        fs = [
            obj.compute(jax.device_put(coefi, device=obj._device), Ai)
            for obj, coefi, Ai in zip(self.objectives, coefs, A)
        ]
        return pconcat(fs)

    def compute_error(self, coefs=None, A=None):
        if A is None:
            A = self.A
        if coefs is None:
            coefs = [obj.opt.coefs for obj in self.objectives]
        fs = [
            obj.compute_error(jax.device_put(coefi, device=obj._device), Ai)
            for obj, coefi, Ai in zip(self.objectives, coefs, A)
        ]
        return pconcat(fs)

    def jac_error(self, coefs=None, A=None):
        if A is None:
            A = self.A
        if coefs is None:
            coefs = [obj.opt.coefs for obj in self.objectives]

        fs = [
            obj.jac_error(jax.device_put(coefi, device=obj._device), Ai)
            for obj, coefi, Ai in zip(self.objectives, coefs, A)
        ]
        return pconcat(fs)

    def jac(self, coefs=None, A=None):
        if A is None:
            A = self.A
        if coefs is None:
            coefs = [obj.opt.coefs for obj in self.objectives]

        fs = [
            obj.jac(jax.device_put(coefi, device=obj._device), Ai)
            for obj, coefi, Ai in zip(self.objectives, coefs, A)
        ]
        return pconcat(fs)

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
    ObjectiveFunctionParallel,
    ObjectiveFunctionParallel._flatten,
    ObjectiveFunctionParallel._unflatten,
)

N = 40
num_nodes = 15
coefs = np.zeros(N)
coefs[2] = 3
eq = Optimizable(N, coefs)
grid1 = jnp.linspace(-jnp.pi, 0, num_nodes, endpoint=False)
grid2 = jnp.linspace(0, jnp.pi, num_nodes, endpoint=False)
grid3 = jnp.concatenate([grid1, grid2])
target1 = grid1**2
target2 = grid2**2
target3 = grid3**2

obj1 = Objective(eq, grid1, target1, device_id=0)
obj2 = Objective(eq, grid2, target2, device_id=1)
obj1.build()
obj2.build()

# we will put different objectives to different devices
obj1 = jax.device_put(obj1, obj1._device)
obj2 = jax.device_put(obj2, obj1._device)
obj1.opt = eq
obj2.opt = eq

objp_fun = ObjectiveFunctionParallel([obj1, obj2])
objp_fun.build()

objective = objp_fun
print(jnp.linalg.norm(objective.compute()))
step = 0
for k in range(3):
    J = objective.jac_error()
    f = objective.compute_error()
    eq.coefs = (
        eq.coefs
        - 1e-1 * jnp.linalg.pinv(J) @ f
    )
    step += 1
print(jnp.linalg.norm(objective.compute()))


