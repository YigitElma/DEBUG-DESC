import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../"))


import functools
from desc.backend import jax, jnp
import time
import gc

l1 = 10000
l2 = 5000


@functools.partial(jax.jit, static_argnames=("dim1", "dim2"))
def fun(dim1=100_000, dim2=1000):
    data = {}
    sum = 0
    for i in range(l1):  # fixed 10 intermediates
        key = f"intermediate_{i}"
        data[key] = (i + 1) * jnp.ones((dim1, dim2), dtype=jnp.float64)
    for i in range(l1):
        key = f"intermediate_{i}"
        data[key] += jnp.sum(jnp.sum(data[key])) * 2.0  # just simulate more ops

    for i in range(l1):
        key = f"intermediate_{i}"
        sum += jnp.sum(jnp.sum(data[key]))

    data["intermediate_0"] *= sum
    return jnp.sum(jnp.sum(data["intermediate_0"]))  # return only one


def shape2memory(array):
    return array.size * array.itemsize / 1024**3  # Convert bytes to GB


dim1 = 100000
dim2 = 10000
a = jnp.ones((dim1, dim2), dtype=jnp.float64)
print(f"Memory usage for a: {shape2memory(a):.2f} GB")
print(f"All data should take {shape2memory(a) * l1:.2f} GB")

del a
res = fun(dim1, dim2).block_until_ready()
print(res)
del res
print("Sleeping...")
gc.collect()
time.sleep(5)

# for i in range(5):
#     res = fun(dim1, dim2).block_until_ready()
#     print(res)
#     del res
#     print("Sleeping...")
#     gc.collect()
#     time.sleep(5)
