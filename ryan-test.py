import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.append(os.path.abspath("../"))

import numpy as np
from desc.coils import SplineXYZCoil, LinearGrid

npts = 1000
N = 50

# Create a SplineXYZCoil of a planar ellipse
s = np.linspace(0, 2 * np.pi, npts)
X = 2 * np.cos(s)
Y = np.ones(npts)
Z = 1 * np.sin(s)
c = SplineXYZCoil(X=X, Y=Y, Z=Z, current=1e6)

# Create a backwards SplineXYZCoil by flipping the coordinates
c_backwards = SplineXYZCoil(
    X=np.flip(X),
    Y=np.flip(Y),
    Z=np.flip(Z),
    current=1e6,
)

# Convert to FourierPlanarCoil
c_planar = c.to_FourierPlanar(N=N, grid=npts, basis="xyz")
c_backwards_planar = c_backwards.to_FourierPlanar(N=N, grid=npts, basis="xyz")
print(c_planar.normal)
print(c_backwards_planar.normal)

grid = LinearGrid(zeta=100)

field_spline = c.compute_magnetic_field(np.zeros((1, 3)), source_grid=grid, basis="xyz")
field_planar = c_planar.compute_magnetic_field(
    np.zeros((1, 3)), source_grid=grid, basis="xyz"
)
field_backwards_spline = c_backwards.compute_magnetic_field(
    np.zeros((1, 3)), source_grid=grid, basis="xyz"
)
field_backwards_planar = c_backwards_planar.compute_magnetic_field(
    np.zeros((1, 3)), source_grid=grid, basis="xyz"
)

np.testing.assert_allclose(
    field_spline,
    field_planar,
    atol=1e-5,
)
np.testing.assert_allclose(
    field_backwards_spline,
    field_backwards_planar,
    atol=1e-5,
)
np.testing.assert_allclose(
    field_planar,
    -field_backwards_planar,
    atol=1e-5,
)
