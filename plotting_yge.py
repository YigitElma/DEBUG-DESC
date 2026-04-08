"""Extra usefull plotting utilities for DESC."""

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from desc.plotting import plot_surfaces, sequential_colors
from desc.grid import LinearGrid


def plot_grid_3d(eq, grid, fig=None, **kwargs):
    """Plot the grid in 3D."""
    data = eq.compute(["X", "Y", "Z"], grid=grid)
    x = data["X"]
    y = data["Y"]
    z = data["Z"]

    if fig is None:
        fig = go.Figure()

    trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="markers",
        marker=dict(
            size=kwargs.pop("size", 2),
            color=kwargs.pop("color", "black"),
        ),
        showlegend=False,
    )

    fig.add_trace(trace)
    return fig


def plot_coil_and_surfaces(eq, coils, **kwargs):
    if not isinstance(coils, (list, tuple)):
        coils = [coils]
    datas = [coil.compute(["R", "Z", "phi"], grid=LinearGrid(zeta=6)) for coil in coils]
    colors = kwargs.pop("colors", sequential_colors)
    if not isinstance(colors, (list, tuple)):
        colors = [colors] * len(coils)
    fig, ax = plot_surfaces(eq, phi=6)
    for i in range(6):
        for k, data in enumerate(datas):
            ax[i].scatter(
                data["R"][i],
                data["Z"][i],
                marker=kwargs.pop("marker", "*"),
                color=colors[k],
                label=f"coil {k}",
            )
    plt.legend()
