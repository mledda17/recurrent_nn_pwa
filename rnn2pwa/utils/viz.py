from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from ..types import PWA

def plot_regions_2d(pwa: PWA, X_bounds, U_bounds, resolution: int = 200):
    """
    Visualize region labels over a 2D domain only when n+m == 2.
    """
    n = pwa.meta["n"]; m = pwa.meta["m"]
    assert n + m == 2, "plot_regions_2d supports total dimension 2 only."
    lo = np.concatenate([X_bounds[0], U_bounds[0]])
    hi = np.concatenate([X_bounds[1], U_bounds[1]])
    xs = np.linspace(lo[0], hi[0], resolution)
    ys = np.linspace(lo[1], hi[1], resolution)
    fig, ax = plt.subplots()
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1])
    ax.set_xlabel("z1"); ax.set_ylabel("z2")

    # paint feasible regions using witness signs
    for reg in pwa.regions.values():
        w = reg.witness
        if w is None:
            continue
        ax.plot(w[0], w[1], ".", ms=6)

    ax.set_title(f"{len(pwa.regions)} feasible regions")
    ax.grid(True, alpha=0.2)
    return fig
