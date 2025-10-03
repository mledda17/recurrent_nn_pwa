import numpy as np
import matplotlib.pyplot as plt
from rnn2pwa.models.rnn_relu import RNN, pattern_from_point

def plot_region_partition_2d(rnn: RNN, X_bounds, U_fixed, axes=(0,1), grid=200, show_boundaries=True, title=None):
    (x_lo, x_hi) = X_bounds
    n_x = rnn.n_x; ax1, ax2 = axes
    xs = np.linspace(x_lo[ax1], x_hi[ax1], grid)
    ys = np.linspace(x_lo[ax2], x_hi[ax2], grid)
    pat_to_id, next_id = {}, 0
    label = np.zeros((grid, grid), dtype=int)

    mid = (x_lo + x_hi) * 0.5
    for j, y in enumerate(ys):
        for i, x1 in enumerate(xs):
            x = mid.copy()
            x[ax1] = x1; x[ax2] = y
            pat = pattern_from_point(rnn, x, U_fixed)
            if pat not in pat_to_id:
                pat_to_id[pat] = next_id; next_id += 1
            label[j, i] = pat_to_id[pat]

    extent = [xs[0], xs[-1], ys[0], ys[-1]]
    plt.figure(figsize=(6,5))
    im = plt.imshow(label, origin="lower", extent=extent, interpolation="nearest", aspect="auto")
    plt.colorbar(im, label="Region ID")
    plt.xlabel(f"x[{ax1}]"); plt.ylabel(f"x[{ax2}]")
    plt.title(title or f"ReLU partition – regions: {len(pat_to_id)} (u fixed)")
    if show_boundaries:
        bx = np.zeros_like(label, dtype=bool); by = np.zeros_like(label, dtype=bool)
        bx[:,1:] = label[:,1:] != label[:, :-1]; by[1:,:] = label[1:,:] != label[:-1,:]
        edges = (bx | by).astype(float)
        yy, xx = np.meshgrid(ys, xs, indexing="ij")
        plt.contour(xx, yy, edges, levels=[0.5], linewidths=0.6)
    plt.tight_layout(); plt.show()
    return pat_to_id, label
