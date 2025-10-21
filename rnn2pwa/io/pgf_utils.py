import os
import numpy as np
from typing import List, Dict, Tuple
from rnn2pwa.models.rnn_relu import RNN, pattern_from_point

def export_feasible_regions_xu_dat(
    rnn: RNN,
    patterns: List[Tuple[Tuple[int, ...], ...]],
    X_bounds: Tuple[np.ndarray, np.ndarray],
    U_bounds: Tuple[np.ndarray, np.ndarray],
    grid: int = 400,
    x_axis: int = 0,
    outdir: str = "plots",
    basename: str = "xu_regions",
):
    """
    Writes:
      - plots/xu_regions_grid.dat   (cols: x  u  reg)
      - plots/xu_regions_wit.dat    (cols: x  u)  [LP witness points]
    """
    os.makedirs(outdir, exist_ok=True)

    n_x, n_u = rnn.n_x, rnn.n_u
    x_lo, x_hi = X_bounds
    u_lo, u_hi = U_bounds

    xs = np.linspace(x_lo[x_axis], x_hi[x_axis], grid)
    us = np.linspace(u_lo[0], u_hi[0], grid) if n_u == 1 else np.linspace(u_lo[0], u_hi[0], grid)

    feasible_set = set(patterns)
    pat_to_id = {pat: i for i, pat in enumerate(patterns)}

    # label grid (-1 = infeasible/unseen by LP)
    lab = -np.ones((grid, grid), dtype=int)

    # other state dims fixed at midpoint
    x_fixed = (x_lo + x_hi) / 2.0

    for j, u_val in enumerate(us):
        for i, x_val in enumerate(xs):
            x = x_fixed.copy()
            x[x_axis] = x_val
            u = np.array([u_val]) if n_u == 1 else np.full(n_u, u_val)
            pat = pattern_from_point(rnn, x, u)
            if pat in feasible_set:
                lab[j, i] = pat_to_id[pat]

    # Dump grid (x,u,reg) for PGFPlots
    Xs, Us = np.meshgrid(xs, us)             # shapes (grid, grid)
    grid_path = os.path.join(outdir, f"{basename}_grid.dat")
    with open(grid_path, "w") as f:
        f.write("x\tu\treg\n")
        for j in range(grid):
            for i in range(grid):
                f.write(f"{Xs[j,i]:.10g}\t{Us[j,i]:.10g}\t{int(lab[j,i])}\n")

    # Optional: LP witnesses if you collected them in your caller
    wit_path = os.path.join(outdir, f"{basename}_wit.dat")
    # The caller will pass witnesses separately if desired; we keep a placeholder filename here.
    return grid_path, wit_path


def export_trajectories_dat(
    t: np.ndarray,
    X_pwa: np.ndarray,
    X_rnn: np.ndarray,
    U: np.ndarray,
    outdir: str = "plots",
    basename: str = "traj"
):
    """
    Writes:
      plots/traj_x_pwa.dat   (t, x1, x2, ...)
      plots/traj_x_rnn.dat   (t, x1, x2, ...)
      plots/traj_u.dat       (t, u1, u2, ...)
    Notes:
      - X_* are typically length T+1; U is length T. We align automatically:
        * time_x = t[:len(X_*)]
        * time_u = t[:len(U)] if len(U) == len(t) else t[:-1] if len(U) == len(t)-1 else t[:len(U)]
    """
    os.makedirs(outdir, exist_ok=True)

    t = np.asarray(t).reshape(-1)  # 1D
    Tt = t.shape[0]

    def _ensure_2d(a):
        a = np.asarray(a)
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        return a

    def _write_series(fname, time_vec, arr, prefix):
        arr = _ensure_2d(arr)
        time_vec = np.asarray(time_vec).reshape(-1, 1)
        # Trim/pad to the shortest along rows
        n = min(time_vec.shape[0], arr.shape[0])
        time_vec = time_vec[:n]
        arr = arr[:n]
        cols = [f"{prefix}{i+1}" for i in range(arr.shape[1])]
        header = "t\t" + "\t".join(cols)
        path = os.path.join(outdir, fname)
        with open(path, "w") as f:
            f.write(header + "\n")
            np.savetxt(f, np.hstack([time_vec, arr]), fmt="%.10g", delimiter="\t")
        print(f"[PGF] wrote {path} (rows={n}, cols={arr.shape[1]+1})")

    # --- States: use full t (usually length T+1)
    if X_pwa is not None:
        X_pwa = _ensure_2d(X_pwa)
        time_x = t[:X_pwa.shape[0]]
        _write_series(f"{basename}_x_pwa.dat", time_x, X_pwa, "x")

    if X_rnn is not None:
        X_rnn = _ensure_2d(X_rnn)
        time_x = t[:X_rnn.shape[0]]
        _write_series(f"{basename}_x_rnn.dat", time_x, X_rnn, "x")

    # --- Inputs: align to length of U
    if U is not None:
        U = _ensure_2d(U)
        Tu = U.shape[0]
        if Tu == Tt:
            time_u = t  # same length
        elif Tu == Tt - 1:
            time_u = t[:-1]  # standard case
        else:
            # Fallback: take first Tu samples of t
            time_u = t[:Tu]
        _write_series(f"{basename}_u.dat", time_u, U, "u")
