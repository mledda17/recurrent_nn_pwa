import numpy as np
from typing import List, Dict, Tuple
import matplotlib as mpl
import matplotlib.colors as mcolors
from rnn2pwa.models.rnn_relu import RNN

mpl.rcParams.update({
    # --- Sizes and layout ---
    "font.size": 8.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,

    # --- Lines and figure ---
    "axes.linewidth": 0.7,
    "lines.linewidth": 0.8,
    "lines.markersize": 3,
    "figure.figsize": (3.35, 2.5),
    "figure.dpi": 200,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

def compute_partition_labels_2d(
    rnn: RNN,
    X_bounds: Tuple[np.ndarray, np.ndarray],
    U_fixed: np.ndarray,
    axes=(0,1),
    grid: int = 250,
):
    (x_lo, x_hi) = X_bounds
    n_x = rnn.n_x
    a1, a2 = axes
    xs = np.linspace(x_lo[a1], x_hi[a1], grid)
    ys = np.linspace(x_lo[a2], x_hi[a2], grid)

    pat_to_id, next_id = {}, 0
    labels = np.zeros((grid, grid), dtype=int)

    mid = (x_lo + x_hi)*0.5
    for j, y in enumerate(ys):
        for i, x1 in enumerate(xs):
            x = mid.copy()
            x[a1] = x1; x[a2] = y
            pat = pattern_from_point(rnn, x, U_fixed)
            if pat not in pat_to_id:
                pat_to_id[pat] = next_id; next_id += 1
            labels[j, i] = pat_to_id[pat]
    return xs, ys, labels, pat_to_id

def plot_partition_with_paths(
    rnn: RNN,
    X_bounds: Tuple[np.ndarray, np.ndarray],
    U_fixed: np.ndarray,
    X_rnn: np.ndarray,
    X_pwa: np.ndarray = None,
    axes=(0,1),
    grid: int = 250,
    title: str = None,
):
    xs, ys, labels, pat_to_id = compute_partition_labels_2d(
        rnn, X_bounds, U_fixed, axes=axes, grid=grid
    )
    extent = [xs[0], xs[-1], ys[0], ys[-1]]
    plt.figure(figsize=(6.5, 5.2))
    # Colormap pastello
    im = plt.imshow(
        labels,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        aspect="auto",
        cmap="Pastel1",
        alpha=0.95,
    )
    # Boundaries sottili
    bx = np.zeros_like(labels, dtype=bool); by = np.zeros_like(labels, dtype=bool)
    bx[:,1:] = labels[:,1:] != labels[:, :-1]; by[1:,:] = labels[1:,:] != labels[:-1,:]
    edges = (bx | by).astype(float)
    YY, XX = np.meshgrid(ys, xs, indexing="ij")
    plt.contour(XX, YY, edges, levels=[0.5], linewidths=0.6, colors="k", alpha=0.4)

    a1, a2 = axes
    # Traiettoria RNN (linea piena)
    plt.plot(X_rnn[:, a1], X_rnn[:, a2], "-", lw=1.8, alpha=0.95, label="RNN")
    # Traiettoria PWA (linea tratteggiata)
    if X_pwa is not None:
        plt.plot(X_pwa[:, a1], X_pwa[:, a2], "--", lw=1.6, alpha=0.95, label="PWA")

    plt.xlabel(f"$x_{a1+1}$")
    plt.ylabel(f"$x_{a2+1}$")
    ttl = title or f"Partizione ReLU (u fisso={np.array2string(U_fixed, precision=2)}) – regioni: {len(pat_to_id)}"
    plt.title(ttl)
    cbar = plt.colorbar(im)
    cbar.set_label("ID regione")
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()
    plt.show()

def plot_region_visit_hist(region_seq: List):
    # ID compatti in ordine di frequenza
    from collections import Counter
    cnt = Counter(region_seq)
    pats, freqs = zip(*cnt.most_common())
    ids = list(range(len(pats)))
    plt.figure(figsize=(7.2, 3.4))
    plt.bar(ids, freqs, color="#87CEFA", alpha=0.8, edgecolor="#336699")
    plt.xlabel("ID regione (ordinati per frequenza)")
    plt.ylabel("Occorrenze")
    plt.title(f"Distribuzione visite regioni  |  #regioni={len(pats)}")
    plt.tight_layout()
    plt.show()

def plot_transition_matrix(region_seq: List):
    # Mappa pattern->ID compatto
    uniq = list({p for p in region_seq})
    idmap = {p:i for i,p in enumerate(uniq)}
    K = len(uniq)
    M = np.zeros((K, K), dtype=int)
    for k in range(len(region_seq)-1):
        i = idmap[region_seq[k]]
        j = idmap[region_seq[k+1]]
        M[i, j] += 1
    plt.figure(figsize=(5.6, 4.8))
    plt.imshow(M, cmap="Blues", interpolation="nearest")
    plt.colorbar(label="conteggi transizione")
    plt.xlabel("to")
    plt.ylabel("from")
    plt.title("Matrice di transizione regioni (conteggi)")
    plt.tight_layout()
    plt.show()

def plot_input_signal(time: np.ndarray, U: np.ndarray):
    plt.figure(figsize=(7.2, 2.8))
    if U.ndim == 1 or U.shape[1] == 1:
        u = U.reshape(-1)
        plt.plot(time[:-1], u, "-", lw=1.8)
        plt.ylabel("$u$")
    else:
        for i in range(U.shape[1]):
            plt.plot(time[:-1], U[:, i], lw=1.8, label=f"$u_{i+1}$")
        plt.ylabel("ingresso")
        plt.legend(frameon=False)
    plt.xlabel("k")
    plt.title("Sequenza di ingresso")
    plt.tight_layout()
    plt.show()

def plot_phase_plane(X_rnn: np.ndarray, X_pwa: np.ndarray = None, axes=(0,1)):
    a1, a2 = axes
    plt.figure(figsize=(6.2,4.8))
    plt.plot(X_rnn[:, a1], X_rnn[:, a2], "-", lw=1.9, label="RNN")
    if X_pwa is not None:
        plt.plot(X_pwa[:, a1], X_pwa[:, a2], "--", lw=1.7, label="PWA")
    plt.xlabel(f"$x_{a1+1}$")
    plt.ylabel(f"$x_{a2+1}$")
    plt.title("Piano di fase")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

def plot_partition_2d(
    rnn: RNN,
    X_bounds: Tuple[np.ndarray, np.ndarray],
    U_fixed: np.ndarray,
    axes=(0, 1),
    grid: int = 250,
    title: str = None,
    cmap: str = "Pastel1",
):
    """
    Disegna SOLO la partizione ReLU su una sezione 2D del piano degli stati (u fissato).
    Nessuna traiettoria viene sovrapposta.
    """
    xs, ys, labels, pat_to_id = compute_partition_labels_2d(
        rnn, X_bounds, U_fixed, axes=axes, grid=grid
    )
    extent = [xs[0], xs[-1], ys[0], ys[-1]]

    plt.figure(figsize=(6.5, 5.2))
    im = plt.imshow(
        labels,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        aspect="auto",
        cmap=cmap,
        alpha=0.95,
    )

    # contorni sottili e poco invadenti
    bx = np.zeros_like(labels, dtype=bool)
    by = np.zeros_like(labels, dtype=bool)
    bx[:, 1:] = labels[:, 1:] != labels[:, :-1]
    by[1:, :] = labels[1:, :] != labels[:-1, :]
    edges = (bx | by).astype(float)

    YY, XX = np.meshgrid(ys, xs, indexing="ij")
    plt.contour(XX, YY, edges, levels=[0.5], linewidths=0.6, colors="k", alpha=0.35)

    a1, a2 = axes
    plt.xlabel(f"$x_{a1+1}$")
    plt.ylabel(f"$x_{a2+1}$")
    ttl = title or f"Partizione ReLU (u fisso={np.array2string(U_fixed, precision=2)}) – regioni: {len(pat_to_id)}"
    plt.title(ttl)

    cbar = plt.colorbar(im)
    cbar.set_label("ID regione")

    plt.tight_layout()
    plt.show()
    return pat_to_id, labels

def plot_partition_xu(rnn, X_bounds, U_bounds, grid=400, title=None):
    from rnn2pwa.models.rnn_relu import pattern_from_point
    xlo, xhi = float(X_bounds[0]), float(X_bounds[1])
    ulo, uhi = float(U_bounds[0]), float(U_bounds[1])
    xs = np.linspace(xlo, xhi, grid)
    us = np.linspace(ulo, uhi, grid)
    pat_to_id, next_id = {}, 0
    lab = np.zeros((grid, grid), dtype=int)
    for j, u in enumerate(us):
        for i, x in enumerate(xs):
            pat = pattern_from_point(rnn, np.array([x]), np.array([u]))
            if pat not in pat_to_id:
                pat_to_id[pat] = next_id; next_id += 1
            lab[j, i] = pat_to_id[pat]
    extent = [xs[0], xs[-1], us[0], us[-1]]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(6.2,5.2))
    im = plt.imshow(lab, origin="lower", extent=extent, aspect="auto",
                    interpolation="nearest", cmap="Pastel1")
    plt.colorbar(im, label="ID regione")
    # contorni
    bx = np.zeros_like(lab, dtype=bool); by = np.zeros_like(lab, dtype=bool)
    bx[:,1:] = lab[:,1:] != lab[:, :-1]; by[1:,:] = lab[1:,:] != lab[:-1,:]
    edges = (bx | by).astype(float)
    U, X = np.meshgrid(us, xs, indexing="ij")
    plt.contour(X, U, edges, levels=[0.5], linewidths=0.7, colors="k", alpha=0.45)
    plt.xlabel("x"); plt.ylabel("u")
    plt.title(title or f"Partizione ReLU in (x,u) | #regioni visibili={len(pat_to_id)}")
    plt.tight_layout(); plt.show()
    return pat_to_id, lab


import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from rnn2pwa.models.rnn_relu import RNN, pattern_from_point


def plot_feasible_regions_xu(
        rnn,
        patterns,
        witnesses,
        X_bounds,
        U_bounds,
        grid=400,
        title=None,
        x_axis=0,
        save=False,
        outdir="plots",
        basename="xu_regions"
):
    import os
    from scipy import ndimage
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    n_x = rnn.n_x
    n_u = rnn.n_u

    ulo, uhi = float(U_bounds[0][0]), float(U_bounds[1][0])
    xlo, xhi = float(X_bounds[0][x_axis]), float(X_bounds[1][x_axis])

    xs = np.linspace(xlo, xhi, grid)
    us = np.linspace(ulo, uhi, grid)

    feasible_set = set(patterns)
    pat_to_id = {pat: i for i, pat in enumerate(patterns)}
    lab = -np.ones((grid, grid), dtype=int)
    x_fixed = (X_bounds[0] + X_bounds[1]) / 2.0

    from rnn2pwa.models.rnn_relu import pattern_from_point

    # Sampling
    for j, u_val in enumerate(us):
        for i, x_val in enumerate(xs):
            x = x_fixed.copy(); x[x_axis] = x_val
            u = np.array([u_val])
            pat = pattern_from_point(rnn, x, u)
            if pat in feasible_set:
                lab[j, i] = pat_to_id[pat]

    extent = [xs[0], xs[-1], us[0], us[-1]]
    plt.figure(figsize=(7, 6))
    lab_masked = np.ma.masked_where(lab == -1, lab)
    n_regions = len(patterns)

    # --- Colormap base ---
    base_cmap = plt.cm.get_cmap("tab20", n_regions)
    norm = mcolors.BoundaryNorm(boundaries=np.arange(-0.5, n_regions + 0.5, 1), ncolors=n_regions)
    im = plt.imshow(
        lab_masked, origin="lower", extent=extent, aspect="auto",
        cmap=base_cmap, norm=norm, interpolation="nearest", alpha=0.9
    )

    # ======================================================
    #   BORDI colorati con il colore scuro della regione
    # ======================================================
    colors = [mcolors.to_rgb(base_cmap(i)) for i in range(n_regions)]

    # edge detection per ogni regione
    for region_id in range(n_regions):
        mask = lab == region_id
        if not np.any(mask):
            continue
        # calcola i bordi del cluster
        edge_mask = np.zeros_like(mask, dtype=bool)
        edge_mask[:, 1:] |= mask[:, 1:] != mask[:, :-1]
        edge_mask[1:, :] |= mask[1:, :] != mask[:-1, :]
        # colorazione bordo con versione più scura
        r, g, b = colors[region_id]
        dark = (r * 0.6, g * 0.6, b * 0.6)
        Y, X = np.meshgrid(us, xs, indexing="ij")
        plt.contour(
            X, Y, edge_mask.astype(float),
            levels=[0.5],
            colors=[dark],
            linewidths=0.9,
            alpha=0.9
        )

    plt.xlabel(r"$x$")
    plt.ylabel(r"$u$")
    plt.xticks([-2, 0, 2])
    plt.yticks([-2, 0, 2])
    plt.title(title or "Feasible ReLU Regions in (x,u)")

    # --- Etichette centrali ---
    for region_id, pat in enumerate(patterns):
        mask = lab == region_id
        if not np.any(mask):
            continue
        cy, cx = ndimage.center_of_mass(mask)
        if np.isnan(cx) or np.isnan(cy):
            continue
        x_c = xs[int(round(cx))]
        u_c = us[int(round(cy))]
        pattern_str = "|".join("".join(map(str, sub)) for sub in pat)
        plt.text(
            x_c, u_c, f"{region_id}\n{pattern_str}",
            ha="center", va="center", fontsize=6.5,
            color="black", weight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.5, lw=0)
        )

    cbar = plt.colorbar(im, ticks=np.arange(n_regions))
    cbar.set_label("Region ID")

    plt.tight_layout()

    # ====== SAVE OPTION ======
    if save:
        os.makedirs(outdir, exist_ok=True)
        pdf_path = os.path.join(outdir, f"{basename}.pdf")
        png_path = os.path.join(outdir, f"{basename}.png")
        plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
        plt.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.02)
        print(f"[SAVED] {pdf_path}\n[SAVED] {png_path}")

    plt.show()



def plot_feasible_regions_2d_state(
        rnn: RNN,
        patterns: List[Tuple[Tuple[int, ...], ...]],
        witnesses: Dict,
        X_bounds: Tuple[np.ndarray, np.ndarray],
        U_fixed: np.ndarray,
        axes: Tuple[int, int] = (0, 1),
        grid: int = 300,
        title: str = None,
):
    """
    Visualize feasible regions in 2D state space with fixed input.
    Only shows patterns that were proven feasible via LP.
    """
    (x_lo, x_hi) = X_bounds
    ax1, ax2 = axes

    xs = np.linspace(x_lo[ax1], x_hi[ax1], grid)
    ys = np.linspace(x_lo[ax2], x_hi[ax2], grid)

    feasible_set = set(patterns)
    pat_to_id = {pat: i for i, pat in enumerate(patterns)}

    lab = -np.ones((grid, grid), dtype=int)
    mid = (x_lo + x_hi) * 0.5

    for j, y in enumerate(ys):
        for i, x1 in enumerate(xs):
            x = mid.copy()
            x[ax1] = x1
            x[ax2] = y
            pat = pattern_from_point(rnn, x, U_fixed)

            if pat in feasible_set:
                lab[j, i] = pat_to_id[pat]

    extent = [xs[0], xs[-1], ys[0], ys[-1]]
    plt.figure(figsize=(6.5, 5.2))

    lab_masked = np.ma.masked_where(lab == -1, lab)
    im = plt.imshow(
        lab_masked,
        origin="lower",
        extent=extent,
        interpolation="nearest",
        aspect="auto",
        cmap="tab20",
        alpha=0.95
    )

    # Boundaries
    bx = np.zeros_like(lab, dtype=bool)
    by = np.zeros_like(lab, dtype=bool)
    bx[:, 1:] = (lab[:, 1:] != lab[:, :-1]) & (lab[:, 1:] != -1) & (lab[:, :-1] != -1)
    by[1:, :] = (lab[1:, :] != lab[:-1, :]) & (lab[1:, :] != -1) & (lab[:-1, :] != -1)
    edges = (bx | by).astype(float)

    YY, XX = np.meshgrid(ys, xs, indexing="ij")
    plt.contour(XX, YY, edges, levels=[0.5], linewidths=0.7, colors="k", alpha=0.4)

    plt.xlabel(f"$x_{ax1 + 1}$")
    plt.ylabel(f"$x_{ax2 + 1}$")

    n_feasible = len(patterns)
    n_visible = np.sum(lab >= 0)
    coverage = 100 * n_visible / (grid * grid)

    #plt.colorbar(im, label="Region ID")
    plt.tight_layout()
    plt.show()

    return pat_to_id, lab


import os
import numpy as np
from typing import List, Tuple, Optional

def export_trajectory_xu_path_dat(
    t: np.ndarray,
    X: np.ndarray,              # (T+1, n_x)
    U: np.ndarray,              # (T, n_u) oppure (T+1, n_u)
    outdir: str = "plots",
    basename: str = "traj_xu_path",
    x_axis: int = 0,            # quale stato mettere sull’asse x
    u_axis: int = 0,            # quale ingresso sull’asse u
    region_ids: Optional[np.ndarray] = None,   # (K,) opzionale: id regione lungo traiettoria
    thin_markers: int = 1       # ogni quanti campioni mettere marker distinti (per alleggerire)
):
    """
    Esporta la traiettoria nello spazio (x,u) come:
      - plots/traj_xu_path.dat   colonne: k  t  x  u  reg
        (reg = -1 se non fornito)
      - plots/traj_xu_start.dat  (un solo punto: start)
      - plots/traj_xu_end.dat    (un solo punto: end)

    La polilinea in PGFPlots si ottiene con \addplot table ... di 'traj_xu_path.dat'.
    """
    os.makedirs(outdir, exist_ok=True)

    # Assicura 1D per t
    t = np.asarray(t).reshape(-1)
    X = np.asarray(X)
    U = np.asarray(U)

    # Allinea lunghezze: usiamo K = min(len(U), len(X))
    # e abbiniamo (x_k, u_k) per k=0..K-1; il tempo è t[:K]
    K = min(X.shape[0], U.shape[0])
    k_vec = np.arange(K)
    t_vec = t[:K]
    x_vec = X[:K, x_axis].reshape(-1)
    u_vec = U[:K, u_axis].reshape(-1)

    if region_ids is None:
        reg = -np.ones(K, dtype=int)
    else:
        reg = np.asarray(region_ids).reshape(-1)[:K]
        if reg.shape[0] < K:
            pad = -np.ones(K - reg.shape[0], dtype=int)
            reg = np.concatenate([reg, pad], axis=0)

    # Scrivi file principale (per linea + eventuali marker colorati)
    path_path = os.path.join(outdir, f"{basename}.dat")
    with open(path_path, "w") as f:
        f.write("k\tt\tx\tu\treg\n")
        for k, tt, xv, uv, rv in zip(k_vec, t_vec, x_vec, u_vec, reg):
            f.write(f"{int(k)}\t{tt:.10g}\t{xv:.10g}\t{uv:.10g}\t{int(rv)}\n")

    # Start / End (comodi per marker dedicati)
    start_path = os.path.join(outdir, f"{basename}_start.dat")
    end_path   = os.path.join(outdir, f"{basename}_end.dat")
    np.savetxt(start_path, np.array([[x_vec[0], u_vec[0]]]),
               header="x\tu", fmt="%.10g", comments="")
    np.savetxt(end_path,   np.array([[x_vec[-1], u_vec[-1]]]),
               header="x\tu", fmt="%.10g", comments="")

    # (Opzionale) versione "snellita" solo per marker radi
    if thin_markers > 1:
        idx = np.arange(K)[::thin_markers]
        thin_path = os.path.join(outdir, f"{basename}_marks_thin.dat")
        with open(thin_path, "w") as f:
            f.write("k\tt\tx\tu\treg\n")
            for i in idx:
                f.write(f"{int(k_vec[i])}\t{t_vec[i]:.10g}\t{x_vec[i]:.10g}\t{u_vec[i]:.10g}\t{int(reg[i])}\n")
    else:
        thin_path = None

    print(f"[PGF] wrote {path_path}")
    print(f"[PGF] wrote {start_path}")
    print(f"[PGF] wrote {end_path}")
    if thin_path:
        print(f"[PGF] wrote {thin_path}")

    return path_path, start_path, end_path, thin_path
