# rnn2pwa/visualize/analysis_plots.py
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from rnn2pwa.models.rnn_relu import RNN, pattern_from_point

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
