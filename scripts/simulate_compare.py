# scripts/simulate_compare.py
import numpy as np
from rnn2pwa.models.rnn_relu import RNN, Layer
from rnn2pwa.regions.adjacency import build_structural_adjacency
from rnn2pwa.utils.discovery import discover_regions, build_local_dynamics_map
from rnn2pwa.utils.signals import constant, step, ramp, impulse, sine, multi_sine, prbs
from rnn2pwa.simulate.rollout import simulate_rnn, simulate_pwa_from_patterns, build_transition_graph
from rnn2pwa.visualize.trajectories import plot_trajectories, plot_error
from rnn2pwa.visualize.graph import plot_graph_networkx
from rnn2pwa.visualize.style import set_paper_style
from rnn2pwa.visualize.analysis_plots import (
    plot_partition_2d,
    plot_input_signal,
)


if __name__ == "__main__":
    np.random.seed(0)
    set_paper_style()

    # --- Rete esempio ---
    n_x, n_u = 2, 1
    W1 = np.array([[ 0.6, -0.2,  0.5],
                   [-0.1,  0.9, -0.3],
                   [ 0.2,  0.4,  0.1]])
    b1 = np.array([0.0, 0.1, -0.05])
    W2 = np.array([[ 1.0,  0.5, -0.2],
                   [ 0.3, -0.7,  0.4]])
    b2 = np.array([0.0, 0.0])
    rnn = RNN(layers=[Layer(W1,b1), Layer(W2,b2)], n_x=n_x, n_u=n_u)

    # --- Domini ---
    X_bounds = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))
    U_bounds = (np.array([-0.5]), np.array([0.5]))

    # --- Scoperta regioni e mappe locali ---
    patterns = discover_regions(rnn, X_bounds, U_bounds, N=4000)
    dyn_map  = build_local_dynamics_map(rnn, patterns)

    # --- Ingresso e simulazioni ---
    # --- Ingresso e simulazioni ---
    T = 200

    # Input signal type
    # 0 - Constant | 1 - Step | 2 - Ramp | 3 - Impulse
    # 4 - Sine     | 5 - Multisine | 6 - PRBS
    signal_type = 1  # scegli qui

    # Parametri (modifica a piacere)
    sig_params = dict(
        value=0.0,  # constant
        amp=0.3,  # ampiezza comune a step/impulse/sine/prbs
        k0=25,  # indice inizio step/impulse
        slope=0.01,  # ramp
        period=40.0,  # sine
        phase=0.0,  # sine
        amps=[0.25, 0.15],  # multisine
        periods=[60.0, 23.0],  # multisine
        phases=[0.0, 1.2],  # multisine
        bitlen=5,  # prbs
    )


    def make_u(T, n_u, kind: int) -> np.ndarray:
        p = sig_params
        if kind == 0:  # constant
            return constant(T, n_u, value=p["value"])
        if kind == 1:  # step
            return step(T, n_u, amp=p["amp"], k0=p["k0"])
        if kind == 2:  # ramp
            return ramp(T, n_u, slope=p["slope"])
        if kind == 3:  # impulse
            return impulse(T, n_u, amp=p["amp"], k0=p["k0"])
        if kind == 4:  # sine
            return sine(T, n_u, amp=p["amp"], period=p["period"], phase=p["phase"])
        if kind == 5:  # multisine
            return multi_sine(T, n_u, amps=p["amps"], periods=p["periods"], phases=p["phases"])
        if kind == 6:  # prbs
            return prbs(T, n_u, amp=p["amp"], bitlen=p["bitlen"])
        # fallback: mix "classico" (sine + prbs)
        s = sine(T, n_u, amp=p["amp"], period=p["period"], phase=p["phase"])
        q = prbs(T, n_u, amp=0.1, bitlen=max(3, p["bitlen"] // 2))
        return s + q


    u_seq = make_u(T, n_u, signal_type)
    u_seq = np.clip(u_seq, U_bounds[0], U_bounds[1])

    x0 = np.array([0.2, -0.3])

    X_rnn = simulate_rnn(rnn, x0, u_seq)
    X_pwa, region_seq = simulate_pwa_from_patterns(rnn, dyn_map, x0, u_seq)

    # --- Grafici principali: traiettorie & errore ---
    time = np.arange(T+1)
    plot_input_signal(time, u_seq)
    plot_trajectories(time, X_rnn, X_pwa, title="RNN vs PWA (stessi ingressi)")
    plot_error(time, X_rnn, X_pwa)

    # --- Piano delle regioni (+ traiettorie sovrapposte) ---
    # Sezione a u fisso: per n_u=1 usiamo u medio come "slice" della partizione
    U_fixed = np.array([float(np.median(u_seq))])
    plot_partition_2d(
        rnn,
        X_bounds=X_bounds,
        U_fixed=U_fixed,
        axes=(0, 1),
        grid=260,
        title="Partizione ReLU (slice u = mediana)",
    )

    # --- Plot grafo ---
    G_struct, facet_witness = build_structural_adjacency(rnn, patterns, X_bounds, U_bounds, eps_pos=1e-6)
    plot_graph_networkx(G_struct)