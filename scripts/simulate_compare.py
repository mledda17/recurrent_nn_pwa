import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from rnn2pwa.models.rnn_relu import RNN, Layer
from rnn2pwa.regions import build_region_adjacency_graph
from rnn2pwa.utils.discovery import discover_regions_via_lp, build_local_dynamics_map
from rnn2pwa.utils.signals import constant, step, ramp, impulse, sine, multi_sine, prbs
from rnn2pwa.simulate.rollout import simulate_rnn, simulate_pwa_from_patterns
from rnn2pwa.visualize.trajectories import plot_trajectories, plot_error
from rnn2pwa.visualize.style import set_paper_style
from rnn2pwa.visualize.analysis_plots import (
    plot_partition_2d,
    plot_input_signal, plot_partition_xu
)

if __name__ == "__main__":
    np.random.seed(0)
    set_paper_style()

    # --- Rete esempio ---
    # Layer 1: 2 neuroni | input=[x;u]∈R^2 ⇒ W1∈R^{2×2}, b1∈R^2
    n_x, n_u = 1, 1

    W1 = np.array([
        [1.0, 0.6],
        [-0.8, 0.9],
    ])
    b1 = np.array([0.05, -0.10])

    # Layer 2: 1 neurone (uscita = x_{k+1}) | input=h1∈R^2 ⇒ W2∈R^{1×2}, b2∈R^1
    W2 = np.array([
        [0.7, -0.4]
    ])
    b2 = np.array([0.0])

    rnn = RNN(layers=[Layer(W1, b1), Layer(W2, b2)], n_x=n_x, n_u=n_u)

    # --- Domini (scalari) ---
    X_bounds = (np.array([-1.5]), np.array([1.5]))
    U_bounds = (np.array([-0.8]), np.array([0.8]))

    # --- Discovery regions and building local dynamics ---
    print("Solving Feasibility Problem")
    patterns, witnesses = discover_regions_via_lp(rnn, X_bounds, U_bounds)
    print(f"Found {len(patterns)} unique regions")

    print("Building local dynamics map...")
    dyn_map = build_local_dynamics_map(rnn, patterns)

    nodes, edges, idmap = build_region_adjacency_graph(rnn, patterns, X_bounds, U_bounds)
    print(f"#nodi = {len(nodes)}, #archi = {len(edges)}")

    # --- Ingresso e simulazioni ---
    T = 200

    # Input signal type
    # 0 - Constant | 1 - Step | 2 - Ramp | 3 - Impulse
    # 4 - Sine     | 5 - Multisine | 6 - PRBS
    signal_type = 3  # multisine for rich dynamics

    # Parametri
    sig_params = dict(
        value=0.0,
        amp=0.3,
        k0=25,
        slope=0.01,
        period=40.0,
        phase=0.0,
        amps=[0.25, 0.15],
        periods=[60.0, 23.0],
        phases=[0.0, 1.2],
        bitlen=5,
    )

    def make_u(T, n_u, kind: int) -> np.ndarray:
        p = sig_params
        if kind == 0:
            return constant(T, n_u, value=p["value"])
        if kind == 1:
            return step(T, n_u, amp=p["amp"], k0=p["k0"])
        if kind == 2:
            return ramp(T, n_u, slope=p["slope"])
        if kind == 3:
            return impulse(T, n_u, amp=p["amp"], k0=p["k0"])
        if kind == 4:
            return sine(T, n_u, amp=p["amp"], period=p["period"], phase=p["phase"])
        if kind == 5:
            return multi_sine(T, n_u, amps=p["amps"], periods=p["periods"], phases=p["phases"])
        if kind == 6:
            return prbs(T, n_u, amp=p["amp"], bitlen=p["bitlen"])
        # fallback
        s = sine(T, n_u, amp=p["amp"], period=p["period"], phase=p["phase"])
        q = prbs(T, n_u, amp=0.1, bitlen=max(3, p["bitlen"] // 2))
        return s + q


    u_seq = make_u(T, n_u, signal_type)
    u_seq = np.clip(u_seq, U_bounds[0], U_bounds[1])

    x0 = np.array([0.2])

    print("Simulating RNN and PWA...")
    X_rnn = simulate_rnn(rnn, x0, u_seq)
    X_pwa, region_seq = simulate_pwa_from_patterns(rnn, dyn_map, x0, u_seq)

    # ========================================
    # VISUALIZATIONS
    # ========================================

    # --- 1. Input signal ---
    print("\nPlotting input signal...")
    time = np.arange(T + 1)
    plot_input_signal(time, u_seq)

    # --- 2. Trajectories comparison ---
    print("Plotting trajectory comparison...")
    plot_trajectories(time, X_rnn, X_pwa, title="RNN vs PWA (same input)")

    # --- 3. Tracking error ---
    print("Plotting tracking error...")
    plot_error(time, X_rnn, X_pwa)

    # --- 4. Partition (u fixed at median) ---
    print("Plotting state space partition...")
    plot_partition_xu(rnn, X_bounds, U_bounds, grid=500,
                      title="ReLU partition nel piano (x,u)")

    G = nx.Graph()
    G.add_nodes_from(range(len(nodes)))
    G.add_edges_from(edges)
    plt.figure(figsize=(6, 5))
    nx.draw(G, with_labels=True, node_size=400, font_size=8)
    plt.title("Grafo delle regioni (nodi=pattern, archi=regioni confinanti)")
    plt.show()

    print("\n=== All plots generated successfully! ===")