import numpy as np
from rnn2pwa.models.rnn_relu import RNN, Layer
from rnn2pwa.regions import build_region_adjacency_graph
from rnn2pwa.utils.discovery import discover_regions_via_lp, build_local_dynamics_map
from rnn2pwa.utils.signals import constant, step, ramp, impulse, sine, multi_sine, prbs
from rnn2pwa.simulate.rollout import simulate_rnn, simulate_pwa_from_patterns
from rnn2pwa.visualize.trajectories import plot_trajectories, plot_error
from rnn2pwa.visualize.style import set_paper_style
from rnn2pwa.visualize.analysis_plots import (
    plot_input_signal,
    plot_feasible_regions_xu
)

if __name__ == "__main__":
    np.random.seed(0)
    set_paper_style()

    # ===== Network: 2 hidden layers × 3 neurons each, with x∈R^1, u∈R^1
    n_x, n_u = 1, 1

    # Layer 1: (3 neurons) takes [x;u] ∈ R^{1+1} → R^3
    W1 = np.array([
        [0.85, 0.60],   # neuron 1
        [-0.50, 0.75],  # neuron 2
        [0.40, -0.55],  # neuron 3
    ], dtype=float)
    b1 = np.array([0.10, -0.15, 0.05], dtype=float)

    # Layer 2: (3 neurons) takes h1∈R^3 → x_{k+1}∈R^1
    W2 = np.array([
        [0.70, -0.30, 0.45],  # output neuron
    ], dtype=float)
    b2 = np.array([0.02], dtype=float)

    rnn = RNN(layers=[Layer(W1, b1), Layer(W2, b2)], n_x=n_x, n_u=n_u)

    # --- Domains: x ∈ R^1, u ∈ R^1 ---
    X_bounds = (np.array([-1.0]), np.array([1.0]))  # 1D state bounds
    U_bounds = (np.array([-1.0]), np.array([1.0]))  # 1D input bounds

    # --- Discovery regions and building local dynamics ---
    print("Solving Feasibility Problem")
    patterns, witnesses = discover_regions_via_lp(rnn, X_bounds, U_bounds)
    print(f"Found {len(patterns)} unique regions")

    print("Building local dynamics map...")
    dyn_map = build_local_dynamics_map(rnn, patterns)

    nodes, edges, idmap = build_region_adjacency_graph(rnn, patterns, X_bounds, U_bounds)
    print(f"#nodes = {len(nodes)}, #edges = {len(edges)}")

    # --- Input signal and simulations ---
    T = 200

    # Input signal type
    # 0 - Constant | 1 - Step | 2 - Ramp | 3 - Impulse
    # 4 - Sine     | 5 - Multisine | 6 - PRBS
    signal_type = 5  # multisine for rich dynamics

    # Parameters
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

    x0 = np.array([0.2], dtype=float)

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

    # --- 4. Partition showing ONLY feasible regions in (x,u) space ---
    print("Plotting feasible regions in (x,u) space...")
    plot_feasible_regions_xu(
        rnn,
        patterns,
        witnesses,
        X_bounds,
        U_bounds,
        grid=1000,
        x_axis=0,  # Only one state dimension
        title="Feasible ReLU Regions in (x,u) space"
    )

    print("\n=== All plots generated successfully! ===")