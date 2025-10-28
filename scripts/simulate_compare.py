import numpy as np
from rnn2pwa.models.rnn_relu import RNN, Layer
from rnn2pwa.io.pgf_utils import export_trajectories_dat, export_feasible_regions_xu_dat
from rnn2pwa.regions import build_region_adjacency_graph
from rnn2pwa.utils.discovery import discover_regions_via_lp, build_local_dynamics_map, print_region_border_info
from rnn2pwa.utils.signals import constant, step, ramp, impulse, sine, multi_sine, prbs
from rnn2pwa.simulate.rollout import simulate_rnn, simulate_pwa_from_patterns
from rnn2pwa.visualize.trajectories import plot_trajectories, plot_error
from rnn2pwa.visualize.style import set_paper_style
from rnn2pwa.visualize.analysis_plots import (
    plot_input_signal,
    plot_feasible_regions_xu,
    export_trajectory_xu_path_dat
)



if __name__ == "__main__":
    np.random.seed(0)
    set_paper_style()

    # ===== Network: 2 hidden layers × 3 neurons each, with x∈R^1, u∈R^1
    n_x, n_u = 1, 1

    # Layer 1: (3 neurons) takes [x;u] ∈ R^{1+1} → R^3
    W1 = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [0.8, -0.6]
    ])
    b1 = np.array([0.0, 0.0, -0.2])

    W2 = np.array([[0.70, -0.25, 0.35]])
    b2 = np.array([0.02])

    rnn = RNN(layers=[Layer(W1, b1), Layer(W2, b2)], n_x=n_x, n_u=n_u)

    # --- Domains: x ∈ R^1, u ∈ R^1 ---
    X_bounds = (np.array([-2.0]), np.array([2.0]))  # 1D state bounds
    U_bounds = (np.array([-2.0]), np.array([2.0]))  # 1D input bounds

    # --- Discovery regions and building local dynamics ---
    print("Solving Feasibility Problem")
    patterns, witnesses = discover_regions_via_lp(rnn, X_bounds, U_bounds)
    # print_region_border_info(rnn, patterns, X_bounds, U_bounds)
    print(f"Found {len(patterns)} unique regions")

    print("Building local dynamics map...")
    dyn_map = build_local_dynamics_map(rnn, patterns)

    nodes, edges, idmap = build_region_adjacency_graph(rnn, patterns, X_bounds, U_bounds)
    print(f"#nodes = {len(nodes)}, #edges = {len(edges)}")

    # --- Input signal and simulations ---
    T = 1000

    # Input signal type
    # 0 - Constant | 1 - Step | 2 - Ramp | 3 - Impulse
    # 4 - Sine     | 5 - Multisine | 6 - PRBS
    signal_type = 6  # multisine for rich dynamics

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

    x0 = np.array([1.5], dtype=float)

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
        grid=400,
        save=True,  # salva automaticamente
        outdir="plots",  # directory
        basename="xu_regions_lp"  # nome base
    )

    # --- Export feasible (x,u) region grid for PGFPlots
    grid_path, wit_path = export_feasible_regions_xu_dat(
        rnn=rnn,
        patterns=patterns,
        X_bounds=X_bounds,
        U_bounds=U_bounds,
        grid=600,  # adjust for resolution/Overleaf size
        x_axis=0,
        outdir="plots",
        basename="xu_regions"
    )

    # Also export witness points (if you want them on the plot)
    # witnesses: Dict[pattern] -> (x_w, u_w)
    wit_points = []
    for pat, (xw, uw) in witnesses.items():
        wit_points.append([xw[0], uw[0] if np.ndim(uw) == 0 else uw[0]])
    if wit_points:
        wit_points = np.array(wit_points)
        np.savetxt("plots/xu_regions_wit.dat", wit_points, fmt="%.10g", header="x\tu", comments="")

    # --- Export trajectories (time, states, input)
    export_trajectories_dat(
        t=time,  # length T+1
        X_pwa=X_pwa,  # (T+1, nx)
        X_rnn=X_rnn,  # (T+1, nx)
        U=u_seq,  # (T, nu) or (T,)
        outdir="plots",
        basename="traj"
    )

    # Se hai già 'patterns' in ordine, crea mappa pattern -> id
    pat_to_id = {pat: i for i, pat in enumerate(patterns)}  # opzionale

    # Calcola id regione lungo traiettoria (opzionale)
    # region_ids = compute_region_ids_along_traj(rnn, X_pwa, u_seq, pat_to_id)  # oppure con X_rnn, come preferisci
    region_ids = None  # se non ti serve il colore per regione

    export_trajectory_xu_path_dat(
        t=time,
        X=X_pwa,  # o X_rnn, scegli cosa proiettare su asse x
        U=u_seq,
        outdir="plots",
        basename="traj_xu_path",
        x_axis=0,
        u_axis=0,
        region_ids=region_ids,
        thin_markers=5  # marker ogni 5 campioni (togli o metti 1 se li vuoi tutti)
    )

    print("\n=== All plots generated successfully! ===")