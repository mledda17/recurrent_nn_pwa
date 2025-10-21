from rnn2pwa.models.rnn_relu import RNN
from rnn2pwa.regions.lp_feasibility import pattern_feasible_lp_relu
from rnn2pwa.regions.local_dynamics import local_affine_relu
import numpy as np
from collections import deque

def discover_regions_via_lp(rnn, X_bounds, U_bounds, eps=1e-9):
    """
    Enumerate all feasible ReLU activation regions of the RNN
    within the given state and input bounds.
    """
    # consider only hidden ReLU layers (exclude final linear)
    sizes = [L.W.shape[0] for L in rnn.layers[:-1]]

    # all-zero pattern as starting point
    all_zero = tuple(tuple(0 for _ in range(n)) for n in sizes)

    patterns, witnesses = [], {}
    visited = set()
    Q = deque([all_zero])

    while Q:
        pat = Q.popleft()
        if pat in visited:
            continue
        visited.add(pat)

        # try to flip each neuron to discover neighbors
        for l, n_l in enumerate(sizes):
            for i in range(n_l):
                q = [list(row) for row in pat]
                q[l][i] = 1 - q[l][i]
                q = tuple(tuple(row) for row in q)
                if q not in visited:
                    Q.append(q)

        # check LP feasibility
        feas, xw, uw = pattern_feasible_lp_relu(rnn, pat, X_bounds, U_bounds)
        if not feas:
            continue

        patterns.append(pat)
        witnesses[pat] = (xw, uw)

    return patterns, witnesses


def print_region_border_info(rnn: RNN, patterns, X_bounds, U_bounds):
    print("\n=== REGION BORDER ANALYSIS ===")
    for pat_idx, sigma in enumerate(patterns, start=1):
        feas, xw, uw = pattern_feasible_lp_relu(rnn, sigma, X_bounds, U_bounds)
        if not feas:
            print(f"\nRegion {pat_idx}: pattern {sigma} --> INFEASIBLE")
            continue

        print(f"\nRegion {pat_idx}: pattern {sigma}")
        for l, sig_l in enumerate(sigma, start=1):
            sig = np.array(sig_l)
            act = np.where(sig == 1)[0]
            inact = np.where(sig == 0)[0]
            for i in act:
                print(f"  Layer {l}, neuron {i}: ACTIVE -> boundary included (a_{l},{i}=0)")
            for i in inact:
                print(f"  Layer {l}, neuron {i}: INACTIVE -> boundary excluded (a_{l},{i}=0)")

        if xw is not None and uw is not None:
            print(f"    → Witness: x={xw}, u={uw}")


def build_local_dynamics_map(rnn: RNN, patterns):
    return {pat: local_affine_relu(rnn, pat) for pat in patterns}
