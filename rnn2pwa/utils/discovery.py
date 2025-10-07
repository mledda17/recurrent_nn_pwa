from rnn2pwa.models.rnn_relu import RNN
from rnn2pwa.regions.lp_feasibility import pattern_feasible_lp_relu
from rnn2pwa.regions.local_dynamics import local_affine_relu
from collections import deque

# discovery.py
def discover_regions_via_lp(rnn, X_bounds, U_bounds, eps=1e-9):
    sizes = [L.W.shape[0] for L in rnn.layers]
    all_zero = tuple(tuple(0 for _ in range(n)) for n in sizes)

    patterns, witnesses = [], {}
    visited = set()
    from collections import deque
    Q = deque([all_zero])

    while Q:
        pat = Q.popleft()
        if pat in visited:
            continue
        visited.add(pat)  # segna subito, così non riesamini lo stesso pat

        # *** ENQUEUE SEMPRE I VICINI ***
        for l, n_l in enumerate(sizes):
            for i in range(n_l):
                q = list(list(row) for row in pat)
                q[l][i] = 1 - q[l][i]
                q = tuple(tuple(row) for row in q)
                if q not in visited:
                    Q.append(q)

        # Check feasibility via LP
        feas, xw, uw = pattern_feasible_lp_relu(rnn, pat, X_bounds, U_bounds, eps)
        if not feas:
            continue
        patterns.append(pat)
        witnesses[pat] = (xw, uw)

    return patterns, witnesses



def build_local_dynamics_map(rnn: RNN, patterns):
    return {pat: local_affine_relu(rnn, pat) for pat in patterns}
