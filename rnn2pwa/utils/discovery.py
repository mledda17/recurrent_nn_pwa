import numpy as np
from typing import List, Tuple
from rnn2pwa.models.rnn_relu import RNN, pattern_from_point

def discover_regions(rnn: RNN, X_bounds, U_bounds, N=5000) -> List[Tuple[Tuple[int,...], ...]]:
    X_lo, X_hi = X_bounds; U_lo, U_hi = U_bounds
    patt = set()
    for _ in range(N):
        x = X_lo + (X_hi - X_lo)*np.random.rand(rnn.n_x)
        u = U_lo + (U_hi - U_lo)*np.random.rand(rnn.n_u)
        patt.add(pattern_from_point(rnn, x, u))
    return list(patt)

def build_local_dynamics_map(rnn: RNN, patterns):
    from rnn2pwa.regions.local_dynamics import local_affine_relu
    return {pat: local_affine_relu(rnn, pat) for pat in patterns}
