from __future__ import annotations
from typing import Dict, Tuple, Set
import numpy as np
from torch import nn
from ..types import PWA, Region, LocalDynamics
from .activation_patterns import enumerate_patterns_by_sampling
from .region_constraints import build_region_constraints
from .feasibility import find_witness
from .local_dynamics import local_affine_map

def convert(m: nn.Module,
            X_bounds: Tuple[np.ndarray, np.ndarray],
            U_bounds: Tuple[np.ndarray, np.ndarray],
            grid: Tuple[int, int] = (21, 21),
            random_samples: int = 0) -> PWA:
    """
    Orchestrates RNN -> PWA conversion on a bounded domain X×U:
      1) discover candidate patterns via sampling,
      2) build inequalities Hσ z <= hσ,
      3) LP feasibility to prune empty regions (and get a witness),
      4) compute local affine dynamics (A_x, B_u, c).
    """
    n = X_bounds[0].size
    m_in = U_bounds[0].size
    d = n + m_in
    pwa = PWA(meta={"n": int(n), "m": int(m_in)})

    # 1) patterns
    patterns: Set[tuple[int, ...]] = enumerate_patterns_by_sampling(
        m, X_bounds, U_bounds, grid=grid, n_samples=random_samples
    )

    # 2-3) constraints + feasibility
    for σ in patterns:
        Hσ, hσ = build_region_constraints(m, σ, n, m_in, X_bounds, U_bounds)
        lo = np.concatenate([X_bounds[0], U_bounds[0]])
        hi = np.concatenate([X_bounds[1], U_bounds[1]])
        ok, z_w = find_witness(Hσ, hσ, d, bounds=(lo, hi))
        if not ok:
            continue
        region = Region(pattern=σ, H=Hσ, h=hσ, witness=z_w)
        pwa.regions[σ] = region

        # 4) local affine maps
        A_x, B_u, c = local_affine_map(m, σ, n, m_in)
        pwa.dynamics[σ] = LocalDynamics(pattern=σ, A_x=A_x, B_u=B_u, c=c)

    # Guards (adjacency) can be added later by intersecting closures of polytopes.
    return pwa
