from __future__ import annotations
from typing import Set, Tuple
import numpy as np
from torch import nn
from ..relu_rnn.eval import pattern_of

def enumerate_patterns_by_sampling(m: nn.Module,
                                   X_bounds: Tuple[np.ndarray, np.ndarray],
                                   U_bounds: Tuple[np.ndarray, np.ndarray],
                                   grid: Tuple[int, int] | None = None,
                                   rng: np.random.Generator | None = None,
                                   n_samples: int = 0) -> Set[Tuple[int, ...]]:
    """
    Discover candidate patterns by grid and/or random sampling over X×U.
    """
    n = X_bounds[0].size; m_in = U_bounds[0].size
    patterns: Set[Tuple[int, ...]] = set()
    if grid is not None:
        gx, gu = grid
        Xs = [np.linspace(X_bounds[0][i], X_bounds[1][i], gx) for i in range(n)]
        Us = [np.linspace(U_bounds[0][j], U_bounds[1][j], gu) for j in range(m_in)]
        for x in np.array(np.meshgrid(*Xs)).T.reshape(-1, n):
            for u in np.array(np.meshgrid(*Us)).T.reshape(-1, m_in):
                σ = pattern_of(m, x[None, :], u[None, :])
                patterns.add(σ)
    if n_samples > 0:
        if rng is None:
            rng = np.random.default_rng(0)
        for _ in range(n_samples):
            x = rng.uniform(X_bounds[0], X_bounds[1])
            u = rng.uniform(U_bounds[0], U_bounds[1])
            σ = pattern_of(m, x[None, :], u[None, :])
            patterns.add(σ)
    return patterns
