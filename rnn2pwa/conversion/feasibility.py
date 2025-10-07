from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
from scipy.optimize import linprog

def find_witness(H: np.ndarray, h: np.ndarray, d: int,
                 bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> Tuple[bool, Optional[np.ndarray]]:
    """
    Solve feasibility LP: find z in R^d s.t. H z <= h.
    We minimize 0^T z (pure feasibility). Optionally use bounds = (lo, hi) on z.
    """
    c = np.zeros(d, dtype=float)
    A_ub = H
    b_ub = h
    if bounds is None:
        bnds = [(None, None)] * d
    else:
        lo, hi = bounds
        bnds = [(float(lo[i]), float(hi[i])) for i in range(d)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bnds, method="highs")
    if res.success:
        return True, res.x
    return False, None
