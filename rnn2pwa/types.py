from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional
import numpy as np

Pattern = Tuple[int, ...]  # concatenated 0/1 for all hidden ReLUs

@dataclass
class Region:
    pattern: Pattern
    H: np.ndarray  # (k, d) with d = n + m
    h: np.ndarray  # (k,)
    witness: Optional[np.ndarray] = None  # z* = [x;u] proving feasibility

@dataclass
class LocalDynamics:
    pattern: Pattern
    A_x: np.ndarray  # (n, n)
    B_u: np.ndarray  # (n, m)
    c: np.ndarray    # (n,)

@dataclass
class Guard:
    src: Pattern
    dst: Pattern
    # Optional hyperplane representation n^T z = b with additional inequalities
    normal: Optional[np.ndarray] = None
    offset: Optional[float] = None

@dataclass
class PWA:
    regions: Dict[Pattern, Region] = field(default_factory=dict)
    dynamics: Dict[Pattern, LocalDynamics] = field(default_factory=dict)
    guards: List[Guard] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)
