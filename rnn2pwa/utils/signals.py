# rnn2pwa/utils/signals.py
import numpy as np
from typing import List, Tuple, Optional

Array = np.ndarray

def _ensure_shape(U: Array, T: int, n_u: int) -> Array:
    U = np.asarray(U)
    if U.ndim == 1:
        if n_u != 1:
            raise ValueError(f"Segnale 1D ma n_u={n_u}. Atteso (T,{n_u}).")
        U = U.reshape(-1, 1)
    if U.shape[0] != T or U.shape[1] != n_u:
        raise ValueError(f"Shape attesa (T,{n_u}), ottenuto {U.shape}.")
    return U

def clip_to_bounds(U: Array, U_bounds: Tuple[Array, Array]) -> Array:
    lo, hi = U_bounds
    return np.clip(U, lo.reshape(1, -1), hi.reshape(1, -1))

# -------- Segnali base --------
def constant(T: int, n_u: int, value: float = 0.0) -> Array:
    return np.full((T, n_u), float(value))

def step(T: int, n_u: int, amp: float = 1.0, k0: int = 0) -> Array:
    U = np.zeros((T, n_u))
    U[k0:, :] = amp
    return U

def ramp(T: int, n_u: int, slope: float = 0.01) -> Array:
    t = np.arange(T).reshape(T, 1)
    return slope * t * np.ones((1, n_u))

def impulse(T: int, n_u: int, amp: float = 1.0, k0: int = 10) -> Array:
    U = np.zeros((T, n_u))
    if 0 <= k0 < T:
        U[k0, :] = amp
    return U

def sine(T: int, n_u: int, amp: float = 1.0, period: float = 40.0, phase: float = 0.0) -> Array:
    k = np.arange(T).reshape(T, 1)
    s = amp * np.sin(2*np.pi * k/period + phase)
    return np.repeat(s, n_u, axis=1)

def multi_sine(T: int, n_u: int, amps: List[float], periods: List[float], phases: Optional[List[float]] = None) -> Array:
    if phases is None: phases = [0.0]*len(amps)
    k = np.arange(T).reshape(T, 1)
    U = np.zeros((T, n_u))
    for a, p, ph in zip(amps, periods, phases):
        U += a * np.sin(2*np.pi * k/p + ph)
    return U

def chirp_linear(T: int, n_u: int, amp: float = 1.0, f0: float = 0.01, f1: float = 0.2) -> Array:
    t = np.arange(T)
    f = f0 + (f1 - f0) * (t / max(T-1, 1))
    phase = 2*np.pi * np.cumsum(f)  # integrazione numerica
    s = amp * np.sin(phase).reshape(T, 1)
    return np.repeat(s, n_u, axis=1)

def white_noise(T: int, n_u: int, std: float = 0.1, seed: Optional[int] = None) -> Array:
    rng = np.random.default_rng(seed)
    return std * rng.standard_normal((T, n_u))

def zoh_random(T: int, n_u: int, amp: float = 0.3, hold: int = 5, seed: Optional[int] = None) -> Array:
    """Piecewise-constant casuale (Zero-Order Hold)."""
    rng = np.random.default_rng(seed)
    U = np.zeros((T, n_u))
    for k in range(0, T, hold):
        val = amp * (2*rng.random((1, n_u)) - 1.0)
        U[k:k+hold, :] = val
    return U

def prbs(T: int, n_u: int, amp: float = 0.3, bitlen: int = 5, seed: Optional[int] = None) -> Array:
    """PRBS semplice: cambia segno ogni 'bitlen' campioni."""
    rng = np.random.default_rng(seed)
    U = np.zeros((T, n_u))
    sign = 2*rng.integers(0, 2, size=(1, n_u)) - 1
    for k in range(T):
        if k % bitlen == 0:
            sign = 2*rng.integers(0, 2, size=(1, n_u)) - 1
        U[k, :] = amp * sign
    return U

# -------- Combinatori --------
def mix(signals: List[Array], weights: Optional[List[float]] = None) -> Array:
    if weights is None:
        weights = [1.0]*len(signals)
    U = np.zeros_like(signals[0])
    for S, w in zip(signals, weights):
        U = U + w * S
    return U

def concat(segments: List[Tuple[Array, int]]) -> Array:
    """segments: lista di (generatore_output, durata). Se durata < output, tronca; se >, ripete."""
    parts = []
    for U_seg, L in segments:
        if U_seg.shape[0] >= L:
            parts.append(U_seg[:L])
        else:
            reps = (L + U_seg.shape[0] - 1) // U_seg.shape[0]
            parts.append(np.vstack([U_seg]*reps)[:L])
    return np.vstack(parts)

def from_npy(path: str, n_u: int) -> Array:
    U = np.load(path)
    if U.ndim == 1: U = U.reshape(-1, 1)
    if U.shape[1] != n_u:
        raise ValueError(f"File {path}: atteso n_u={n_u}, trovato {U.shape[1]}")
    return U
