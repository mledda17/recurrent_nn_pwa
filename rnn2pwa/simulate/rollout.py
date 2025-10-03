# rnn2pwa/simulate/rollout.py
import numpy as np
from typing import Dict, Tuple, List
from rnn2pwa.models.rnn_relu import RNN, forward_step, pattern_from_point

def _normalize_U(U: np.ndarray, n_u: int) -> np.ndarray:
    U = np.asarray(U)
    # (T,)  -> (T,1)
    if U.ndim == 1:
        if n_u != 1:
            raise ValueError(f"U.ndim==1 ma n_u={n_u}. Atteso (T,{n_u}).")
        U = U.reshape(-1, 1)
    # (n_u, T) -> (T, n_u)
    if U.shape[0] == n_u and (U.ndim == 2 and U.shape[1] != n_u):
        U = U.T
    # ora devo avere (T, n_u)
    if U.ndim != 2 or U.shape[1] != n_u:
        raise ValueError(f"U deve avere shape (T,{n_u}), ottenuto {U.shape}.")
    return U

def _normalize_x0(x0: np.ndarray, n_x: int) -> np.ndarray:
    x0 = np.asarray(x0).reshape(-1)
    if x0.size != n_x:
        raise ValueError(f"x0 deve avere dimensione ({n_x},), ottenuto {x0.shape}.")
    return x0

def simulate_rnn(rnn: RNN, x0: np.ndarray, U: np.ndarray) -> np.ndarray:
    U = _normalize_U(U, rnn.n_u)
    x = _normalize_x0(x0, rnn.n_x)
    X = [x.copy()]
    for k in range(U.shape[0]):
        u_k = U[k].reshape(rnn.n_u,)        # (n_u,)
        x = forward_step(rnn, x, u_k)
        X.append(x.copy())
    return np.vstack(X)

def simulate_pwa_from_patterns(rnn: RNN, dyn_map: Dict, x0: np.ndarray, U: np.ndarray) -> Tuple[np.ndarray, List]:
    U = _normalize_U(U, rnn.n_u)
    x = _normalize_x0(x0, rnn.n_x)
    X = [x.copy()]
    regions = []
    for k in range(U.shape[0]):
        u_k = U[k].reshape(rnn.n_u,)
        pat = pattern_from_point(rnn, x, u_k)
        regions.append(pat)
        A, B, c = dyn_map[pat]
        x = A @ x + B @ u_k + c
        X.append(x.copy())
    return np.vstack(X), regions

def build_transition_graph(rnn: RNN, X: np.ndarray, U: np.ndarray):
    U = _normalize_U(U, rnn.n_u)
    G = {}
    for k in range(U.shape[0]):
        u_k = U[k].reshape(rnn.n_u,)
        p0 = pattern_from_point(rnn, X[k],   u_k)
        p1 = pattern_from_point(rnn, X[k+1], u_k)
        G.setdefault(p0, set()).add(p1)
    return G
