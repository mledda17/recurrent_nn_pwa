import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Layer:
    W: np.ndarray
    b: np.ndarray

@dataclass
class RNN:
    layers: List[Layer]
    n_x: int
    n_u: int

def relu(v: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, v)

def forward_step(rnn: RNN, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    x = np.asarray(x).reshape(-1)
    u = np.asarray(u).reshape(-1)
    if x.size != rnn.n_x:
        raise ValueError(f"[forward_step] x ha dim {x.size}, atteso {rnn.n_x}")
    if u.size != rnn.n_u:
        raise ValueError(f"[forward_step] u ha dim {u.size}, atteso {rnn.n_u}")
    h = np.concatenate([x, u])
    for layer in rnn.layers:
        a = layer.W @ h + layer.b
        h = relu(a)
    return h

def pattern_from_point(rnn: RNN, x: np.ndarray, u: np.ndarray) -> Tuple[Tuple[int,...], ...]:
    x = np.asarray(x).reshape(-1)
    u = np.asarray(u).reshape(-1)
    if x.size != rnn.n_x or u.size != rnn.n_u:
        raise ValueError(f"[pattern_from_point] shape err: x {x.shape}, u {u.shape}")
    h = np.concatenate([x, u])
    sigmas = []
    for layer in rnn.layers:
        a = layer.W @ h + layer.b
        sigmas.append(tuple((a > 0.0).astype(int).tolist()))
        h = relu(a)
    return tuple(sigmas)
