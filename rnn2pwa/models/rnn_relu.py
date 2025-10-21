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
    x = np.asarray(x).reshape(-1); u = np.asarray(u).reshape(-1)
    h = np.concatenate([x, u])
    L = len(rnn.layers)
    # hidden layers 1..L-1: Linear + ReLU
    for layer in rnn.layers[:-1]:
        a = layer.W @ h + layer.b
        h = relu(a)
    # last layer L: Linear ONLY
    W_L, b_L = rnn.layers[-1].W, rnn.layers[-1].b
    y = W_L @ h + b_L
    return y

# --- rnn_relu.py: pattern_from_point ---
def pattern_from_point(rnn: RNN, x: np.ndarray, u: np.ndarray) -> Tuple[Tuple[int,...], ...]:
    x = np.asarray(x).reshape(-1); u = np.asarray(u).reshape(-1)
    h = np.concatenate([x, u])
    sigmas = []
    L = len(rnn.layers)
    # solo hidden layers (niente ultimo layer)
    for layer in rnn.layers[:-1]:
        a = layer.W @ h + layer.b
        sigmas.append(tuple((a >= 0.0).astype(int).tolist()))
        h = relu(a)
    return tuple(sigmas)

