from __future__ import annotations
from typing import Tuple, List
import numpy as np
import torch
from torch import nn

def _sigma_layers(sigma: Tuple[int, ...], widths: List[int]) -> List[np.ndarray]:
    splits = np.cumsum(widths[:-1])
    parts = np.split(np.array(sigma, dtype=int), splits) if splits.size else [np.array(sigma, int)]
    return [np.diag(p) for p in parts]

def _hidden_widths(m: nn.Module) -> List[int]:
    layers = list(m.children())
    widths = []
    for i in range(0, len(layers) - 1, 2):
        lin = layers[i]
        assert isinstance(lin, nn.Linear)
        widths.append(lin.out_features)
    return widths

def local_affine_map(m: nn.Module, sigma: Tuple[int, ...], n: int, m_in: int):
    """
    Compute (A_x, B_u, c) such that x_next = A_x x + B_u u + c for fixed activation pattern σ.
    Recursion over hidden layers:
       h = D (W h_prev + b), with h_0 = [x;u]
    Final: x_next = W_L h_{L-1} + b_L
    """
    layers = list(m.children())
    assert isinstance(layers[-1], nn.Linear), "Last layer must be Linear"
    widths = _hidden_widths(m)
    Dmats = _sigma_layers(sigma, widths)

    d_in = n + m_in
    # Start with identity to map z0 -> z0
    A = np.eye(d_in)
    b = np.zeros((d_in,))

    # propagate through hidden layers
    for ell, D in enumerate(Dmats):
        lin = layers[2*ell]
        W = lin.weight.detach().cpu().numpy()
        bb = lin.bias.detach().cpu().numpy()
        # h = D (W @ prev + b)
        A = D @ W @ A
        b = D @ (W @ b + bb)

    # final linear map
    W_L = layers[-1].weight.detach().cpu().numpy()  # (n, width_{L-1})
    b_L = layers[-1].bias.detach().cpu().numpy()    # (n,)
    A_total = W_L @ A
    c = W_L @ b + b_L

    # split columns into x and u contributions
    A_x = A_total[:, :n]
    B_u = A_total[:, n:]
    return A_x, B_u, c
