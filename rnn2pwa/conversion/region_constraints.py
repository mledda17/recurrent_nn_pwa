from __future__ import annotations
from typing import Tuple, List
import numpy as np
import torch
from torch import nn

def _layer_dims(m: nn.Module) -> List[int]:
    """Return widths of hidden layers (ReLU after each hidden Linear)."""
    widths = []
    layers = list(m.children())
    for i in range(0, len(layers) - 1, 2):
        lin = layers[i]
        assert isinstance(lin, nn.Linear)
        widths.append(lin.out_features)
    return widths

def _affine_pre_activations(m: nn.Module, sigma: Tuple[int, ...], n: int, m_in: int):
    """
    Build affine forms for z^(ell)(x,u) = A_z^(ell) @ [x;u] + b_z^(ell) under fixed σ
    using recursion: z_1 = W1 z0 + b1; z_ell = W_ell D_{ell-1} z_{ell-1} + b_ell.
    Returns lists (A_z_list, b_z_list).
    """
    layers = list(m.children())
    assert isinstance(layers[-1], nn.Linear), "Last layer must be Linear"
    # Build diagonal masks per hidden layer from sigma
    widths = _layer_dims(m)
    splits = np.cumsum(widths[:-1])
    sigma_layers = np.split(np.array(sigma, dtype=int), splits) if splits.size else [np.array(sigma, int)]
    Dmats = [np.diag(s) for s in sigma_layers]  # ReLU on z >= 0 treated active

    d = n + m_in
    A_prev = np.eye(d)
    b_prev = np.zeros((d,))

    A_z_list: List[np.ndarray] = []
    b_z_list: List[np.ndarray] = []

    li = 0
    for ell in range(len(Dmats)):
        lin = layers[2*ell]     # Linear
        W = lin.weight.detach().cpu().numpy()
        b = lin.bias.detach().cpu().numpy()
        D = Dmats[ell]
        # z_ell = W @ (h_{ell-1}) + b, with h_{ell-1} = D_{ell-1} z_{ell-1}
        # For ell=1, h_0 = z0 so we take D as identity conceptually; but using recursion:
        if ell == 0:
            A_z = W @ A_prev
            b_z = W @ b_prev + b
        else:
            # in recursion we already stored z_{ell-1} affine in z0:
            A_prev_z = A_z_list[-1]
            b_prev_z = b_z_list[-1]
            A_z = W @ (Dmats[ell - 1] @ A_prev_z)
            b_z = W @ (Dmats[ell - 1] @ b_prev_z) + b
        A_z_list.append(A_z)
        b_z_list.append(b_z)
        li += 2  # skip ReLU
    return A_z_list, b_z_list

def build_region_constraints(m: nn.Module, sigma: Tuple[int, ...], n: int, m_in: int,
                             X_bounds: Tuple[np.ndarray, np.ndarray] | None = None,
                             U_bounds: Tuple[np.ndarray, np.ndarray] | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return Hσ, hσ such that Hσ [x;u] <= hσ encodes activation inequalities for σ,
    plus optional box constraints for X and U.
    For each hidden unit r in layer ell:
      if σ_r^(ell) = 1  -> - z_r^(ell) <= 0
      if σ_r^(ell) = 0  ->   z_r^(ell) <= 0
    """
    layers = list(m.children())
    # widths of hidden layers
    widths = _layer_dims(m)
    # split sigma per layer
    if len(sigma) != sum(widths):
        raise ValueError(f"sigma length {len(sigma)} != total hidden units {sum(widths)}")
    splits = np.cumsum(widths[:-1])
    sigma_layers = np.split(np.array(sigma, dtype=int), splits) if splits.size else [np.array(sigma, int)]

    # Build affine pre-activations for each hidden layer
    A_z_list, b_z_list = _affine_pre_activations(m, sigma, n, m_in)

    Hs, hs = [], []
    # loop per layer and per neuron with *local* indexing
    for (A_z, b_z), sig_layer in zip(A_z_list, sigma_layers):
        assert A_z.shape[0] == sig_layer.size, "Layer width mismatch with sigma chunk."
        for i in range(A_z.shape[0]):
            alpha = A_z[i, :]  # (d,)
            beta = b_z[i]      # ()
            active = int(sig_layer[i])
            if active == 1:
                # - (alpha^T z + beta) <= 0
                Hs.append(-alpha)
                hs.append(beta)
            else:
                #   (alpha^T z + beta) <= 0
                Hs.append(alpha)
                hs.append(-beta)

    H = np.vstack(Hs) if Hs else np.zeros((0, n + m_in))
    h = np.asarray(hs, dtype=float)

    # Add domain boxes (optional)
    def box_to_ineq(lo: np.ndarray, hi: np.ndarray, offset: int) -> Tuple[np.ndarray, np.ndarray]:
        I = np.eye(n + m_in)
        rows, rhs = [], []
        for k in range(lo.size):
            e = I[offset + k]
            rows.append(e);   rhs.append(hi[k])
            rows.append(-e);  rhs.append(-lo[k])
        return np.vstack(rows), np.asarray(rhs, float)

    if X_bounds is not None:
        HX, hX = box_to_ineq(X_bounds[0], X_bounds[1], 0)
        H = np.vstack([H, HX]) if H.size else HX
        h = np.concatenate([h, hX]) if h.size else hX
    if U_bounds is not None:
        HU, hU = box_to_ineq(U_bounds[0], U_bounds[1], n)
        H = np.vstack([H, HU]) if H.size else HU
        h = np.concatenate([h, hU]) if h.size else hU

    return H, h

