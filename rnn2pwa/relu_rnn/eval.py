from __future__ import annotations
from typing import Tuple, List
import torch
from torch import nn
import numpy as np

@torch.no_grad()
def forward_pre_activations(m: nn.Module, x: np.ndarray, u: np.ndarray) -> List[np.ndarray]:
    """
    Return pre-activations z^(ell) for each hidden layer (numpy arrays),
    assuming a Sequential-like structure: [Linear, ReLU, Linear, ReLU, ..., Linear].
    """
    m.eval()
    device = next(m.parameters()).device if any(p.requires_grad for p in m.parameters()) else "cpu"
    z0 = np.concatenate([x, u], axis=-1)
    t = torch.as_tensor(z0, dtype=torch.float32, device=device)
    zs: List[np.ndarray] = []
    out = t
    layers = list(m.children())
    i = 0
    while i < len(layers) - 1:
        lin = layers[i]; act = layers[i + 1]
        assert isinstance(lin, nn.Linear), "Expected Linear layer"
        assert isinstance(act, nn.ReLU), "Expected ReLU after each hidden Linear"
        z = lin(out)
        zs.append(z.detach().cpu().numpy())
        out = act(z)
        i += 2
    # final linear (no ReLU)
    assert isinstance(layers[-1], nn.Linear), "Last layer must be Linear"
    return zs

# AFTER (correct: collect every hidden layer; use the first sample only)
def pattern_of(m: nn.Module, x: np.ndarray, u: np.ndarray) -> Tuple[int, ...]:
    """
    Compute activation pattern σ across all hidden layers.
    Deterministic tie-break: z == 0 -> active (1).
    """
    zs = forward_pre_activations(m, x, u)
    bits: list[int] = []
    for z in zs:
        # use first sample in batch
        row = z[0] if z.ndim == 2 else z
        bits.extend((row >= 0.0).astype(int).tolist())
    return tuple(bits)


