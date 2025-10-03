import numpy as np
from typing import Tuple
from rnn2pwa.models.rnn_relu import RNN

def local_affine_relu(rnn: RNN, sigma: Tuple[Tuple[int,...], ...]):
    tilde_W, tilde_b = [], []
    for layer, sig_l in zip(rnn.layers, sigma):
        D = np.diag(np.array(sig_l, dtype=float))
        tilde_W.append(D @ layer.W)
        tilde_b.append(D @ layer.b)

    T = tilde_W[0]
    for TW in tilde_W[1:]:
        T = TW @ T

    n_x = rnn.n_x
    c = np.zeros(n_x)
    L = len(rnn.layers)
    for ell in range(L):
        post = np.eye(n_x) if ell == L-1 else tilde_W[ell+1]
        for j in range(ell+2, L):
            post = tilde_W[j] @ post
        c += post @ tilde_b[ell]

    A = T[:, :rnn.n_x]
    B = T[:, rnn.n_x:]
    return A, B, c
