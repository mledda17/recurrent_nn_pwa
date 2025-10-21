import numpy as np
from typing import Tuple
from rnn2pwa.models.rnn_relu import RNN

def local_affine_relu(rnn: RNN, sigma):
    # sigma contiene solo hidden layers
    tilde_W, tilde_b = [], []
    hidden_layers = rnn.layers[:-1]
    out_layer = rnn.layers[-1]

    for layer, sig_l in zip(hidden_layers, sigma):
        D = np.diag(np.array(sig_l, dtype=float))
        tilde_W.append(D @ layer.W)
        tilde_b.append(D @ layer.b)

    # composizione attraverso i soli hidden
    T = tilde_W[0]
    for TW in tilde_W[1:]:
        T = TW @ T

    # ultimo layer lineare
    W_L, b_L = out_layer.W, out_layer.b
    T_total = W_L @ T
    c = b_L.copy()
    # bias accumulato dai hidden
    post = W_L
    for ell in range(len(tilde_W)):
        c += post @ tilde_b[ell]
        if ell+1 < len(tilde_W):
            post = post @ tilde_W[ell+1]

    n_x = rnn.n_x
    A = T_total[:, :n_x]
    B = T_total[:, n_x:]
    return A, B, c

