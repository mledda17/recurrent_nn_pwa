import numpy as np
from typing import Tuple, Optional
from scipy.optimize import linprog
from rnn2pwa.models.rnn_relu import RNN

def pattern_feasible_lp_relu(rnn: RNN, sigma, X_bounds, U_bounds, eps=1e-6):
    layers = rnn.layers
    n_x, n_u = rnn.n_x, rnn.n_u
    sizes = [L.W.shape[0] for L in layers]

    idx, cur = {}, 0
    idx["x"] = slice(cur, cur+n_x); cur += n_x
    idx["u"] = slice(cur, cur+n_u); cur += n_u
    for l, n_l in enumerate(sizes, start=1):
        idx[f"a{l}"] = slice(cur, cur+n_l); cur += n_l
        idx[f"h{l}"] = slice(cur, cur+n_l); cur += n_l
    nvar = cur

    Aeq_rows, beq = [], []
    for l, (layer, sig_l) in enumerate(zip(layers, sigma), start=1):
        W, b = layer.W, layer.b
        n_l = W.shape[0]
        row = np.zeros((n_l, nvar)); row[:, idx[f"a{l}"]] = np.eye(n_l)
        if l == 1:
            row[:, idx["x"]] -= W[:, :n_x]
            row[:, idx["u"]] -= W[:, n_x:]
        else:
            row[:, idx[f"h{l-1}"]] -= W
        Aeq_rows.append(row); beq.append(b.copy())

        sig = np.array(sig_l)
        act = np.where(sig==1)[0]; inact = np.where(sig==0)[0]
        if act.size:
            R = np.zeros((act.size, nvar))
            for r, i in enumerate(act):
                R[r, idx[f"h{l}"].start+i] = 1.0
                R[r, idx[f"a{l}"].start+i] = -1.0
            Aeq_rows.append(R); beq.append(np.zeros(act.size))
        if inact.size:
            R = np.zeros((inact.size, nvar))
            for r, i in enumerate(inact):
                R[r, idx[f"h{l}"].start+i] = 1.0
            Aeq_rows.append(R); beq.append(np.zeros(inact.size))

    Aeq = np.vstack(Aeq_rows) if Aeq_rows else None
    beq = np.concatenate(beq) if beq else None

    Aub_rows, bub = [], []
    for l, sig_l in enumerate(sigma, start=1):
        sig = np.array(sig_l)
        act = np.where(sig==1)[0]; inact = np.where(sig==0)[0]
        if act.size:
            R = np.zeros((act.size, nvar))
            for r, i in enumerate(act):
                R[r, idx[f"a{l}"].start+i] = -1.0
            Aub_rows.append(R); bub.append(-eps*np.ones(act.size))
        if inact.size:
            R = np.zeros((inact.size, nvar))
            for r, i in enumerate(inact):
                R[r, idx[f"a{l}"].start+i] = 1.0
            Aub_rows.append(R); bub.append(np.zeros(inact.size))

    Aub = np.vstack(Aub_rows) if Aub_rows else None
    bub = np.concatenate(bub) if bub else None

    lb = -np.inf*np.ones(nvar); ub = +np.inf*np.ones(nvar)
    X_lo, X_hi = X_bounds; U_lo, U_hi = U_bounds
    lb[idx["x"]], ub[idx["x"]] = X_lo, X_hi
    lb[idx["u"]], ub[idx["u"]] = U_lo, U_hi
    bounds = list(zip(lb, ub))

    res = linprog(np.zeros(nvar), A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
    if not res.success:
        return False, None, None
    z = res.x
    return True, z[idx["x"]], z[idx["u"]]
