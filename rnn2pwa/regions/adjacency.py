# rnn2pwa/regions/adjacency.py
import numpy as np
from typing import Tuple, Dict, Set, List, Optional
from scipy.optimize import linprog
from rnn2pwa.models.rnn_relu import RNN

Pattern = Tuple[Tuple[int, ...], ...]
FlipIndex = Tuple[int, int]  # (layer_index, neuron_index)

def _flip_bit(pat: Pattern, l: int, i: int) -> Pattern:
    new = [list(row) for row in pat]
    new[l][i] = 1 - new[l][i]
    return tuple(tuple(r) for r in new)

def _boundary_lp_feasible_on_facet(
    rnn: RNN,
    sigma: Pattern,
    flip: FlipIndex,
    X_bounds: Tuple[np.ndarray, np.ndarray],
    U_bounds: Tuple[np.ndarray, np.ndarray],
    eps_pos: float = 1e-6,
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Verifica se esiste un punto sulla 'faccia' comune tra la regione di 'sigma'
    e quella ottenuta flip(p) sul neurone (l,i):
      - per tutti i neuroni diversi da (l,i): come in sigma (ReLU), con a >= eps_pos se attivo, a <= 0 se inattivo
      - per (l,i): imponiamo a_{l,i} = 0 e h_{l,i} = 0 (vincolo di frontiera)
    Se fattibile -> le due regioni confinano.
    """
    layers = rnn.layers
    n_x, n_u = rnn.n_x, rnn.n_u
    sizes = [L.W.shape[0] for L in layers]

    # Layout variabili: [x | u | a1 | h1 | ... | aL | hL]
    idx, cur = {}, 0
    idx["x"] = slice(cur, cur + n_x); cur += n_x
    idx["u"] = slice(cur, cur + n_u); cur += n_u
    for l, n_l in enumerate(sizes, start=1):
        idx[f"a{l}"] = slice(cur, cur + n_l); cur += n_l
        idx[f"h{l}"] = slice(cur, cur + n_l); cur += n_l
    nvar = cur

    Aeq_rows, beq = [], []

    for l, (layer, sig_l) in enumerate(zip(layers, sigma), start=1):
        W, b = layer.W, layer.b
        n_l = W.shape[0]

        # a_l - W h_{l-1} = b_l  (h0 = [x;u])
        row = np.zeros((n_l, nvar))
        row[:, idx[f"a{l}"]] = np.eye(n_l)
        if l == 1:
            row[:, idx["x"]] -= W[:, :n_x]
            row[:, idx["u"]] -= W[:, n_x:]
        else:
            row[:, idx[f"h{l-1}"]] -= W
        Aeq_rows.append(row); beq.append(b.copy())

        # ReLU equalities:
        #  - per neuroni != flip:
        #       attivo:   h - a = 0
        #       inattivo: h = 0
        #  - per (flip):  h = 0 e (imporremo anche a=0 con un'uguaglianza sotto)
        sig_arr = np.array(sig_l)
        for i in range(n_l):
            R = np.zeros(nvar)
            if (l-1, i) == (flip[0], flip[1]):
                # h_{l,i} = 0 (a_{l,i}=0 sarà in Aeq più giù)
                R[idx[f"h{l}"].start + i] = 1.0
                Aeq_rows.append(R.reshape(1, -1)); beq.append(np.array([0.0]))
            else:
                if sig_arr[i] == 1:
                    # h - a = 0
                    R[idx[f"h{l}"].start + i] = 1.0
                    R[idx[f"a{l}"].start + i] = -1.0
                    Aeq_rows.append(R.reshape(1, -1)); beq.append(np.array([0.0]))
                else:
                    # h = 0
                    R[idx[f"h{l}"].start + i] = 1.0
                    Aeq_rows.append(R.reshape(1, -1)); beq.append(np.array([0.0]))

    # Uguaglianza di frontiera per il flip: a_{l,i} = 0
    l0, i0 = flip
    Rflip = np.zeros(nvar)
    Rflip[idx[f"a{l0+1}"].start + i0] = 1.0
    Aeq_rows.append(Rflip.reshape(1, -1)); beq.append(np.array([0.0]))

    Aeq = np.vstack(Aeq_rows) if Aeq_rows else None
    beq = np.concatenate(beq) if beq else None

    # Disuguaglianze per i neuroni != flip:
    #   attivo:   a >= eps_pos  =>  -a <= -eps_pos
    #   inattivo: a <= 0
    Aub_rows, bub = [], []
    for l, sig_l in enumerate(sigma, start=1):
        sig_arr = np.array(sig_l)
        for i in range(sig_arr.size):
            if (l-1, i) == (flip[0], flip[1]):
                continue  # il flip è in uguaglianza a=0
            if sig_arr[i] == 1:
                R = np.zeros(nvar); R[idx[f"a{l}"].start + i] = -1.0
                Aub_rows.append(R); bub.append(-eps_pos)
            else:
                R = np.zeros(nvar); R[idx[f"a{l}"].start + i] = 1.0
                Aub_rows.append(R); bub.append(0.0)

    Aub = np.vstack(Aub_rows) if Aub_rows else None
    bub = np.array(bub) if bub else None

    # Limiti su x,u
    lb = -np.inf * np.ones(nvar); ub = +np.inf * np.ones(nvar)
    X_lo, X_hi = X_bounds; U_lo, U_hi = U_bounds
    lb[idx["x"]], ub[idx["x"]] = X_lo, X_hi
    lb[idx["u"]], ub[idx["u"]] = U_lo, U_hi
    bounds = list(zip(lb, ub))

    res = linprog(
        c=np.zeros(nvar),
        A_ub=Aub, b_ub=bub,
        A_eq=Aeq, b_eq=beq,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        return False, None, None
    z = res.x
    return True, z[idx["x"]], z[idx["u"]]

def build_structural_adjacency(
    rnn: RNN,
    patterns: List[Pattern],
    X_bounds: Tuple[np.ndarray, np.ndarray],
    U_bounds: Tuple[np.ndarray, np.ndarray],
    eps_pos: float = 1e-6,
) -> Tuple[Dict[Pattern, Set[Pattern]], Dict[Tuple[Pattern, Pattern], Tuple[np.ndarray, np.ndarray]]]:
    """
    Restituisce:
      - G: grafo non orientato {pattern -> set(pattern confinanti)}
      - witnesses: mappa (p,q) -> (x*, u*) sulla faccia comune (se trovata)
    Solo coppie che differiscono per un solo bit vengono testate (facce).
    """
    pat_set = set(patterns)
    L = len(rnn.layers)
    sizes = [Lr.W.shape[0] for Lr in rnn.layers]

    G: Dict[Pattern, Set[Pattern]] = {p: set() for p in patterns}
    witnesses: Dict[Tuple[Pattern, Pattern], Tuple[np.ndarray, np.ndarray]] = {}

    for p in patterns:
        # prova a flippare ogni neurone: se il pattern esiste, test faccia
        for l in range(L):
            for i in range(sizes[l]):
                q = _flip_bit(p, l, i)
                if q not in pat_set:
                    continue
                # evita di testare due volte (usa ordine lessicografico)
                if q < p:
                    continue
                feas, xw, uw = _boundary_lp_feasible_on_facet(rnn, p, (l, i), X_bounds, U_bounds, eps_pos=eps_pos)
                if feas:
                    G[p].add(q); G[q].add(p)
                    witnesses[(p, q)] = (xw, uw)
                    witnesses[(q, p)] = (xw, uw)
    return G, witnesses
