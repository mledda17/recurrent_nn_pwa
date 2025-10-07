import numpy as np
from typing import Dict, List, Tuple, Set
from scipy.optimize import linprog
from rnn2pwa.models.rnn_relu import RNN

Pattern = Tuple[Tuple[int, ...], ...]


def _boundary_feasible_lp_relu_soft(rnn, sig_a, sig_b, X_bounds, U_bounds,
                                    tau=1e-7, eps_active=1e-10, eps_inactive=1e-10):
    """
    Verifica se Rc(sig_a) e Rc(sig_b) confinano: differenza di UN neurone.
    Frontiera: |a_{l*,i*}| <= tau. Per gli altri:
      - attivo: a >= eps_active e h=a
      - inattivo: a <= -eps_inactive e h=0
    """
    import numpy as np
    from scipy.optimize import linprog
    layers = rnn.layers; n_x, n_u = rnn.n_x, rnn.n_u
    sizes = [L.W.shape[0] for L in layers]

    # individua unico neurone che differisce
    diff = [(l,i)
            for l,(pa,pb) in enumerate(zip(sig_a, sig_b), start=1)
            for i,(va,vb) in enumerate(zip(pa,pb)) if va!=vb]
    if len(diff)!=1: return False
    l_star, i_star = diff[0]

    # indicizzazione variabili
    idx, cur = {}, 0
    idx["x"]=slice(cur,cur+n_x); cur+=n_x
    idx["u"]=slice(cur,cur+n_u); cur+=n_u
    for l,n_l in enumerate(sizes, start=1):
        idx[f"a{l}"]=slice(cur,cur+n_l); cur+=n_l
        idx[f"h{l}"]=slice(cur,cur+n_l); cur+=n_l
    nvar=cur

    Aeq_rows, beq = [], []
    Aub_rows, bub = [], []

    # propagazione a_l = W*[prev] + b
    for l,(layer,pa,pb) in enumerate(zip(layers, sig_a, sig_b), start=1):
        W,b = layer.W, layer.b
        n_l = W.shape[0]
        row = np.zeros((n_l, nvar)); row[:, idx[f"a{l}"]] = np.eye(n_l)
        if l==1:
            row[:, idx["x"]] -= W[:, :n_x]
            row[:, idx["u"]] -= W[:, n_x:]
        else:
            row[:, idx[f"h{l-1}"]] -= W
        Aeq_rows.append(row); beq.append(b.copy())

        pa = np.asarray(pa); pb = np.asarray(pb)
        for i in range(n_l):
            # ReLU coerenza h vs a
            r = np.zeros(nvar); r[idx[f"h{l}"].start+i] = 1.0
            if (l==l_star and i==i_star):
                # niente eq h=a: alla frontiera h può stare a 0 comunque
                Aeq_rows.append(r.reshape(1,-1)); beq.append(np.array([0.0]))
                # |a| <= tau
                r1 = np.zeros(nvar); r1[idx[f"a{l}"].start+i] = 1.0
                r2 = -r1.copy()
                Aub_rows += [r1.reshape(1,-1), r2.reshape(1,-1)]
                bub += [np.array([+tau]), np.array([+tau])]
            else:
                # neurone concorde su A e B
                if pa[i]==pb[i]==1:
                    # h=a
                    r[idx[f"a{l}"].start+i] = -1.0
                    Aeq_rows.append(r.reshape(1,-1)); beq.append(np.array([0.0]))
                    # a >= eps_active  ->  -a <= -eps_active
                    rr = np.zeros(nvar); rr[idx[f"a{l}"].start+i] = -1.0
                    Aub_rows.append(rr.reshape(1,-1)); bub.append(np.array([-eps_active]))
                elif pa[i]==pb[i]==0:
                    # h=0, a <= -eps_inactive
                    Aeq_rows.append(r.reshape(1,-1)); beq.append(np.array([0.0]))
                    rr = np.zeros(nvar); rr[idx[f"a{l}"].start+i] = 1.0
                    Aub_rows.append(rr.reshape(1,-1)); bub.append(np.array([-eps_inactive]))
                else:
                    return False  # differiscono ma non è il neurone di frontiera

    Aeq = np.vstack(Aeq_rows) if Aeq_rows else None
    beq = np.concatenate(beq) if beq else None
    Aub = np.vstack(Aub_rows) if Aub_rows else None
    bub = np.concatenate(bub) if bub else None

    lb = -np.inf*np.ones(nvar); ub = +np.inf*np.ones(nvar)
    X_lo,X_hi = X_bounds; U_lo,U_hi = U_bounds
    lb[idx["x"]], ub[idx["x"]] = X_lo, X_hi
    lb[idx["u"]], ub[idx["u"]] = U_lo, U_hi
    bounds = list(zip(lb,ub))

    res = linprog(np.zeros(nvar), A_ub=Aub, b_ub=bub, A_eq=Aeq, b_eq=beq,
                  bounds=bounds, method="highs")
    return bool(res.success)



def build_region_adjacency_graph(
    rnn: RNN,
    patterns: List[Pattern],
    X_bounds: Tuple[np.ndarray, np.ndarray],
    U_bounds: Tuple[np.ndarray, np.ndarray],
    eps: float = 1e-8,
) -> Tuple[List[Pattern], Set[Tuple[int, int]], Dict[Pattern, int]]:
    """
    Costruisce il grafo dei contatti tra regioni.
    - nodi: pattern in 'patterns'
    - archi: (i,j) se le regioni i e j sono confinanti (frontiera condivisa non vuota)

    Ritorna: (nodes, edges, idmap) con idmap: pattern -> id compatto.
    """
    nodes = list(patterns)
    idmap = {p: i for i, p in enumerate(nodes)}
    edges: Set[Tuple[int, int]] = set()

    patset = set(nodes)
    sizes = [L.W.shape[0] for L in rnn.layers]

    # Per ogni pattern, prova a flippare 1 neurone per volta: candidato vicino
    for p in nodes:
        for l, n_l in enumerate(sizes):
            for i in range(n_l):
                q = [list(row) for row in p]
                q[l][i] = 1 - q[l][i]
                q = tuple(tuple(row) for row in q)  # Pattern candidato
                if q not in patset:
                    continue
                # LP sulla frontiera a_{l,i}=0 per verificare contatto reale
                if _boundary_feasible_lp_relu_soft(rnn, p, q, X_bounds, U_bounds):
                    a, b = idmap[p], idmap[q]
                    if a != b:
                        edges.add((min(a, b), max(a, b)))
    return nodes, edges, idmap
