# lp_feasibility.py
import numpy as np
from scipy.optimize import linprog

def _as_1d(arr):
    """Ensure a numpy 1D float array."""
    a = np.asarray(arr, dtype=float)
    return a.reshape(-1)

def pattern_feasible_lp_relu(rnn, pattern, bounds_x, bounds_u):
    """
    Check feasibility of a ReLU activation pattern via LP.

    Variables order:
        z = [ x (n_x), u (n_u), a1 (n1), h1 (n1), a2 (n2), h2 (n2), ..., aL (n_out) ]
    Notes:
        - Only hidden layers (all except last) are ReLU-activated.
        - Last layer is linear: we include only a_last (no h_last).
        - Objective is zero: feasibility problem.

    Args:
        rnn: object with attribute .layers (list of Layer), each Layer has W (2D) and b (1D/2D).
        pattern: list/tuple of binary vectors, one per hidden layer, length must match hidden sizes.
                 Example for 3 ReLU neurons in the first (and only) hidden layer: pattern = [[1,0,1]]
        bounds_x: tuple (x_min, x_max), each 1D array-like of length n_x
        bounds_u: tuple (u_min, u_max), each 1D array-like of length n_u

    Returns:
        feasible (bool), x_witness (np.ndarray or None), u_witness (np.ndarray or None)
    """
    try:
        layers = rnn.layers
    except Exception:
        print("[WARN] rnn object has no 'layers' attribute.")
        return False, None, None

    # Parse bounds
    x_min = _as_1d(bounds_x[0])
    x_max = _as_1d(bounds_x[1])
    u_min = _as_1d(bounds_u[0])
    u_max = _as_1d(bounds_u[1])
    n_x = x_min.size
    n_u = u_min.size

    # Split layers: hidden (ReLU) and last (linear)
    if len(layers) < 1:
        print("[WARN] Empty network.")
        return False, None, None
    hidden_layers = layers[:-1]
    last_layer = layers[-1]

    # Hidden sizes and consistency checks
    hidden_sizes = []
    for L in hidden_layers:
        W = np.asarray(L.W, dtype=float)
        b = _as_1d(L.b)
        if W.shape[0] != b.size:
            print("[WARN] Inconsistent layer shapes: W rows != b size.")
            return False, None, None
        hidden_sizes.append(W.shape[0])

    # Pattern checks
    # pattern must have one entry per hidden layer
    if len(hidden_layers) == 0:
        # No hidden ReLU layers: allow pattern to be empty list/tuple
        if pattern not in ([], (), None):
            # ignore provided pattern but warn
            pass
    else:
        if not isinstance(pattern, (list, tuple)):
            print("[WARN] pattern must be a list/tuple of per-layer binary vectors.")
            return False, None, None
        if len(pattern) != len(hidden_layers):
            print(f"[WARN] pattern length {len(pattern)} does not match number of hidden layers {len(hidden_layers)}.")
            return False, None, None
        # check each layer length
        for l, (p_l, n_l) in enumerate(zip(pattern, hidden_sizes)):
            if len(p_l) != n_l:
                print(f"[WARN] pattern[{l}] length {len(p_l)} does not match hidden size {n_l}.")
                return False, None, None
            # check binary content
            if np.any((np.asarray(p_l) != 0) & (np.asarray(p_l) != 1)):
                print(f"[WARN] pattern[{l}] must be binary (0/1).")
                return False, None, None

    # Dimensions along the chain
    # input to first hidden layer is [x; u] of size (n_x + n_u)
    # each hidden layer l: a_l, h_l have size hidden_sizes[l]
    # last layer maps from h_{L-1} to a_last of size n_out
    W_last = np.asarray(last_layer.W, dtype=float)
    b_last = _as_1d(last_layer.b)
    n_out = W_last.shape[0]
    if n_out != b_last.size:
        print("[WARN] Last layer W rows != b size.")
        return False, None, None

    # Count total decision variables:
    # x (n_x) + u (n_u) + sum over hidden of (a_l + h_l) + a_last (n_out)
    n_vars = n_x + n_u + sum(2 * s for s in hidden_sizes) + n_out

    # Prepare constraint containers
    A_eq_list = []
    b_eq_list = []
    A_ub_list = []
    b_ub_list = []

    # Index bookkeeping
    idx = 0
    idx_x = np.arange(idx, idx + n_x); idx += n_x
    idx_u = np.arange(idx, idx + n_u); idx += n_u
    idx_prev_h = np.concatenate([idx_x, idx_u])  # h_0

    # Build constraints for hidden layers
    for l, L in enumerate(hidden_layers):
        W = np.asarray(L.W, dtype=float)
        b = _as_1d(L.b)
        n_l = W.shape[0]

        # a_l and h_l indices
        idx_a_l = np.arange(idx, idx + n_l); idx += n_l
        idx_h_l = np.arange(idx, idx + n_l); idx += n_l

        # a_l = W * h_{l-1} + b
        Aeq = np.zeros((n_l, n_vars))
        beq = np.zeros(n_l)
        for i in range(n_l):
            Aeq[i, idx_a_l[i]] = 1.0
            # subtract W[i, :] on previous h indices
            Aeq[i, idx_prev_h] -= W[i, :]
            beq[i] = b[i]
        A_eq_list.append(Aeq)
        b_eq_list.append(beq)

        # ReLU constraints per neuron based on pattern
        sigma = np.asarray(pattern[l], dtype=int) if len(hidden_layers) > 0 else np.zeros(n_l, dtype=int)
        for i in range(n_l):
            if sigma[i] == 1:
                # Active: h_i = a_i and a_i >= 0  -> h_i - a_i = 0,  -a_i <= 0
                row_eq = np.zeros(n_vars)
                row_eq[idx_h_l[i]] = 1.0
                row_eq[idx_a_l[i]] = -1.0
                A_eq_list.append(row_eq.reshape(1, -1))
                b_eq_list.append(np.array([0.0]))

                row_ub = np.zeros(n_vars)
                row_ub[idx_a_l[i]] = -1.0  # -a_i <= 0  => a_i >= 0
                A_ub_list.append(row_ub.reshape(1, -1))
                b_ub_list.append(np.array([0.0]))
            else:
                # Inactive: h_i = 0 and a_i <= 0  -> h_i = 0,  a_i <= 0
                row_eq = np.zeros(n_vars)
                row_eq[idx_h_l[i]] = 1.0
                A_eq_list.append(row_eq.reshape(1, -1))
                b_eq_list.append(np.array([0.0]))

                row_ub = np.zeros(n_vars)
                row_ub[idx_a_l[i]] = 1.0   # a_i <= 0
                A_ub_list.append(row_ub.reshape(1, -1))
                b_ub_list.append(np.array([0.0]))

        # next layer input becomes h_l
        idx_prev_h = idx_h_l

    # Last layer (linear only): a_last = W_last * h_{L-1} + b_last
    idx_a_last = np.arange(idx, idx + n_out)
    Aeq_last = np.zeros((n_out, n_vars))
    beq_last = np.zeros(n_out)
    for i in range(n_out):
        Aeq_last[i, idx_a_last[i]] = 1.0
        Aeq_last[i, idx_prev_h] -= W_last[i, :]
        beq_last[i] = b_last[i]
    A_eq_list.append(Aeq_last)
    b_eq_list.append(beq_last)

    # Stack constraints
    A_eq = np.vstack(A_eq_list) if len(A_eq_list) > 0 else None
    b_eq = np.concatenate(b_eq_list) if len(b_eq_list) > 0 else None
    if len(A_ub_list) > 0:
        A_ub = np.vstack(A_ub_list)
        b_ub = np.concatenate(b_ub_list)
    else:
        A_ub, b_ub = None, None

    # Variable bounds
    bounds = []
    # x bounds
    for i in range(n_x):
        bounds.append((x_min[i], x_max[i]))
    # u bounds
    for i in range(n_u):
        bounds.append((u_min[i], u_max[i]))
    # remaining vars unbounded
    for _ in range(n_vars - n_x - n_u):
        bounds.append((None, None))

    # Zero objective (feasibility)
    c = np.zeros(n_vars, dtype=float)

    # Solve LP safely
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=bounds, method="highs")
    except Exception as e:
        print(f"[WARN] linprog raised an exception for pattern {pattern}: {e}")
        return False, None, None

    if not res.success:
        # Infeasible or numerical issue
        return False, None, None

    # Extract witness (x,u)
    x_w = res.x[idx_x]
    u_w = res.x[idx_u]
    return True, x_w, u_w
