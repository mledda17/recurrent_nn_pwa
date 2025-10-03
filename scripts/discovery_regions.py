import numpy as np
from rnn2pwa.models.rnn_relu import RNN, Layer
from rnn2pwa.utils.discovery import discover_regions

if __name__ == "__main__":
    np.random.seed(0)
    n_x, n_u = 2, 1
    W1 = np.array([[0.6,-0.2,0.5],[-0.1,0.9,-0.3],[0.2,0.4,0.1]]); b1 = np.array([0.0,0.1,-0.05])
    W2 = np.array([[1.0,0.5,-0.2],[0.3,-0.7,0.4]]); b2 = np.array([0.0,0.0])
    rnn = RNN([Layer(W1,b1), Layer(W2,b2)], n_x, n_u)

    X_bounds = (np.array([-1.0,-1.0]), np.array([1.0,1.0]))
    U_bounds = (np.array([-0.5]), np.array([0.5]))

    pats = discover_regions(rnn, X_bounds, U_bounds, N=4000)
    print("Unique regions discovered:", len(pats))
