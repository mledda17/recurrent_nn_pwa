import numpy as np
import matplotlib.pyplot as plt

def plot_trajectories(time, X_rnn, X_pwa, title="RNN vs PWA"):
    plt.figure(figsize=(8,4))
    for i in range(X_rnn.shape[1]):
        plt.plot(time, X_rnn[:,i], label=f"RNN x[{i}]")
        plt.plot(time, X_pwa[:,i], "--", label=f"PWA x[{i}]")
    plt.xlabel("k"); plt.ylabel("state"); plt.title(title)
    plt.legend(); plt.tight_layout(); plt.show()

def plot_error(time, X_rnn, X_pwa):
    plt.figure(figsize=(6,3.5))
    err = np.linalg.norm(X_rnn - X_pwa, axis=1)
    plt.plot(time, err)
    plt.xlabel("k"); plt.ylabel("||err||2")
    plt.title("Trajectory error (RNN - PWA)")
    plt.tight_layout(); plt.show()
