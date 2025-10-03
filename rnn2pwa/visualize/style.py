# rnn2pwa/visualize/style.py
import matplotlib as mpl

def set_paper_style():
    mpl.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 140,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "#FBFBFB",
        "figure.facecolor": "white",
    })
