import matplotlib.pyplot as plt

def plot_graph_networkx(G: dict):
    try:
        import networkx as nx
    except ImportError:
        print("networkx not installed: skipping graph plot.")
        return
    g = nx.DiGraph()
    for n, neigh in G.items():
        g.add_node(n)
        for m in neigh: g.add_edge(n, m)
    plt.figure(figsize=(7,6))
    pos = nx.spring_layout(g, seed=0)
    # label compatto: indice progressivo
    labels = {node: str(i) for i, node in enumerate(g.nodes())}
    nx.draw_networkx_nodes(g, pos, node_size=400)
    nx.draw_networkx_labels(g, pos, labels=labels, font_size=8)
    nx.draw_networkx_edges(g, pos, arrows=True, arrowstyle="-|>", width=1.2)
    plt.title(f"Region transition graph (|V|={g.number_of_nodes()}, |E|={g.number_of_edges()})")
    plt.axis("off"); plt.tight_layout(); plt.show()
