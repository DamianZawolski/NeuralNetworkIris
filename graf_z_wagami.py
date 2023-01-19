import matplotlib.pyplot as plt
import networkx as nx

def graf_wielodzielny():

    layers = [["x1","x2","x3","x4"],["v1","v2","v3","v4"],["y1","y2","y3"],]

    G = nx.Graph()
    for (i, layer) in enumerate(layers):
        G.add_nodes_from(layer, layer=i)
    for i in range(4):
        for j in range(4):
            G.add_edge(f"x{i+1}", f"v{j+1}")
    for i in range(4):
        for j in range(3):
            G.add_edge(f"v{i + 1}", f"y{j + 1}", weight=i)
    return G

def rysuj_graf(wagi_wejsciowe, wagi_ukryte):
    podpisy = {}
    subset_color = [
        "darkorange",
        "violet",
        "limegreen"
    ]
    iter=0
    for i in range(4):
        for j in range(4):
            podpis = (f"x{i+1}",f"v{j+1}")
            podpisy[podpis] = round(wagi_wejsciowe[iter],2)
            iter+=1
    iter = 0
    for i in range(4):
        for j in range(3):
            podpis = (f"v{i + 1}", f"y{j + 1}")
            podpisy[podpis] = round(wagi_ukryte[iter],2)
            iter+=1
    G = graf_wielodzielny()
    color = [subset_color[data["layer"]] for v, data in G.nodes(data=True)]
    pos = nx.multipartite_layout(G, subset_key="layer")
    plt.figure(figsize=(8, 8))

    nx.draw(G, pos, node_color=color, with_labels=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=podpisy, label_pos=0.8)
    plt.axis("equal")
    plt.savefig('graf.png', bbox_inches='tight')
    plt.show()

