import numpy as np
import networkx as nx
import random
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree


def generate_tree_topology(n_nodes: int) -> nx.Graph:
    """
    Создаёт случайное дерево с помощью MST на случайных точках.
    """
    points = np.random.rand(n_nodes, 2)
    dist = distance_matrix(points, points)
    mst = minimum_spanning_tree(dist).tocoo()
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for i, j in zip(mst.row, mst.col):
        G.add_edge(i, j)
    return G


def generate_random_graph(n_nodes: int, edge_prob: float = 0.4) -> nx.Graph:
    """
    Создаёт связный случайный граф.
    """
    G = nx.erdos_renyi_graph(n_nodes, edge_prob)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(n_nodes, edge_prob)
    return G


def assign_legs_and_sizes(graph: nx.Graph, leg_size_range=(2, 16)):
    """
    Назначает каждой паре узлов (ребру) общий индекс (букву) и размер.
    """
    edge_legs = {}
    edge_sizes = {}
    used_labels = []
    leg_counter = 0

    for u, v in graph.edges():
        label = chr(97 + leg_counter)  # 'a', 'b', ...
        size = random.randint(*leg_size_range)
        edge_legs[(u, v)] = label
        edge_legs[(v, u)] = label
        edge_sizes[label] = size
        used_labels.append(label)
        leg_counter += 1

    return edge_legs, edge_sizes


def build_einsum_inputs(graph: nx.Graph, edge_legs, edge_sizes):
    tensors = []
    subscripts = []

    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        labels = [edge_legs[(node, neighbor)] for neighbor in neighbors]
        subscripts.append("".join(labels))
        shape = [edge_sizes[label] for label in labels]
        tensor = np.random.rand(*shape)
        tensors.append(tensor)

    input_subs = ",".join(subscripts)
    all_labels = sorted(set("".join(subscripts)))
    output_subs = "".join(all_labels)
    einsum_expr = f"{input_subs}->{output_subs}"

    return einsum_expr, tensors
