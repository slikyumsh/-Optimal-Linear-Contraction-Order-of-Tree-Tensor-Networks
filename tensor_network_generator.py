# tensor_network_generator.py

import numpy as np
import networkx as nx
import random
from scipy.spatial import distance_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

def generate_tree_topology(n_nodes: int) -> nx.Graph:
    points = np.random.rand(n_nodes, 2)
    dist = distance_matrix(points, points)
    mst = minimum_spanning_tree(dist).tocoo()
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for i, j in zip(mst.row, mst.col):
        G.add_edge(int(i), int(j))
    return G

def generate_random_graph(n_nodes: int, edge_prob: float = 0.4) -> nx.Graph:
    G = nx.erdos_renyi_graph(n_nodes, edge_prob)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(n_nodes, edge_prob)
    return G

def assign_legs_and_sizes(graph: nx.Graph, leg_size_range=(2, 16)):
    edge_legs = {}
    edge_sizes = {}
    leg_counter = 0
    for u, v in sorted(graph.edges()):
        label = chr(97 + leg_counter)
        size = random.randint(*leg_size_range)
        edge_legs[(u, v)] = label
        edge_legs[(v, u)] = label
        edge_sizes[label] = size
        leg_counter += 1
    return edge_legs, edge_sizes

def build_einsum_inputs(graph: nx.Graph, edge_legs, edge_sizes):
    tensors = []
    subscripts = []
    for node in sorted(graph.nodes()):
        nbrs = sorted(graph.neighbors(node))
        labels = [edge_legs[(node, nbr)] for nbr in nbrs]
        subscripts.append("".join(labels))
        shape = [edge_sizes[l] for l in labels]
        tensors.append(np.random.rand(*shape))
    input_subs = ",".join(subscripts)
    all_labels = sorted(set("".join(subscripts)))
    output_subs = "".join(all_labels)
    return f"{input_subs}->{output_subs}", tensors
