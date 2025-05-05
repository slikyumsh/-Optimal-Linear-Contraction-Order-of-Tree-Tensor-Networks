#!/usr/bin/env python3
# full_experiment.py

import time
import random
import numpy as np
import opt_einsum as oe
import cotengra as ctg
import networkx as nx
import heapq
import itertools
import matplotlib.pyplot as plt
import pandas as pd

from tensor_network_generator import (
    generate_tree_topology,
    generate_random_graph,
    assign_legs_and_sizes,
    build_einsum_inputs
)

def extract_cost(info):
    """Извлекает числовую оценку флопов из PathInfo."""
    for attr in ('opt_cost', 'cost', 'flops', 'total_cost'):
        if hasattr(info, attr):
            return getattr(info, attr)
    raise RuntimeError(f"Cannot extract cost from PathInfo: {info}")

def run_opt_einsum(expr, tensors, method):
    t0 = time.time()
    _, info = oe.contract_path(expr, *tensors, optimize=method)
    return extract_cost(info), time.time() - t0

def run_cotengra(expr, tensors):
    inp, out = expr.split('->')
    inputs = inp.split(',')
    size_dict = {idx: dim
                 for subs, t in zip(inputs, tensors)
                 for idx, dim in zip(subs, t.shape)}
    opt = ctg.HyperOptimizer(methods=['kahypar', 'greedy', 'labels'])
    t0 = time.time()
    tree = opt.search(inputs, out, size_dict)
    cost = int(tree.contraction_cost())
    return cost, time.time() - t0

def tensor_ikkbz(G, edge_legs, edge_sizes):
    """
    Двухфазный ASI-алгоритм IKKBZ для дерева G.
    Возвращает суммарный flop-cost.
    """
    # 1) Root the tree at node 0
    root = 0
    T = nx.dfs_tree(G, source=root)       # directed parent->child
    parent = {v: p for p, v in T.edges()}
    cluster_edges = dict(edge_legs)
    size_dict = dict(edge_sizes)

    # 2) Recursively summarize subtree v under parent p
    def summarize_subtree(v, p):
        full_C = 0
        # collect current indices: edge to parent (if any) and edges to children
        neighbors = ([p] if p is not None else []) + list(T.successors(v))
        idxs = set(cluster_edges[(v, u)] for u in neighbors)
        # recurse on children
        for w in T.successors(v):
            c_full, c_idxs, c_merge = summarize_subtree(w, v)
            full_C += c_full
            idxs |= c_idxs
        # compute merge cost for v->p
        if p is None:
            merge = 0
        else:
            leg = cluster_edges[(v, p)]
            no_par = idxs - {leg}
            T_size = np.prod([size_dict[i] for i in no_par]) if no_par else 1
            merge = T_size * size_dict[leg]
        full_C += merge
        return full_C, idxs, merge

    # 3) Build heap of events (v->parent) with ASI-score priority
    REMOVED = object()
    entry_finder = {}
    heap = []
    counter = itertools.count()

    def add_or_update(v):
        p = parent.get(v)
        if p is None:
            return
        full_C, idxs, merge = summarize_subtree(v, p)
        leg = cluster_edges[(v, p)]
        no_par = idxs - {leg}
        size_no_par = np.prod([size_dict[i] for i in no_par]) if no_par else 1
        sigma = (full_C, size_dict[leg] - size_no_par)
        # approximate ASI key
        key = full_C / max(1, abs(sigma[1]))
        entry = [key, next(counter), v]
        if v in entry_finder:
            entry_finder[v][-1] = REMOVED
        entry_finder[v] = entry
        heapq.heappush(heap, entry)

    for v in G.nodes():
        add_or_update(v)

    total_cost = 0
    active = set(G.nodes())

    # 4) Main contraction loop
    while len(active) > 1:
        key, _, v = heapq.heappop(heap)
        if v is REMOVED or v not in active:
            continue
        p = parent[v]
        _, _, merge = summarize_subtree(v, p)
        total_cost += merge
        active.remove(v)
        # reparent children of v to p
        for w in list(T.successors(v)):
            lbl = cluster_edges[(w, v)]
            cluster_edges[(w, p)] = lbl
            cluster_edges[(p, w)] = lbl
            parent[w] = p
            T.add_edge(p, w)
        T.remove_node(v)
        # update heap entries for p and its children
        add_or_update(p)
        for w in T.successors(p):
            add_or_update(w)

    return total_cost

def run_tensor_ikkbz(G, edge_legs, edge_sizes):
    t0 = time.time()
    cost = tensor_ikkbz(G, edge_legs, edge_sizes)
    return cost, time.time() - t0

def main():
    ns = [6, 8, 10, 12]
    records = []

    for n in ns:
        for is_tree in (True, False):
            for seed in (0, 1, 2):
                random.seed(seed)
                np.random.seed(seed)
                G = (generate_tree_topology(n)
                     if is_tree else generate_random_graph(n, 0.4))
                edge_legs, edge_sizes = assign_legs_and_sizes(G)
                expr, tensors = build_einsum_inputs(G, edge_legs, edge_sizes)

                cg, tg = run_opt_einsum(expr, tensors, 'greedy')
                cd, td = run_opt_einsum(expr, tensors, 'dynamic-programming')
                ch, th = run_cotengra(expr, tensors)
                ci, ti = (run_tensor_ikkbz(G, edge_legs, edge_sizes)
                          if is_tree else (np.nan, np.nan))

                records.append({
                    'n': n,
                    'tree': is_tree,
                    'cost_greedy': cg,   'time_greedy': tg,
                    'cost_dp': cd,       'time_dp': td,
                    'cost_hyper': ch,    'time_hyper': th,
                    'cost_ikkbz': ci,    'time_ikkbz': ti
                })

    df = pd.DataFrame(records)
    df.to_csv('results_n.csv', index=False)
    print("Saved results_n.csv")

    # Split into tree and graph
    for is_tree, label in ((True, 'tree'), (False, 'graph')):
        sub = df[df['tree'] == is_tree]

        # Cost plot
        plt.figure()
        for alg in ('greedy', 'dp', 'hyper', 'ikkbz'):
            if alg == 'ikkbz' and not is_tree:
                continue
            means = sub.groupby('n')[f'cost_{alg}'].mean()
            plt.plot(means.index, means.values, marker='o', label=alg)
        plt.xscale('linear')
        plt.yscale('log')
        plt.xlabel('Number of nodes')
        plt.ylabel('Average flop cost')
        plt.title(f'{label.capitalize()} networks: Cost vs n')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'cost_{label}.png')

        # Time plot
        plt.figure()
        for alg in ('greedy', 'dp', 'hyper', 'ikkbz'):
            if alg == 'ikkbz' and not is_tree:
                continue
            means = sub.groupby('n')[f'time_{alg}'].mean()
            plt.plot(means.index, means.values, marker='o', label=alg)
        plt.xscale('linear')
        plt.yscale('log')
        plt.xlabel('Number of nodes')
        plt.ylabel('Average time (s)')
        plt.title(f'{label.capitalize()} networks: Time vs n')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'time_{label}.png')

if __name__ == '__main__':
    main()
