import time
import random
import numpy as np
import opt_einsum as oe
import cotengra as ctg
import networkx as nx
import warnings
import heapq
import itertools

from tensor_network_generator import (
    generate_tree_topology,
    generate_random_graph,
    assign_legs_and_sizes,
    build_einsum_inputs
)

def extract_cost(info):
    """Извлекает числовую оценку флопов из PathInfo."""
    for attr in ('opt_cost','cost','flops','total_cost'):
        if hasattr(info, attr):
            return getattr(info, attr)
    raise RuntimeError(f"Не удалось извлечь число флопов из PathInfo: {info}")

def run_opt_einsum(expr, tensors, method):
    t0 = time.time()
    _, info = oe.contract_path(expr, *tensors, optimize=method)
    return extract_cost(info), time.time() - t0

def run_cotengra_hyper(expr, tensors):
    # Собираем size_dict
    inp, out = expr.split('->')
    inputs = inp.split(',')
    size_dict = {idx:dim for subs,t in zip(inputs,tensors) for idx,dim in zip(subs,t.shape)}

    opt = ctg.HyperOptimizer(methods=['kahypar','greedy','labels'])
    t0 = time.time()
    tree = opt.search(inputs, out, size_dict)
    cost = int(tree.contraction_cost())
    return cost, time.time() - t0

def tensor_ikkbz(G, edge_legs, edge_sizes):
    """
    Двухфазный ASI TensorIKKBZ из статьи.
    Возвращает (total_cost, order).
    """
    # 1) Root tree at 0
    root = 0
    T = nx.dfs_tree(G, source=root)          # directed parent->child
    parent = {v:p for p,v in T.edges()}

    # 2) cluster_edges хранит лейблы текущих рёбер между кластерами
    cluster_edges = dict(edge_legs)
    size_dict = dict(edge_sizes)

    # 3) Рекурсивный подсчёт: возвращает (full_C, idxs, merge_cost)
    def summarize_subtree(v, p):
        full_C = 0
        # текущие индексы v: к родителю (если есть) и ко всем детям
        neighbors = []
        if p is not None: neighbors.append(p)
        children = list(T.successors(v))
        neighbors += children

        idxs = set(cluster_edges[(v,u)] for u in neighbors)
        # рекурсивно добавляем детей
        for w in children:
            child_full, child_idxs, child_merge = summarize_subtree(w, v)
            full_C += child_full
            idxs |= child_idxs

        # локальный merge_cost v->p
        if p is None:
            merge_cost = 0
        else:
            par_leg = cluster_edges[(v,p)]
            # индексы без ребра к родителю
            no_par = idxs - {par_leg}
            T_size = np.prod([size_dict[i] for i in no_par]) if no_par else 1
            merge_cost = T_size * size_dict[par_leg]

        full_C += merge_cost
        return full_C, idxs, merge_cost

    # 4) Строим min-кучу по ASI-score для каждого v->parent
    REMOVED = object()
    entry_finder = {}
    heap = []
    counter = itertools.count()

    def add_or_update(v):
        p = parent.get(v)
        if p is None:
            return
        full_C, idxs, merge_cost = summarize_subtree(v, p)
        par_leg = cluster_edges[(v,p)]
        no_par = idxs - {par_leg}
        size_no_par = np.prod([size_dict[i] for i in no_par]) if no_par else 1
        sigma = (full_C, size_dict[par_leg] - size_no_par)
        # ключ = full_C / (|sigma2| or 1)
        key = full_C / max(1, abs(sigma[1]))
        entry = [key, next(counter), v, sigma]
        if v in entry_finder:
            entry_finder[v][2] = REMOVED
        entry_finder[v] = entry
        heapq.heappush(heap, entry)

    for v in G.nodes():
        add_or_update(v)

    total_cost = 0
    order = []
    active = set(G.nodes())

    # 5) Основной цикл слияний
    while len(active) > 1:
        key, _, v, sigma = heapq.heappop(heap)
        if v is REMOVED or v not in active:
            continue
        p = parent[v]
        # берём только локальную merge_cost
        _, _, merge_cost = summarize_subtree(v, p)
        total_cost += merge_cost
        order.append((v,p))
        active.remove(v)

        # переназначаем детей v -> p
        for w in list(T.successors(v)):
            lbl = cluster_edges[(w,v)]
            cluster_edges[(w,p)] = lbl
            cluster_edges[(p,w)] = lbl
            parent[w] = p
            T.add_edge(p,w)
        T.remove_node(v)

        # обновляем приоритеты p и его детей
        add_or_update(p)
        for w in T.successors(p):
            add_or_update(w)

    return total_cost, order

def run_tensor_ikkbz(G, edge_legs, edge_sizes):
    t0 = time.time()
    cost, order = tensor_ikkbz(G, edge_legs, edge_sizes)
    return cost, time.time() - t0

def run_single_experiment(n_nodes=8, tree=True, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    warnings.filterwarnings("ignore")

    G = generate_tree_topology(n_nodes) if tree else generate_random_graph(n_nodes, edge_prob=0.4)
    edge_legs, edge_sizes = assign_legs_and_sizes(G)
    expr, tensors = build_einsum_inputs(G, edge_legs, edge_sizes)

    print(f"\nEinsum expression: {expr}")
    results = []

    cg, tg = run_opt_einsum(expr, tensors, 'greedy')
    results.append({"name":"opt_einsum.greedy","cost":cg,"time":tg})

    cd, td = run_opt_einsum(expr, tensors, 'dynamic-programming')
    results.append({"name":"opt_einsum.dp",    "cost":cd,"time":td})

    ch, th = run_cotengra_hyper(expr, tensors)
    results.append({"name":"cotengra.hyper",   "cost":ch,"time":th})

    if tree:
        ci, ti = run_tensor_ikkbz(G, edge_legs, edge_sizes)
        results.append({"name":"tensor.ikkbz",    "cost":ci,"time":ti})

    for r in results:
        print(f"{r['name']:20} | Cost: {r['cost']:.2e} | Time: {r['time']:.4f} sec")

if __name__ == "__main__":
    run_single_experiment(n_nodes=10, tree=True)
    run_single_experiment(n_nodes=10, tree=False)
