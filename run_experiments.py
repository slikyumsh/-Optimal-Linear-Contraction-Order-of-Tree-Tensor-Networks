import time
import random
import numpy as np
import opt_einsum as oe
import cotengra as ctg
import networkx as nx
import warnings

from tensor_network_generator import (
    generate_tree_topology,
    generate_random_graph,
    assign_legs_and_sizes,
    build_einsum_inputs
)

def extract_cost(info):
    """
    Пробуем достать из PathInfo сначала opt_cost, потом другие поля.
    """
    if hasattr(info, 'opt_cost'):
        return info.opt_cost
    if hasattr(info, 'flops'):
        return info.flops
    if hasattr(info, 'total_cost'):
        return info.total_cost
    raise RuntimeError("Не удалось извлечь число флопов из PathInfo")

def run_opt_einsum_greedy(expr, tensors):
    start = time.time()
    path, info = oe.contract_path(expr, *tensors, optimize='greedy')
    cost = extract_cost(info)
    end = time.time()
    return {"name": "opt_einsum.greedy", "cost": cost, "time": end - start}

def run_opt_einsum_dp(expr, tensors):
    start = time.time()
    path, info = oe.contract_path(expr, *tensors, optimize='dynamic-programming')
    cost = extract_cost(info)
    end = time.time()
    return {"name": "opt_einsum.dp", "cost": cost, "time": end - start}

def run_cotengra_hyper(expr, tensors):
    size_dict = {}
    input_subscripts = expr.split('->')[0].split(',')

    for subs, tensor in zip(input_subscripts, tensors):
        for idx, dim in zip(subs, tensor.shape):
            if idx in size_dict:
                if size_dict[idx] != dim:
                    raise ValueError(f"Inconsistent dimension for index '{idx}': {size_dict[idx]} vs {dim}")
            else:
                size_dict[idx] = dim

    opt = ctg.HyperOptimizer(methods=['greedy', 'labels'])

    start = time.time()
    try:
        path_info = opt.search(input_subscripts, expr.split('->')[1], size_dict)
        cost = float(path_info.get_total_cost())
    except Exception:
        cost = float('inf')
    end = time.time()

    return {
        "name": "cotengra.hyper",
        "cost": cost,
        "time": end - start
    }

def compute_subtree_cost(node, T, subscripts_map, size_dict):
    children = list(T.successors(node))
    if not children:
        indices = subscripts_map[node]
        size = np.prod([size_dict[i] for i in indices])
        return set(indices), size, 0

    total_cost = 0
    current_indices = set(subscripts_map[node])
    for child in children:
        idxs, sz, cost = compute_subtree_cost(child, T, subscripts_map, size_dict)
        total_cost += cost
        current_indices |= idxs

    current_size = np.prod([size_dict[i] for i in current_indices])
    total_cost += current_size

    return current_indices, current_size, total_cost

def run_tensor_ikkbz(expr, tensors, G):
    start = time.time()

    if not nx.is_tree(G):
        raise ValueError("TensorIKKBZ применяется только к деревьям!")

    root = list(G.nodes())[0]
    T = nx.dfs_tree(G, source=root)

    input_subscripts = expr.split('->')[0].split(',')
    size_dict = {}
    for subs, t in zip(input_subscripts, tensors):
        for i, dim in zip(subs, t.shape):
            if i not in size_dict:
                size_dict[i] = dim
            elif size_dict[i] != dim:
                raise ValueError(f"Inconsistent dimension for index {i}: {size_dict[i]} vs {dim}")

    subscripts_map = dict(zip(G.nodes(), input_subscripts))
    _, _, cost = compute_subtree_cost(root, T, subscripts_map, size_dict)

    end = time.time()
    return {
        "name": "tensor.ikkbz",
        "cost": cost,
        "time": end - start
    }

def run_single_experiment(n_nodes=8, tree=True, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    warnings.filterwarnings("ignore")

    G = generate_tree_topology(n_nodes) if tree else generate_random_graph(n_nodes, edge_prob=0.4)
    edge_legs, edge_sizes = assign_legs_and_sizes(G)
    expr, tensors = build_einsum_inputs(G, edge_legs, edge_sizes)

    print(f"\nEinsum expression: {expr}")

    results = []

    try:
        results.append(run_opt_einsum_greedy(expr, tensors))
    except Exception as e:
        print("Greedy failed:", e)

    try:
        results.append(run_opt_einsum_dp(expr, tensors))
    except Exception as e:
        print("DP failed:", e)

    try:
        results.append(run_cotengra_hyper(expr, tensors))
    except Exception as e:
        print("Cotengra Hyper failed:", e)

    if tree:
        try:
            results.append(run_tensor_ikkbz(expr, tensors, G))
        except Exception as e:
            print("TensorIKKBZ failed:", e)

    for r in results:
        print(f"{r['name']:20} | Cost: {r['cost']:.2e} | Time: {r['time']:.4f} sec")

if __name__ == "__main__":
    run_single_experiment(n_nodes=10, tree=True)   # дерево
    run_single_experiment(n_nodes=10, tree=False)  # недерево
