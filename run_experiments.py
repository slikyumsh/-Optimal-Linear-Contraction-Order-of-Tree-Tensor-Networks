import os
os.environ["PYTHONWARNINGS"] = "ignore"

import time
import random
import numpy as np
import opt_einsum as oe
import cotengra as ctg
from tensor_network_generator import (
    generate_tree_topology,
    generate_random_graph,
    assign_legs_and_sizes,
    build_einsum_inputs
)


def run_opt_einsum_greedy(expr, tensors):
    start = time.time()
    path_info = oe.contract_path(expr, *tensors, optimize='greedy')[1]
    end = time.time()
    return {"name": "opt_einsum.greedy", "cost": path_info.opt_cost, "time": end - start}


def run_opt_einsum_dp(expr, tensors):
    start = time.time()
    path_info = oe.contract_path(expr, *tensors, optimize='dynamic-programming')[1]
    end = time.time()
    return {"name": "opt_einsum.dp", "cost": path_info.opt_cost, "time": end - start}


def run_cotengra_hyper(expr, tensors):
    size_dict = {}
    input_subscripts = expr.split('->')[0].split(',')

    for subs, tensor in zip(input_subscripts, tensors):
        for idx, dim in zip(subs, tensor.shape):
            if idx in size_dict and size_dict[idx] != dim:
                raise ValueError(f"Inconsistent dimension for index '{idx}': {size_dict[idx]} vs {dim}")
            size_dict[idx] = dim

    opt = ctg.HyperOptimizer(methods=['greedy', 'labels'])

    start = time.time()
    try:
        path_info = opt.search(expr, size_dict)
        cost = float(path_info.get('flops', float('inf')))
    except Exception:
        print("⚠️  Cotengra failed to return a valid cost.")
        cost = float('inf')
    end = time.time()

    return {
        "name": "cotengra.hyper",
        "cost": cost,
        "time": end - start
    }


def log_to_csv(results, n_nodes, tree, filename="results.csv"):
    import csv
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["n_nodes", "tree", "method", "cost", "time"])
        if not file_exists:
            writer.writeheader()
        for r in results:
            writer.writerow({
                "n_nodes": n_nodes,
                "tree": tree,
                "method": r["name"],
                "cost": r["cost"],
                "time": r["time"]
            })


def plot_results(filename="results.csv"):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(filename)

    for metric in ['cost', 'time']:
        plt.figure()
        for (tree, method), group in df.groupby(['tree', 'method']):
            label = f"{'Tree' if tree else 'Graph'} | {method}"
            plt.plot(group['n_nodes'], group[metric], marker='o', label=label)
        plt.xlabel('Number of Nodes')
        plt.ylabel(metric.capitalize())
        plt.title(f"{metric.capitalize()} by Method")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{metric}_plot.png")
        plt.close()


def run_single_experiment(n_nodes=8, tree=True, seed=42):
    random.seed(seed)
    np.random.seed(seed)

    if tree:
        G = generate_tree_topology(n_nodes)
    else:
        G = generate_random_graph(n_nodes, edge_prob=0.4)

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

    for r in results:
        print(f"{r['name']:20} | Cost: {r['cost']:.2e} | Time: {r['time']:.4f} sec")

    log_to_csv(results, n_nodes, tree)


if __name__ == "__main__":
    # Очистим старый файл результатов
    if os.path.exists("results.csv"):
        os.remove("results.csv")

    for tree in [True, False]:
        for n in range(5, 11):
            run_single_experiment(n_nodes=n, tree=tree)

    plot_results()
