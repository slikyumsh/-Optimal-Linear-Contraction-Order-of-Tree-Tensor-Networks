# Optimal Linear Contraction Order of Tree Tensor Networks

This repository contains implementations and experiments for comparing tensor network contraction order algorithms on tree and general graph topologies. It includes:

* **Greedy** and **Dynamic Programming** heuristics using [opt\_einsum](https://github.com/alexcrichton/opt-einsum).
* **Hyper-optimization** using [Cotengra](https://github.com/jcmgray/cotengra) with sampling methods (`kahypar`, `greedy`, `labels`).
* **Two-phase ASI IKKBZ** algorithm from the literature, specialized for tree network contraction, guaranteeing optimal flop count for trees.
* Scripts to run benchmarks across varying network sizes, and generate CSV results and plots.

---

## Repository Structure

```
.- full_experiment.py          # Runs experiments varying n and topology, produces CSV & plots
.- run_experiments.py         # Single-size experiment comparing all methods
.- tensor_network_generator.py # Graph generators and einsum input builders
.- results_n.csv              # (auto) CSV of results for multiple n
.- cost_tree.png, cost_graph.png    # (auto) Cost vs n plots
.- time_tree.png, time_graph.png    # (auto) Time vs n plots
`README.md`                   # This file
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/Optimal-Linear-Contraction-Order-of-Tree-Tensor-Networks.git
   cd Optimal-Linear-Contraction-Order-of-Tree-Tensor-Networks
   ```

2. Create and activate a Python virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install required packages:

   ```bash
   pip install --upgrade pip setuptools wheel
   pip install numpy scipy networkx matplotlib pandas opt_einsum cotengra kahypar optuna nevergrad cmaes
   ```

   > **Note:** For full Cotengra hyper-optimization, install at least one of `optuna`, `nevergrad`, or `cmaes`.

## Usage

### Single Experiment

To run contraction order comparison on two sample networks (tree and general graph) of size *n*=10:

```bash
python run_experiments.py
```

This prints costs and times for each method.

### Full Benchmark Across Sizes

To sweep over multiple network sizes and generate result files and plots:

```bash
python full_experiment.py
```

Outputs:

* `results_n.csv`: benchmark data
* `cost_tree.png`, `cost_graph.png`: flop-cost vs number of nodes (log-scale)
* `time_tree.png`, `time_graph.png`: runtime vs number of nodes (log-scale)

## Algorithms Implemented

1. **opt\_einsum.greedy**: simple greedy contraction path.
2. **opt\_einsum.dp**: dynamic programming path minimizing flops and memory pressure.
3. **cotengra.hyper**: hyper-optimized path using Cotengra with sampling.
4. **tensor.ikkbz**: Two-phase ASI IKKBZ algorithm achieving optimal flop count on trees.

## Extending and Customizing

* **Network Topologies**: Modify `tensor_network_generator.py` to implement other graph models or weight distributions.
* **Hyper-optimizer Settings**: In `run_experiments.py` or `full_experiment.py`, adjust `ctg.HyperOptimizer` parameters like `n_trials`, sampling methods, or random seed.
* **Plot Styles**: Change Matplotlib settings or output formats in the plotting section of `full_experiment.py`.

## References

* *Optimal Linear Contraction Order of Tree Tensor Networks* (Article title)
* [opt\_einsum documentation](https://github.com/alexcrichton/opt-einsum)
* [Cotengra hyper-optimization](https://github.com/jcmgray/cotengra)

---

Feel free to open issues or submit pull requests for improvements!
Happy tensor contracting!

```
```
