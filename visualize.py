# visualize.py
# Gate distribution plots — shows how pruned vs active gates cluster.

import os
import matplotlib.pyplot as plt

from config import OUTPUT_DIR, PLOT_FILENAME
from utils import collect_all_gates


COLORS = ["#4C72B0", "#DD8452", "#55A868"]


def plot_gate_distributions(results: list, models: list):
    """
    Saves a side-by-side histogram of gate values for each lambda.

    A successful run shows:
    - A large spike near 0  (pruned connections)
    - A separate cluster above 0.1  (active connections)

    As lambda increases, the spike at 0 gets bigger.

    Args:
        results: list of (lambda, accuracy, sparsity) tuples
        models:  list of trained SelfPruningNet models (same order as results)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4))
    if len(results) == 1:
        axes = [axes]

    for ax, (lam, acc, sparsity), model, color in zip(axes, results, models, COLORS):
        gates = collect_all_gates(model)

        ax.hist(gates, bins=60, color=color, edgecolor="none", alpha=0.85)
        ax.axvline(0.01, color="red", linestyle="--", linewidth=1.2, label="Prune threshold (0.01)")
        ax.set_title(f"lambda = {lam:.0f}  |  acc = {acc:.1f}%  |  sparse = {sparsity:.1f}%", fontsize=10)
        ax.set_xlabel("Gate value")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Frequency")
    plt.suptitle("Gate Value Distribution Across lambda Values", fontsize=13)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, PLOT_FILENAME)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gate distribution plot saved → {save_path}")
