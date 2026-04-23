# evaluate.py
# Evaluation utilities: test accuracy + sparsity level.
# Can also be run standalone to inspect a saved model checkpoint.

import logging
import torch

from config import DEVICE, PRUNE_THRESHOLD
from model import PrunableLinear, SelfPruningNet
from utils import get_data_loaders

log = logging.getLogger(__name__)


def evaluate(model: torch.nn.Module, loader, device) -> tuple[float, float]:
    """
    Returns (test_accuracy %, sparsity %) for the given model.

    Sparsity is defined as the fraction of FC weights whose gate value
    is below PRUNE_THRESHOLD. A gate < 0.01 contributes less than 1%
    of its weight's magnitude — practically zero.
    """
    accuracy  = _compute_accuracy(model, loader, device)
    sparsity  = _compute_sparsity(model)
    return accuracy, sparsity


def _compute_accuracy(model, loader, device) -> float:
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            preds   = model(data).argmax(dim=1)
            correct += preds.eq(target).sum().item()

    return 100.0 * correct / len(loader.dataset)


def _compute_sparsity(model, threshold: float = PRUNE_THRESHOLD) -> float:
    total  = 0
    pruned = 0

    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates   = torch.sigmoid(m.gate_scores).detach()
            total  += gates.numel()
            pruned += (gates < threshold).sum().item()

    if total == 0:
        return 0.0
    return 100.0 * pruned / total


def print_sparsity_report(model: torch.nn.Module):
    """Prints a per-layer breakdown of gate sparsity."""
    print(f"\n{'Layer':<30} {'Total gates':>12} {'Pruned':>10} {'Sparsity':>10}")
    print("─" * 65)

    for name, m in model.named_modules():
        if isinstance(m, PrunableLinear):
            gates   = torch.sigmoid(m.gate_scores).detach()
            total   = gates.numel()
            pruned  = (gates < PRUNE_THRESHOLD).sum().item()
            pct     = 100.0 * pruned / total
            print(f"{name:<30} {total:>12,} {pruned:>10,} {pct:>9.2f}%")

    print()


#  Standalone usage 

if __name__ == "__main__":
    import sys

    checkpoint = sys.argv[1] if len(sys.argv) > 1 else None

    if checkpoint:
        model = SelfPruningNet().to(DEVICE)
        model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
        print(f"Loaded checkpoint: {checkpoint}")
    else:
        print("No checkpoint provided — using a randomly initialised model for demo.")
        model = SelfPruningNet().to(DEVICE)

    _, test_loader = get_data_loaders()
    acc, sparsity  = evaluate(model, test_loader, DEVICE)

    print(f"Test accuracy : {acc:.2f}%")
    print(f"FC sparsity   : {sparsity:.2f}%")
    print_sparsity_report(model)
