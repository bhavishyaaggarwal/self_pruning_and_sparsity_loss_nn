# train.py
# Training loop for the self-pruning network.
# Run directly:  python train.py

import logging
import torch
import torch.optim as optim
import torch.nn.functional as F

from config import DEVICE, LEARNING_RATE, NUM_EPOCHS, LAMBDA_VALUES
from model import SelfPruningNet
from evaluate import evaluate
from utils import get_data_loaders, compute_sparsity_loss, ensure_output_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def train_one_epoch(model, loader, optimizer, lambda_val, device):
    """
    One full pass over the training set.
    Returns average total loss for the epoch.
    """
    model.train()
    total_loss = 0.0

    for data, target in loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        logits       = model(data)
        ce_loss      = F.cross_entropy(logits, target)
        sparsity_loss = compute_sparsity_loss(model)

        loss = ce_loss + lambda_val * sparsity_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def run_experiment(lambda_val, train_loader, test_loader):
    """
    Trains one model for a given lambda value.
    Returns (accuracy, sparsity, trained_model).
    """
    log.info(f"{'─'*50}")
    log.info(f"Starting experiment  λ = {lambda_val}")
    log.info(f"{'─'*50}")

    model     = SelfPruningNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(1, NUM_EPOCHS + 1):
        avg_loss = train_one_epoch(model, train_loader, optimizer, lambda_val, DEVICE)
        acc, sparsity = evaluate(model, test_loader, DEVICE)

        log.info(
            f"Epoch {epoch:02d}/{NUM_EPOCHS}  "
            f"loss={avg_loss:.4f}  acc={acc:.2f}%  sparsity={sparsity:.2f}%"
        )

    log.info(f"λ={lambda_val}  →  acc={acc:.2f}%  sparsity={sparsity:.2f}%")
    return acc, sparsity, model


def main():
    ensure_output_dir()

    log.info(f"Device: {DEVICE}")
    log.info("Loading CIFAR-10 …")
    train_loader, test_loader = get_data_loaders()

    results = []
    trained_models = []

    for lam in LAMBDA_VALUES:
        acc, sparsity, model = run_experiment(lam, train_loader, test_loader)
        results.append((lam, acc, sparsity))
        trained_models.append(model)

    # Summary table
    log.info("\nFinal Results")
    log.info(f"{'Lambda':<12} {'Accuracy':>10} {'Sparsity':>10}")
    log.info("─" * 35)
    for lam, acc, sparsity in results:
        log.info(f"{lam:<12} {acc:>9.2f}%  {sparsity:>8.2f}%")

    # Save gate distribution plot
    from visualize import plot_gate_distributions
    plot_gate_distributions(results, trained_models)

    return results, trained_models


if __name__ == "__main__":
    main()
