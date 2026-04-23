# utils.py
# Shared helpers: data loading, sparsity loss, gate utilities.

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from config import (
    DATA_DIR, BATCH_SIZE, NUM_WORKERS,
    CIFAR_MEAN, CIFAR_STD, OUTPUT_DIR
)
from model import PrunableLinear


#  Data 

def get_data_loaders():
    """
    Returns (train_loader, test_loader) for CIFAR-10.
    Downloads the dataset on first run (~170 MB).
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])

    train_set = datasets.CIFAR10(root=DATA_DIR, train=True,  download=True, transform=transform)
    test_set  = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    return train_loader, test_loader


#  Sparsity loss 

def compute_sparsity_loss(model: torch.nn.Module) -> torch.Tensor:
    """
    L1 penalty = mean of all gate values across every PrunableLinear layer.

    Why L1 and not L2?
        L2 gradient → 0 as gate → 0, so it stalls near zero.
        L1 gradient is a constant ±lambda regardless of gate magnitude — it keeps
        pushing even very small gates all the way to exactly 0.

    Why mean and not sum?
        sum(gates) ≈ 200,000 for this model (400K gates × avg 0.5).
        Even lambda = 1e-3 produces a sparsity term of ~200, swamping the CE loss (~1.6).
        mean(gates) always sits in [0, 1], keeping lambda directly interpretable.

    Total loss = CrossEntropy + lambda × mean(all_gates)
    """
    all_gates = torch.cat([
        torch.sigmoid(m.gate_scores).view(-1)
        for m in model.modules()
        if isinstance(m, PrunableLinear)
    ])
    return all_gates.mean()


#  Gate utilities 

def collect_all_gates(model: torch.nn.Module):
    """Returns a flat numpy array of all gate values in the model."""
    gates = []
    for m in model.modules():
        if isinstance(m, PrunableLinear):
            gates.extend(m.get_gates().cpu().numpy().flatten())
    return gates


#  Filesystem 

def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
