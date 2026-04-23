# model.py
# Defines the PrunableLinear layer and the full CNN + prunable FC network.

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import FC_HIDDEN_1, FC_HIDDEN_2, NUM_CLASSES, DROPOUT_RATE


class PrunableLinear(nn.Module):
    """
    A linear layer where every weight has a paired learnable gate.

    How it works:
        gates        = sigmoid(gate_scores)   # squashed to (0, 1)
        pruned_w     = weight * gates         # gates mask weak connections
        output       = F.linear(x, pruned_w, bias)

    When a gate approaches 0, it multiplies its weight by ~0, effectively
    removing that connection. The sigmoid is smooth and differentiable, so
    gradients flow back to gate_scores automatically through PyTorch autograd.
    No custom backward() is needed.

    Init note: gate_scores start from randn. This means some gates begin near
    0 (sigmoid ≈ 0.5) and some near the extremes. The asymmetry is intentional
    — it gives the L1 penalty real variation to work with from epoch 1.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()

        self.in_features  = in_features
        self.out_features = out_features

        # Standard weight and bias, same as nn.Linear
        self.weight      = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias        = nn.Parameter(torch.zeros(out_features))

        # One gate score per weight. Same shape as weight.
        # These are registered as parameters so the optimizer updates them.
        self.gate_scores = nn.Parameter(torch.randn(out_features, in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates         = torch.sigmoid(self.gate_scores)   # (out, in), values in (0, 1)
        pruned_weights = self.weight * gates               # element-wise mask
        return F.linear(x, pruned_weights, self.bias)

    def get_gates(self) -> torch.Tensor:
        """Returns the current gate values (detached, for inspection)."""
        return torch.sigmoid(self.gate_scores).detach()


class SelfPruningNet(nn.Module):
    """
    Hybrid architecture: VGG-style CNN backbone + prunable FC classifier.

    Design rationale:
    - Conv layers capture spatial structure in images. They share parameters
      across positions, so each filter is already compact and critical — not
      a good candidate for unstructured pruning.
    - FC layers are dense and over-parameterized. They're where most redundancy
      lives, making them the natural place to apply gate-based pruning.

    Structure:
        3 × (Conv → Conv → BN → ReLU → MaxPool)   ← feature extractor
        PrunableLinear(2048 → 512) → Dropout → ReLU
        PrunableLinear(512  → 256) → Dropout → ReLU
        PrunableLinear(256  → 10)                  ← classifier head
    """

    def __init__(self):
        super().__init__()

        #  Feature extractor (standard conv, not pruned) 
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 32 × 16 × 16
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 64 × 8 × 8
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),   # 128 × 4 × 4  →  flat = 2048
        )

        #  Prunable classifier head 
        self.fc1     = PrunableLinear(128 * 4 * 4, FC_HIDDEN_1)
        self.fc2     = PrunableLinear(FC_HIDDEN_1,  FC_HIDDEN_2)
        self.fc3     = PrunableLinear(FC_HIDDEN_2,  NUM_CLASSES)
        self.dropout = nn.Dropout(DROPOUT_RATE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = x.view(x.size(0), -1)   # flatten

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)

        return x

    def prunable_layers(self):
        """Yields all PrunableLinear modules in the network."""
        for m in self.modules():
            if isinstance(m, PrunableLinear):
                yield m
