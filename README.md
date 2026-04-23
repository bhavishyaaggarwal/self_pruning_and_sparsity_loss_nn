# Self-Pruning Neural Network on CIFAR-10

A neural network that learns to prune its own weights during training — not as a post-processing step, but as a core part of the learning process.
- By:- Bhavishya Aggarwal
---

## Why this matters

Deploying neural networks on real hardware is often constrained by memory and latency budgets. The standard workflow is to train first, then prune afterward. But post-hoc pruning has a known problem: you're removing weights the model already depended on, which requires careful fine-tuning to recover lost accuracy.

This project explores a different approach: let the network decide during training which weights it needs. Each weight has a paired learnable gate. The network is incentivized to push unnecessary gates toward zero. By the end of training, you have both the classification accuracy and a sparse model — at the same time.

---

## How it works

### The gate mechanism

Every weight in the fully connected layers is multiplied by a gate before it's used:

```
gates         = sigmoid(gate_scores)    # squashed to (0, 1)
pruned_weight = weight × gates
output        = pruned_weight @ x + bias
```

`gate_scores` is a learnable parameter with the same shape as `weight`. The sigmoid keeps gates strictly between 0 and 1. When a gate is near 0, it effectively disconnects that weight from the network.

Because sigmoid is differentiable everywhere, gradients flow back to `gate_scores` automatically through PyTorch's autograd. No custom backward pass needed.

### The sparsity loss

Cross-entropy alone gives the network no reason to close any gates. We add an L1 penalty on top:

```
Total Loss = CrossEntropy + lambda × mean(all gate values)
```

**Why L1?** The L2 gradient shrinks as a gate approaches zero, so pruning stalls. L1 has a constant gradient — it keeps pushing even very small gates all the way to exactly 0.

**Why mean, not sum?** With ~400K FC gates initialized near 0.5, `sum(gates) ≈ 200,000`. Even `lambda = 0.001` produces a sparsity term of ~200, which completely dominates the CE loss (~1.6). Using `mean(gates)` keeps the term in `[0, 1]` regardless of model size, making lambda directly interpretable as *"how much do I weight sparsity vs accuracy?"*

### Architecture

A plain MLP on CIFAR-10 tops out around 44–45% accuracy — not enough resolution for meaningful gate decisions. Adding a CNN backbone (3 × VGG-style conv blocks) pushes accuracy to ~80–83%, leaving the prunable FC head to focus on what it does well.

```
Conv → Conv → BN → ReLU → MaxPool   (×3 blocks)
       ↓
PrunableLinear(2048 → 512) → Dropout → ReLU
PrunableLinear(512  → 256) → Dropout → ReLU
PrunableLinear(256  → 10)
```

Pruning only applies to the FC layers. Conv filters are spatially shared — each one covers the whole image, so every filter is more critical per-parameter than a FC weight.

---

## Results

Trained for 20 epochs on CIFAR-10 with Adam (lr = 1e-3), batch size 128.

| Lambda | Test Accuracy | Sparsity (FC) |
|--------|:------------:|:-------------:|
| 1.0    | 80.82%       | 14.74%        |
| 5.0    | 82.26%       | 25.33%        |
| 20.0   | 82.75%       | 29.98%        |

The accuracy-sparsity relationship looks inverted here, which is counterintuitive. It happens because the CNN backbone is doing most of the classification work. The FC gates cover only a fraction of total parameters, so increasing lambda prunes connections that weren't contributing much anyway — and may even act as mild regularization, slightly improving accuracy.

A lambda around 5 is a reasonable default: ~25% of FC weights removed with no accuracy cost.

---

## Engineering insights

**Gradient flow through gates** — The gate mechanism is fully differentiable. Backprop through `weight * sigmoid(gate_score)` gives:
- `L/weight` = `gate × upstream_grad` (gate scales the weight gradient)
- `L/gate_score` = `weight × sigmoid' × upstream_grad` (classic chain rule through sigmoid)

No extra code needed. Standard `loss.backward()` handles both.

**Sparsity normalization matters more than it looks** — Switching the sparsity loss from `sum` to `mean` was the single change that made lambda values interpretable and training stable. With `sum`, useful lambda values were in the `1e-6` range and hard to reason about. With `mean`, lambda = 1, 5, 20 produce clearly different pruning behaviors that match intuition.

**Why not prune conv layers?** Conv filters are spatially shared, so each filter sees every spatial location in the input. Removing one filter removes that feature detector globally. Unstructured pruning of individual conv weights is theoretically possible but gives no inference speedup without specialized sparse matmul support. FC layers, by contrast, are just matrix multiplications — structured pruning (whole neurons) of FC layers can actually speed up inference.

---

## Limitations

- Sparsity only applies to the FC head, not the conv backbone (~90% of parameters are unpruned).
- Unstructured gate pruning doesn't give real inference speedup without sparse hardware support. Structured pruning (removing entire neurons) would be needed for that.
- 20 epochs isn't fully converged. Sparsity tends to accelerate in later epochs — more training time would likely push sparsity higher without hurting accuracy.
- No warmup schedule for lambda. Starting with a lower lambda and ramping it up might produce a better sparsity-accuracy frontier.

---

## Future directions

- **Structured pruning**: extend the gate mechanism to operate per-neuron or per-channel instead of per-weight. This would allow physically removing rows/columns from weight matrices and achieve real speedup.
- **Extend to conv layers**: group gates by filter and apply an L1 penalty per filter rather than per weight.
- **Lambda warmup**: start lambda at 0 and linearly increase it over the first few epochs to give the model time to form good representations before pruning pressure kicks in.
- **Checkpoint & reload**: save trained models so evaluation doesn't require retraining. Currently all state is in-memory.

---

## Project structure

```
self_pruning_nn/
├── model.py          PrunableLinear layer + SelfPruningNet architecture
├── train.py          Training loop, lambda sweep, results table
├── evaluate.py       Test accuracy + sparsity calculation, per-layer report
├── visualize.py      Gate distribution histograms
├── utils.py          Data loading, sparsity loss, shared helpers
├── config.py         All hyperparameters in one place
├── app.py            FastAPI wrapper (health check, /train, /metrics)
├── requirements.txt
└── outputs/
    └── gate_dist_all.png
```

---

## Setup and usage

**Install dependencies**

```bash
pip install -r requirements.txt
```

**Run training** (downloads CIFAR-10 on first run, ~170 MB)

```bash
python train.py
```

This trains three models (lambda = 1.0, 5.0, 20.0), prints a results table, and saves the gate distribution plot to `outputs/gate_dist_all.png`.

**Evaluate a specific checkpoint**

```bash
python evaluate.py path/to/checkpoint.pt
```

**Start the API**

```bash
uvicorn app:app --reload
```

Then:
- `GET  http://localhost:8000/`        → health check
- `POST http://localhost:8000/train`   → start training (blocking)
- `GET  http://localhost:8000/metrics` → latest accuracy + sparsity

**Change lambda values**

Edit `config.py`:

```python
LAMBDA_VALUES = [1.0, 5.0, 20.0]   # ← modify this list
```

---

## What the gate plot shows

A successful run shows a bimodal distribution: a large spike near 0 (pruned connections) and a smaller cluster of values above 0.1 (active connections). The red dashed line marks the 0.01 prune threshold.

As lambda increases, the left spike grows — more connections are being removed.

---

Developed as a case study for the Tredence AI Engineering Internship, 2025.  
Tested on Google Colab T4 GPU. One lambda config trains in ~5–6 minutes.
