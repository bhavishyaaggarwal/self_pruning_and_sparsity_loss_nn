TO get full FASTAPI vesion to train the model look in second branck

# Self-Pruning Neural Network on CIFAR-10

This project implements a neural network that learns to prune its own weights during training. Instead of pruning after the fact, the network has learnable gates attached to each weight that gradually close off unnecessary connections on their own.

---

## What this is

The core idea is pretty simple. Every weight in the fully connected layers has a paired "gate" value. During the forward pass, the effective weight is:

    effective_weight = weight * sigmoid(gate_score)

Since sigmoid outputs are between 0 and 1, a gate near 0 basically turns off that connection. To actually push gates toward 0, an L1 penalty is added on top of the classification loss:

    Total Loss = CrossEntropy + lambda * mean(gate values)

The mean is important. Early on I was using sum of all gates, which caused the sparsity term to completely dominate the loss and made training unstable. Normalizing by the number of gates fixed this.

---

## How the project evolved

I did not get to the final version in one shot. Here is roughly what happened:

**Version 1 — plain MLP**

Started with a simple fully connected network where all three layers were PrunableLinear. Accuracy came out around 44-45% on CIFAR-10. Sparsity was technically non-zero but only barely (under 0.1%). The model was learning to classify but not pruning in any meaningful way.

The main issue was that a flat MLP is just not a good fit for image data. It has no way to capture spatial patterns, so most of its capacity goes toward just trying to remember pixel correlations. Not much was left over for the gates to make clean on/off decisions.

**Version 2 — adding a CNN backbone**

Added three VGG-style convolutional blocks before the FC head. Each block is Conv → Conv → BatchNorm → ReLU → MaxPool. This pushed accuracy up to around 81-82%.

The PrunableLinear layers stayed the same, just now sitting after the CNN feature extractor instead of directly on the raw pixels. Sparsity improved too, getting to around 16-28% depending on lambda.

One thing I noticed here: the sparsity-accuracy relationship looked inverted at first. Higher lambda seemed to give slightly higher accuracy, which is the opposite of what you'd expect. This happens because the CNN is doing most of the classification work. The FC gates are only covering a fraction of the total parameters, so changing lambda does not hurt accuracy much — it just prunes connections that were not doing much anyway.

**Version 3 — fixing the sparsity loss**

Switched the sparsity loss from sum to mean. This kept the regularization term in a reasonable range regardless of model size, which made lambda values actually interpretable. Also added dropout (0.4) to the FC layers and widened the lambda range.

Final results with lambda = 1.0, 5.0, 20.0:

| Lambda | Test Accuracy | Sparsity |
|--------|--------------|----------|
| 1.0    | 80.82%       | 14.74%   |
| 5.0    | 82.26%       | 25.33%   |
| 20.0   | 82.75%       | 29.98%   |

A moderate lambda around 5 seems to be the sweet spot — decent sparsity without losing accuracy.

---

## File structure

    self_pruning_nn.ipynb   main notebook with all three versions
    outputs/
        gate_dist_all.png   gate value histograms for each lambda

---

## How to run

Install dependencies:

    pip install torch torchvision matplotlib numpy

Open the notebook and run the cells in order. CIFAR-10 will download automatically the first time (~170MB). The final cell saves the gate distribution plot to the outputs folder.

If you want to try different lambda values, the list is near the bottom of the last code cell:

    lambda_values = [1.0, 5.0, 20.0]

---

## What the gate distribution plot shows

After training, most gates end up near 0 (pruned) or somewhere above 0.1 (active). A good run shows a spike near zero and a separate cluster of values that stayed open. The red dashed line at 0.01 is the threshold used to count a gate as pruned.

As lambda increases, the spike near zero gets bigger, which means more connections are being removed.

---

## Things that could be improved

- Apply pruning to the conv layers as well, not just FC. Right now only the classifier head is sparse.
- Try a structured pruning approach where entire neurons or channels are removed, not just individual weights. That would give actual speedup at inference time.
- Train for more epochs. 20 epochs is enough to see the behavior but accuracy is probably not fully saturated.
- The sparsity grows slowly in early epochs and accelerates later. A warmup period where lambda starts low and increases gradually might give better results.

---

Ran on Google Colab with a T4 GPU. Training one lambda config takes about 5-6 minutes.
