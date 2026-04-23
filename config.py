# config.py
# All hyperparameters and paths live here.
# Change values here — no need to dig into train.py or model.py.

import torch

#  Device 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#  Data 
DATA_DIR        = "./data"
BATCH_SIZE      = 128
NUM_WORKERS     = 2  # set to 0 on Windows if DataLoader hangs

# CIFAR-10 channel-wise mean and std (standard values)
CIFAR_MEAN = (0.5, 0.5, 0.5)
CIFAR_STD  = (0.5, 0.5, 0.5)

#  Model 
NUM_CLASSES     = 10
FC_HIDDEN_1     = 512
FC_HIDDEN_2     = 256
DROPOUT_RATE    = 0.4

#  Training 
LEARNING_RATE   = 1e-3
NUM_EPOCHS      = 20

# Lambda values to sweep over.
# Higher lambda -> more sparsity, potentially lower accuracy.
# With mean-normalized sparsity loss, these are directly interpretable:
#   lambda = 1.0  -> mild pruning pressure
#   lambda = 5.0  -> moderate
#   lambda = 20.0 -> aggressive
LAMBDA_VALUES   = [1.0, 5.0, 20.0]

#  Evaluation 
# Gates below this threshold are counted as pruned
PRUNE_THRESHOLD = 1e-2

#  Output 
OUTPUT_DIR      = "./outputs"
PLOT_FILENAME   = "gate_dist_all.png"
