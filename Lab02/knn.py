"""
Implements a K-Nearest Neighbor classifier in PyTorch.
"""
import torch
from typing import Dict, List


def hello():
    """
    This is a sample function that we will try to import and run to ensure that
    our environment is correctly set up.
    """
    print("Hello from knn.py!")


def compute_distances_two_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)

    x_train_flat = x_train.reshape(num_train, -1)
    x_test_flat = x_test.reshape(num_test, -1)

    for i in range(num_train):
        for j in range(num_test):
            diff = x_train_flat[i] - x_test_flat[j]
            dists[i, j] = (diff * diff).sum()

    return dists


def compute_distances_one_loop(x_train: torch.Tensor, x_test: torch.Tensor):
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)

    x_train_flat = x_train.reshape(num_train, -1)
    x_test_flat = x_test.reshape(num_test, -1)

    for i in range(num_train):
        diff = x_train_flat[i] - x_test_flat
        dists[i] = (diff * diff).sum(dim=1)

    return dists


def compute_distances_no_loops(x_train: torch.Tensor, x_test: torch.Tensor):
    num_train = x_train.shape[0]
    num_test = x_test.shape[0]
    dists = x_train.new_zeros(num_train, num_test)
    x_train_flat = x_train.reshape(num_train, -1)
    x_test_flat  = x_test.reshape(num_test, -1)

    train_sq = (x_train_flat ** 2).sum(dim=1).unsqueeze(1)
    test_sq  = (x_test_flat  ** 2).sum(dim=1).unsqueeze(0)
    cross    = x_train_flat @ x_test_flat.T

    dists = train_sq + test_sq - 2 * cross
    return dists
