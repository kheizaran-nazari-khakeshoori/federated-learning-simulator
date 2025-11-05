"""FedProx - Federated Optimization in Heterogeneous Networks (Li et al. 2020)

Adds proximal term to local objective to limit divergence:
    h_k(w) = F_k(w) + mu/2 * ||w - w_global||^2

Effect: slower but more stable on non-IID data.
"""
import random
import numpy as np
from .base import BaseFLAlgorithm

class FedProx(BaseFLAlgorithm):
    name = "FedProx"

    def __init__(self, mu: float = 0.1):
        self.mu = mu  # proximal strength

    def local_update(self, global_acc: float, n_epochs: int, dataset_diff: float) -> float:
        heterogeneity = random.uniform(-6, 6)
        epoch_gain = n_epochs * random.uniform(0.6, 1.1) * dataset_diff
        epoch_gain *= 0.92  # proximal regularization slows gain (Li et al.)
        # mu penalty: larger mu -> closer to global_acc
        proximal_penalty = self.mu * random.uniform(0.2, 0.8)
        local_acc = global_acc + heterogeneity + epoch_gain - proximal_penalty + random.uniform(-1.2, 1.2)
        local_acc = min(98.5, max(5, local_acc - (global_acc/100)*2))
        return local_acc

    def aggregate(self, client_accs: list[float], global_acc: float) -> float:
        # Same weighted average as FedAvg, but more stable (less noise)
        avg = sum(client_accs) / len(client_accs) if client_accs else global_acc
        momentum = 0.18  # slightly more conservative
        new_global = global_acc * momentum + avg * (1 - momentum)
        new_global += random.uniform(-0.3, 0.5)  # reduced variance
        return min(99.0, max(global_acc, new_global))

    def aggregate_weights(self, client_weights: list[np.ndarray], client_sizes: list[int] | None = None) -> np.ndarray:
        # In real PyTorch: add proximal term during local SGD, aggregation same as FedAvg
        return super().aggregate_weights(client_weights, client_sizes)
