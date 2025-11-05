"""FedAvg - Communication-Efficient Learning of Deep Networks from Decentralized Data (McMahan et al. 2017)

Model: each client does local SGD for E epochs, server does weighted average:
    w_{t+1} = sum_k (n_k / n) * w_k^{t+1}
"""
import random
import numpy as np
from .base import BaseFLAlgorithm

class FedAvg(BaseFLAlgorithm):
    name = "FedAvg"

    def local_update(self, global_acc: float, n_epochs: int, dataset_diff: float) -> float:
        # Synthetic accuracy model: global + heterogeneity + epoch gain
        heterogeneity = random.uniform(-6, 6)          # non-IID variance
        epoch_gain = n_epochs * random.uniform(0.6, 1.1) * dataset_diff
        local_acc = global_acc + heterogeneity + epoch_gain + random.uniform(-1.5, 1.5)
        # diminishing returns
        local_acc = min(98.5, max(5, local_acc - (global_acc/100)*2))
        return local_acc

    def aggregate(self, client_accs: list[float], global_acc: float) -> float:
        # FedAvg: simple average blended with previous global (momentum 0.15)
        avg = sum(client_accs) / len(client_accs) if client_accs else global_acc
        momentum = 0.15
        new_global = global_acc * momentum + avg * (1 - momentum)
        new_global += random.uniform(-0.4, 0.8)  # network noise
        return min(99.0, max(global_acc, new_global))

    # Real weight example (numpy) for future torch replacement
    def aggregate_weights(self, client_weights: list[np.ndarray], client_sizes: list[int] | None = None) -> np.ndarray:
        return super().aggregate_weights(client_weights, client_sizes)
