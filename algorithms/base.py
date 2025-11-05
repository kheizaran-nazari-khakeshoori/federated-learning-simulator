"""Base class for FL algorithms."""
import numpy as np
import random

class BaseFLAlgorithm:
    """Shared logic for synthetic simulation + real numpy model."""
    name = "Base"

    def local_update(self, global_acc: float, n_epochs: int, dataset_diff: float) -> float:
        """Synthetic client accuracy simulation - override for real model."""
        raise NotImplementedError

    def aggregate(self, client_accs: list[float], global_acc: float) -> float:
        """Server aggregation: w_global = f(w_clients) -> new global accuracy."""
        raise NotImplementedError

    # helper for dummy model weights (numpy) - simulates FedAvg weight averaging
    def aggregate_weights(self, client_weights: list[np.ndarray], client_sizes: list[int] | None = None) -> np.ndarray:
        """Weighted average of numpy weight vectors: sum(n_k/n * w_k)"""
        if client_sizes is None:
            client_sizes = [1] * len(client_weights)
        total = sum(client_sizes)
        avg = np.zeros_like(client_weights[0], dtype=float)
        for w, n in zip(client_weights, client_sizes):
            avg += w * (n / total)
        return avg
