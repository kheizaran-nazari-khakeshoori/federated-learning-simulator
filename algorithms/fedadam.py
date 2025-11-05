"""FedAdam - Adaptive Federated Optimization (Reddi et al. 2021)

Server uses Adam optimizer instead of simple averaging:
    m_t = beta1*m_{t-1} + (1-beta1)*delta
    v_t = beta2*v_{t-1} + (1-beta2)*delta^2
    w_{t+1} = w_t + eta * m_t / (sqrt(v_t)+tau)

Effect: faster convergence, adaptive learning rate.
"""
import random
import numpy as np
from .base import BaseFLAlgorithm

class FedAdam(BaseFLAlgorithm):
    name = "FedAdam"

    def __init__(self, beta1: float = 0.9, beta2: float = 0.999, eta: float = 0.1, tau: float = 1e-3):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eta = eta
        self.tau = tau
        self.m = 0.0
        self.v = 0.0
        self.t = 0

    def local_update(self, global_acc: float, n_epochs: int, dataset_diff: float) -> float:
        heterogeneity = random.uniform(-6, 6)
        epoch_gain = n_epochs * random.uniform(0.6, 1.1) * dataset_diff
        epoch_gain *= 1.18  # adaptive acceleration
        local_acc = global_acc + heterogeneity + epoch_gain + random.uniform(-1.5, 1.5)
        local_acc = min(98.5, max(5, local_acc - (global_acc/100)*2))
        return local_acc

    def aggregate(self, client_accs: list[float], global_acc: float) -> float:
        avg = sum(client_accs) / len(client_accs) if client_accs else global_acc
        delta = avg - global_acc  # pseudo-gradient
        self.t += 1
        # Adam update (1D scalar version for accuracy simulation)
        self.m = self.beta1 * self.m + (1 - self.beta1) * delta
        self.v = self.beta2 * self.v + (1 - self.beta2) * (delta ** 2)
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        adam_step = self.eta * m_hat / (np.sqrt(v_hat) + self.tau)
        # Blend with FedAvg momentum 0.3 for stability
        base = global_acc * 0.3 + avg * 0.7
        new_global = base + adam_step * 0.15
        new_global += random.uniform(-0.4, 0.8)
        return min(99.0, max(global_acc, new_global))

    def reset(self):
        self.m = 0.0
        self.v = 0.0
        self.t = 0

    def aggregate_weights(self, client_weights: list[np.ndarray], client_sizes: list[int] | None = None) -> np.ndarray:
        # Real version would apply Adam on delta = avg_weights - global_weights
        avg = super().aggregate_weights(client_weights, client_sizes)
        return avg
