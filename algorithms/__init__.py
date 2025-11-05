"""Federated Learning algorithms - import all for GUI."""
from .fedavg import FedAvg
from .fedprox import FedProx
from .fedadam import FedAdam

__all__ = ["FedAvg", "FedProx", "FedAdam"]

def get_algorithm(name: str):
    name = name.lower()
    if name == "fedavg":
        return FedAvg()
    elif name == "fedprox":
        return FedProx(mu=0.1)
    elif name == "fedadam":
        return FedAdam()
    else:
        raise ValueError(f"Unknown algorithm: {name}")
