"""Unit test for FedAvg/Prox/Adam."""
from algorithms import get_algorithm
import numpy as np

def test_algorithms_aggregate():
    w1 = np.array([1., 2., 3.])
    w2 = np.array([3., 2., 1.])
    for name in ["FedAvg", "FedProx", "FedAdam"]:
        algo = get_algorithm(name)
        avg = algo.aggregate_weights([w1, w2], [1, 1])
        assert np.allclose(avg, [2., 2., 2.]), name
        acc = algo.aggregate([80., 90.], 70.)
        assert 70 < acc < 95, f"{name} {acc}"

if __name__ == "__main__":
    test_algorithms_aggregate()
    print("PASS test_algorithms_aggregate")
