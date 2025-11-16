"""Unit test for Dirichlet non-IID partition."""
from data.datasets import get_dataset, partition_non_iid
import numpy as np

def test_partition_non_iid():
    (Xtr, ytr), _ = get_dataset("Synthetic")
    parts = partition_non_iid(Xtr, ytr, n_clients=5, alpha=0.1, seed=0)
    assert len(parts) == 5
    total = sum(len(p[0]) for p in parts)
    assert total == len(Xtr)
    # alpha 0.1 should be highly skewed (some client has <50 of one class)
    hist0 = np.bincount(parts[0][1], minlength=10)
    assert (hist0 == 0).sum() >= 2  # at least 2 missing labels

if __name__ == "__main__":
    test_partition_non_iid()
    print("PASS test_partition_non_iid")
