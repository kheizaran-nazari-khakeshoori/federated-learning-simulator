"""Unit test for SimpleCNN."""
from models.cnn import SimpleCNN
from data.datasets import get_dataset
import numpy as np

def test_simplecnn_train():
    (Xtr, ytr), (Xte, yte) = get_dataset("Synthetic")
    m = SimpleCNN(input_dim=Xtr.shape[1], hidden=32, lr=0.05, seed=0)
    acc = m.train(Xtr[:500], ytr[:500], epochs=1, batch_size=32)
    assert acc >= 15  # hard synthetic still learns
    assert m.count_params() == 784*32 + 32 + 32*10 + 10
    assert "32" in m.summary()

if __name__ == "__main__":
    test_simplecnn_train()
    print("PASS test_simplecnn_train")
