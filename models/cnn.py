"""
Real numpy CNN/MLP for FL - replaces synthetic local_update().
- If torch available: uses torch CNN (2 conv + FC)
- Else: pure numpy MLP (784->128->10) with ReLU, softmax, SGD - no external deps beyond numpy

Both expose same API: train(X_k, y_k, epochs) -> acc, get_weights()/set_weights()
"""
import numpy as np

# ---------- Numpy MLP (offline fallback, works without torch) ----------
class SimpleCNN:
    """Numpy MLP mimicking CNN: Input -> FC128 ReLU -> FC10 logits. For 28x28=784 flat."""
    def __init__(self, input_dim=784, hidden=128, output=10, lr=0.1, seed=0):
        rng = np.random.RandomState(seed)
        # He init
        self.W1 = rng.randn(input_dim, hidden).astype(np.float16).astype(np.float32)  # float16 save * np.sqrt(2./input_dim)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = rng.randn(hidden, output).astype(np.float32) * np.sqrt(2./hidden)
        self.b2 = np.zeros(output, dtype=np.float32)
        self.lr = lr
        self.input_dim = input_dim
        self.hidden = hidden
        self.output = output

    def _relu(self, x): return np.maximum(0, x)
    def _relu_grad(self, x): return (x > 0).astype(np.float32)
    def _softmax(self, logits):
        e = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def forward(self, X):
        z1 = X @ self.W1 + self.b1  # (N, hidden)
        h = self._relu(z1)
        logits = h @ self.W2 + self.b2  # (N, 10)
        cache = (X, z1, h, logits)
        return logits, cache

    def predict(self, X):
        logits, _ = self.forward(X)
        return np.argmax(logits, axis=1)

    def evaluate(self, X, y):
        pred = self.predict(X)
        return float(np.mean(pred == y) * 100)

    def clip_grad(self, g, clip=1.0):
        n=np.linalg.norm(g)
        return g * min(1, clip/(n+1e-6))
    def train(self, X, y, epochs=1, batch_size=32, verbose=False):
        """SGD with L2 decay 1e-4 to prevent 100% overfit on synthetic."""
        n = X.shape[0]
        wd = 1e-4
        for ep in range(epochs):
            idx = np.random.permutation(n)
            X_shuf, y_shuf = X[idx], y[idx]
            for i in range(0, n, batch_size):
                xb = X_shuf[i:i+batch_size]
                yb = y_shuf[i:i+batch_size]
                logits, cache = self.forward(xb)
                Xb, z1, h, logits = cache
                probs = self._softmax(logits)
                y_onehot = np.zeros_like(probs)
                y_onehot[np.arange(len(yb)), yb] = 1
                dlogits = (probs - y_onehot) / len(yb)
                dW2 = h.T @ dlogits + wd * self.W2
                db2 = dlogits.sum(axis=0)
                dh = dlogits @ self.W2.T
                dz1 = dh * self._relu_grad(z1)
                dW1 = Xb.T @ dz1 + wd * self.W1
                db1 = dz1.sum(axis=0)
                dW2 += np.random.randn(*dW2.shape)*0.01  # gaussian noise
                if np.isnan(dW2).any(): self.lr *= 0.5
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
        return self.evaluate(X, y)

    def summary(self) -> str:
        return f"SimpleCNN {self.input_dim}->{self.hidden}->{self.output} ({self.count_params()} params)"

    def count_params(self) -> int:
        """Return total trainable params for logging."""
        return int(self.W1.size + self.b1.size + self.W2.size + self.b2.size)

    def get_weights(self):
        # flat vector for FedAvg averaging
        return np.concatenate([self.W1.ravel(), self.b1, self.W2.ravel(), self.b2])

    def set_weights(self, vec):
        # vec must match shape
        s1 = self.input_dim * self.hidden
        s2 = self.hidden
        s3 = self.hidden * self.output
        self.W1 = vec[0:s1].reshape(self.input_dim, self.hidden)
        self.b1 = vec[s1:s1+s2]
        self.W2 = vec[s1+s2:s1+s2+s3].reshape(self.hidden, self.output)
        self.b2 = vec[s1+s2+s3:]

    def copy(self):
        m = SimpleCNN(self.input_dim, self.hidden, self.output, self.lr)
        m.set_weights(self.get_weights().copy())
        return m

# ---------- Torch CNN (if available) ----------
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class TorchCNN(nn.Module):
        def __init__(self, input_dim=784, hidden=128, output=10, lr=0.01):
            super().__init__()
            # For MNIST 28x28, use CNN if input_dim==784, else MLP
            if input_dim == 784:
                self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
                self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
                self.fc1 = nn.Linear(32*7*7, hidden)
                self.fc2 = nn.Linear(hidden, output)
                self.use_conv = True
            else:
                self.fc1 = nn.Linear(input_dim, hidden)
                self.fc2 = nn.Linear(hidden, output)
                self.use_conv = False
            self.lr = lr
            self.optim = torch.optim.SGD(self.parameters(), lr=lr)

        def forward(self, x):
            if self.use_conv:
                # x (N,784) -> (N,1,28,28)
                x = x.view(-1, 1, 28, 28)
                x = F.relu(self.conv1(x))
                x = F.max_pool2d(x, 2)  # 14x14
                x = F.relu(self.conv2(x))
                x = F.max_pool2d(x, 2)  # 7x7
                x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

        def train_model(self, X, y, epochs=1, batch_size=32):
            self.train()
            X_t = torch.from_numpy(X).float()
            y_t = torch.from_numpy(y).long()
            ds = torch.utils.data.TensorDataset(X_t, y_t)
            dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
            for _ in range(epochs):
                for xb, yb in dl:
                    self.optim.zero_grad()
                    out = self.forward(xb)
                    loss = F.cross_entropy(out, yb)
                    loss.backward()
                    self.optim.step()
            return self.evaluate(X, y)

        def evaluate(self, X, y):
            self.eval()
            with torch.no_grad():
                X_t = torch.from_numpy(X).float()
                out = self.forward(X_t)
                pred = out.argmax(dim=1).numpy()
                return float((pred == y).mean() * 100)

        def get_weights(self):
            return np.concatenate([p.detach().cpu().numpy().ravel() for p in self.parameters()])

        def set_weights(self, vec):
            pointer = 0
            for p in self.parameters():
                num = p.numel()
                arr = vec[pointer:pointer+num].reshape(p.shape)
                p.data.copy_(torch.from_numpy(arr))
                pointer += num

    def get_model(input_dim=784, output_dim=10, use_torch=True, **kwargs):
        if use_torch:
            return TorchCNN(input_dim=input_dim, **kwargs)
        return SimpleCNN(input_dim=input_dim, **kwargs)

except ImportError:
    def get_model(input_dim=784, output_dim=10, use_torch=False, **kwargs):
        return SimpleCNN(input_dim=input_dim, **kwargs)

if __name__ == "__main__":
    # quick test with synthetic data
    from data.datasets import get_dataset, partition_non_iid
    (Xtr, ytr), (Xte, yte) = get_dataset("Synthetic")
    parts = partition_non_iid(Xtr, ytr, 2, alpha=0.5)
    m = SimpleCNN(input_dim=Xtr.shape[1])
    print("init acc", m.evaluate(Xte, yte))
    acc = m.train(parts[0][0], parts[0][1], epochs=3)
    print("after 3 epochs on client0", acc, m.evaluate(Xte, yte))

# tabular mlp branch for non-image data
def tabular_mlp(input_dim, hidden=64):
    return SimpleCNN(input_dim=input_dim, hidden=hidden)

def confusion_matrix(y_true, y_pred, n_classes=10):
    import numpy as np
    m=np.zeros((n_classes,n_classes),dtype=int)
    for t,p in zip(y_true,y_pred): m[t,p]+=1
    return m
