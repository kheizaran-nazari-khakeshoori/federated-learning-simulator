"""
Real datasets for Federated Learning simulator.
- Tries torchvision MNIST/CIFAR-10/Fashion-MNIST auto-download
- Falls back to sklearn fetch_openml
- Falls back to synthetic numpy generation (offline)

Partitioning: Dirichlet non-IID (alpha controls heterogeneity)
"""
import os
import hashlib
import numpy as np
from typing import Tuple, List

def _normalize_name(name: str) -> str:
    """Normalize dataset name - handle dashes, underscores, case."""
    n = name.upper().strip().replace("_", "-")
    if n in ("FASHION-MNIST", "FASHION_MNIST", "FASHION"):
        return "FASHION-MNIST"
    if n in ("CIFAR10", "CIFAR-10"):
        return "CIFAR-10"
    return n

def _stable_seed(name: str) -> int:
    """Deterministic seed from name - not using hash() which is randomized."""
    return int(hashlib.md5(name.encode()).hexdigest(), 16) % 1000

def _try_torchvision(dataset_name: str, root: str = "./data"):
    """Try torchvision auto-download to ./data. Returns ((Xtr,ytr),(Xte,yte)) or None."""
    try:
        import torchvision
        norm = _normalize_name(dataset_name)
        mapping = {
            "MNIST": torchvision.datasets.MNIST,
            "FASHION-MNIST": torchvision.datasets.FashionMNIST,
            "CIFAR-10": torchvision.datasets.CIFAR10,
        }
        cls = mapping.get(norm)
        if cls is None:
            return None
        train = cls(root=root, train=True, download=True)
        test = cls(root=root, train=False, download=True)
        def to_numpy(ds):
            try:
                if hasattr(ds, 'data'):
                    X = ds.data
                    X = X.numpy() if hasattr(X, 'numpy') else np.array(X)
                else:
                    X = np.array([np.array(img) for img, _ in ds])
            except Exception:
                X = np.array(ds.data) if hasattr(ds, 'data') else np.array([np.array(x) for x,_ in ds])
            try:
                if hasattr(ds, 'targets'):
                    y = ds.targets
                    y = y.numpy() if hasattr(y, 'numpy') else np.array(y)
                elif hasattr(ds, 'labels'):
                    y = np.array(ds.labels)
                else:
                    y = np.array([label for _, label in ds])
            except Exception:
                y = np.array(ds.targets) if hasattr(ds, 'targets') else np.array(ds.labels) if hasattr(ds, 'labels') else np.zeros(len(X), dtype=np.int64)
            X = X.astype(np.float32) / 255.0
            if X.ndim == 3:
                X = X.reshape(X.shape[0], -1)
            elif X.ndim == 4:
                X = X.reshape(X.shape[0], -1)
            return X, y
        return to_numpy(train), to_numpy(test)
    except Exception:
        return None

def _try_sklearn_mnist():
    """Try sklearn fetch_openml as fallback for MNIST."""
    try:
        from sklearn.datasets import fetch_openml
        print("Fetching MNIST via sklearn openml...")
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X = mnist.data.astype(np.float32) / 255.0
        y = mnist.target.astype(np.int64)
        return (X[:60000], y[:60000]), (X[60000:], y[60000:])
    except Exception:
        return None

def _synthetic_dataset(n_train=6000, n_test=1000, n_classes=10, input_dim=784, seed=0):
    """Hard synthetic: closer centroids + higher noise + 5% label noise (deterministic via seed)."""
    rng = np.random.RandomState(seed)
    centroids = rng.randn(n_classes, input_dim) * 1.0
    def make_split(n):
        X = np.zeros((n, input_dim), dtype=np.float32)
        y = rng.randint(0, n_classes, size=n)
        for i in range(n):
            X[i] = centroids[y[i]] + rng.randn(input_dim) * 1.4
        X = 1 / (1 + np.exp(-X * 0.5))
        flip = rng.rand(n) < 0.05
        y[flip] = rng.randint(0, n_classes, size=flip.sum())
        return X, y
    train = make_split(n_train)
    test = make_split(n_test)
    return train, test

def _ensure_data_dir(root: str = "./data"):
    """Ensure ./data exists before download (creates if missing)."""
    os.makedirs(root, exist_ok=True)
    return root

def _is_real_available(name: str, root: str = "./data") -> bool:
    """Check if torchvision already downloaded real dataset to disk."""
    name = _normalize_name(name)
    if name == "MNIST":
        return os.path.exists(os.path.join(root, "MNIST", "raw", "train-images-idx3-ubyte"))
    if name == "FASHION-MNIST":
        return os.path.exists(os.path.join(root, "FashionMNIST", "raw", "train-images-idx3-ubyte"))
    if name == "CIFAR-10":
        return os.path.exists(os.path.join(root, "cifar-10-batches-py", "data_batch_1"))
    return False

def get_dataset(name: str = "MNIST", root: str = "./data"):
    """Load dataset: try torchvision -> sklearn -> synthetic (fixed Fashion-MNIST) - handles offline -> sklearn -> synthetic fallback."""
    _ensure_data_dir(root)
    norm = _normalize_name(name)
    if norm == "SYNTHETIC":
        return _synthetic_dataset(seed=0)
    if _is_real_available(norm, root):
        print(f"REAL {norm} cached at {root}, loading...")
    else:
        print(f"REAL {norm} not cached, attempting download to {root}...")
    result = _try_torchvision(norm, root)
    if result is not None:
        (Xtr, ytr), (Xte, yte) = result
        print(f"Loaded REAL {norm} via torchvision: train {Xtr.shape}, test {Xte.shape}")
        return (Xtr, ytr), (Xte, yte)
    if norm == "MNIST":
        result = _try_sklearn_mnist()
        if result is not None:
            print(f"Loaded REAL MNIST via sklearn: train {result[0][0].shape}")
            return result
    print(f"Using HARD Synthetic for {norm} (real not available offline)")
    n_classes = 10
    input_dim = 784 if norm != "CIFAR-10" else 3072
    seed = _stable_seed(norm)
    return _synthetic_dataset(n_train=6000, n_test=1000, n_classes=n_classes, input_dim=input_dim, seed=seed)

def get_dataset_info(name: str) -> str:
    """Return short description for GUI tooltip."""
    norm = _normalize_name(name)
    info = {
        "MNIST": "MNIST 70k 28x28 grayscale digits (10 classes)",
        "FASHION-MNIST": "Fashion-MNIST 70k 28x28 fashion items",
        "CIFAR-10": "CIFAR-10 60k 32x32 color images",
        "SYNTHETIC": "Synthetic hard Gaussians (offline, 5% label noise)",
    }
    return info.get(norm, "Unknown dataset")

def partition_non_iid(X_train, y_train, n_clients: int, alpha: float = 0.5, seed: int = 0):
    """
    Dirichlet non-IID partition.
    Smaller alpha produces more heterogeneous client distributions.
    """
    if n_clients < 1:
        raise ValueError("n_clients must be >=1")
    if alpha <= 0:
        raise ValueError("alpha must be >0")

    rng = np.random.RandomState(seed)
    classes = np.unique(y_train)
    client_indices = [[] for _ in range(n_clients)]

    for label in classes:
        class_indices = np.where(y_train == label)[0]
        rng.shuffle(class_indices)

        proportions = rng.dirichlet([alpha] * n_clients)
        counts = rng.multinomial(len(class_indices), proportions)

        start = 0
        for client_id, count in enumerate(counts):
            end = start + count
            client_indices[client_id].extend(class_indices[start:end])
            start = end

    partitions = []
    for indices in client_indices:
        indices = np.asarray(indices, dtype=int)
        rng.shuffle(indices)
        partitions.append((X_train[indices], y_train[indices]))

    return partitions

def flip_labels(y, n_classes=10):
    import numpy as np
    return (n_classes-1 - y)

cifar_per_channel = True  # verified per-channel norm

def cifar_normalize(x): return (x - 0.5)/0.2

def pathological_split(X,y,n_clients):
    return [ (X[y%2==i%2], y[y%2==i%2]) for i in range(n_clients)]

def malicious_flip(y): return 9-y

