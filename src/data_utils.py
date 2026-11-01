"""
Veri yükleme ve ön işleme modülü
--------------------------------
Fashion-MNIST veri setini (IDX .gz formatı) okur, normalize eder,
eğitim/doğrulama olarak böler ve PyTorch DataLoader nesneleri oluşturur.
Veri dosyaları 'data/' klasöründe yoksa resmi depodan indirilir.
"""
import gzip
import os
import urllib.request

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

CLASS_NAMES = [
    "T-shirt/üst", "Pantolon", "Kazak", "Elbise", "Ceket",
    "Sandalet", "Gömlek", "Spor ayakkabı", "Çanta", "Bot",
]

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}
BASE_URL = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"

# Eğitim setinin piksel ortalaması ve standart sapması ([0,1] ölçeğinde)
MEAN, STD = 0.2860, 0.3530


def _ensure_files(data_dir):
    os.makedirs(data_dir, exist_ok=True)
    for fname in FILES.values():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            print(f"[indiriliyor] {fname}")
            urllib.request.urlretrieve(BASE_URL + fname, path)


def _read_images(path):
    with gzip.open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    return data.reshape(-1, 28, 28)


def _read_labels(path):
    with gzip.open(path, "rb") as f:
        return np.frombuffer(f.read(), dtype=np.uint8, offset=8).astype(np.int64)


def load_fashion_mnist(data_dir="data"):
    """Ham veriyi numpy dizileri olarak döndürür (uint8 görüntüler, int64 etiketler)."""
    _ensure_files(data_dir)
    x_train = _read_images(os.path.join(data_dir, FILES["train_images"]))
    y_train = _read_labels(os.path.join(data_dir, FILES["train_labels"]))
    x_test = _read_images(os.path.join(data_dir, FILES["test_images"]))
    y_test = _read_labels(os.path.join(data_dir, FILES["test_labels"]))
    return x_train, y_train, x_test, y_test


def to_tensor(x):
    """uint8 (N,28,28) -> float32 (N,1,28,28), [0,1] aralığına ölçekleme + standartlaştırma."""
    x = torch.from_numpy(x.astype(np.float32) / 255.0).unsqueeze(1)
    return (x - MEAN) / STD


def augment_batch(x):
    """Basit veri artırma (yalnızca eğitimde):
    - %50 olasılıkla yatay çevirme
    - 2 piksel dolgu + rastgele 28x28 kırpma (±2 piksel kaydırma)
    """
    n = x.size(0)
    flip = torch.rand(n) < 0.5
    x = x.clone()
    x[flip] = torch.flip(x[flip], dims=[3])
    pad_val = (0.0 - MEAN) / STD  # siyah arka plan değeri
    padded = torch.nn.functional.pad(x, (2, 2, 2, 2), value=pad_val)
    out = torch.empty_like(x)
    dx = torch.randint(0, 5, (n,))
    dy = torch.randint(0, 5, (n,))
    for i in range(n):
        out[i] = padded[i, :, dy[i]:dy[i] + 28, dx[i]:dx[i] + 28]
    return out


def get_loaders(data_dir="data", batch_size=64, val_size=6000, seed=42):
    x_train, y_train, x_test, y_test = load_fashion_mnist(data_dir)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x_train))
    val_idx, tr_idx = idx[:val_size], idx[val_size:]

    xtr, ytr = to_tensor(x_train[tr_idx]), torch.from_numpy(y_train[tr_idx])
    xva, yva = to_tensor(x_train[val_idx]), torch.from_numpy(y_train[val_idx])
    xte, yte = to_tensor(x_test), torch.from_numpy(y_test)

    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(TensorDataset(xtr, ytr), batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(TensorDataset(xva, yva), batch_size=512)
    test_loader = DataLoader(TensorDataset(xte, yte), batch_size=512)
    return train_loader, val_loader, test_loader
