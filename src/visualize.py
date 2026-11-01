"""Sonuçların görselleştirilmesi (matplotlib)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from data_utils import CLASS_NAMES, MEAN, STD  # noqa: E402

BLUE, ORANGE, GRAY = "#2E5A88", "#D9822B", "#8A8F98"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def _img(t):
    return (t.squeeze().numpy() * STD + MEAN).clip(0, 1)


def plot_dataset_samples(dataset, path):
    x, y = dataset.tensors
    fig, axes = plt.subplots(3, 10, figsize=(12, 4))
    for c in range(10):
        idx = np.where(y.numpy() == c)[0][:3]
        for r, i in enumerate(idx):
            ax = axes[r, c]
            ax.imshow(_img(x[i]), cmap="gray")
            ax.axis("off")
            if r == 0:
                ax.set_title(CLASS_NAMES[c], fontsize=9)
    plt.suptitle("Fashion-MNIST: her sınıftan örnek görüntüler", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_history(h, path):
    ep = np.arange(1, len(h["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(ep, h["train_acc"], "-o", color=BLUE, ms=4, label="Eğitim")
    ax[0].plot(ep, h["val_acc"], "-s", color=ORANGE, ms=4, label="Doğrulama")
    ax[0].set(title="Doğruluk", xlabel="Epoch", ylabel="Doğruluk")
    ax[1].plot(ep, h["train_loss"], "-o", color=BLUE, ms=4, label="Eğitim")
    ax[1].plot(ep, h["val_loss"], "-s", color=ORANGE, ms=4, label="Doğrulama")
    ax[1].set(title="Kayıp (Cross-Entropy)", xlabel="Epoch", ylabel="Kayıp")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(10), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASS_NAMES)
    thr = cm.max() / 2
    for i in range(10):
        for j in range(10):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thr else "black")
    ax.set(xlabel="Tahmin edilen sınıf", ylabel="Gerçek sınıf", title="Karmaşıklık Matrisi (Test seti)")
    plt.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_predictions(dataset, y_pred, y_prob, y_true, outdir, n=12, seed=0):
    x = dataset.tensors[0]
    rng = np.random.default_rng(seed)
    for kind, mask in (("dogru", y_pred == y_true), ("yanlis", y_pred != y_true)):
        idx = rng.choice(np.where(mask)[0], size=min(n, mask.sum()), replace=False)
        fig, axes = plt.subplots(2, 6, figsize=(12, 4.8))
        for ax, i in zip(axes.flat, idx):
            ax.imshow(_img(x[i]), cmap="gray")
            ax.axis("off")
            col = "green" if kind == "dogru" else "red"
            ax.set_title(f"T: {CLASS_NAMES[y_pred[i]]} ({y_prob[i].max():.0%})\nG: {CLASS_NAMES[y_true[i]]}",
                         fontsize=8, color=col)
        baslik = "Doğru tahmin örnekleri" if kind == "dogru" else "Yanlış tahmin örnekleri"
        plt.suptitle(f"{baslik}  (T = tahmin, G = gerçek)", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"ornek_tahminler_{kind}.png"), dpi=150)
        plt.close()


def plot_per_class(y_true, y_pred, path):
    from sklearn.metrics import precision_recall_fscore_support
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred)
    order = np.argsort(f)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(np.array(CLASS_NAMES)[order], f[order], color=BLUE)
    for i, v in enumerate(f[order]):
        ax.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=8)
    ax.set(xlim=(min(0.6, f.min() - 0.05), 1.02), xlabel="F1-Skoru", title="Sınıf bazında F1-Skoru (Test)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_comparison(results, path):
    names = list(results)
    acc = [results[k]["accuracy"] for k in names]
    f1 = [results[k]["f1_macro"] for k in names]
    xi = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(xi - 0.18, acc, 0.36, color=BLUE, label="Doğruluk")
    b2 = ax.bar(xi + 0.18, f1, 0.36, color=ORANGE, label="F1 (makro)")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003, f"{b.get_height():.3f}",
                    ha="center", fontsize=8)
    ax.set_xticks(xi, names)
    ax.set(ylim=(0.8, 0.96), ylabel="Skor", title="Model karşılaştırması (Test seti)")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
