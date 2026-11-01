"""
Eğitilmiş modeli kullanarak tahmin yapma
----------------------------------------
Kullanım (proje klasöründen):
    python src/predict.py                       # test setinden rastgele 8 görüntü
    python src/predict.py --image resim.png     # kendi görüntünüz (kıyafet, açık arka plan olabilir)
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import CLASS_NAMES, MEAN, STD, load_fashion_mnist, to_tensor  # noqa: E402
from models import FashionCNN  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "outputs/models/fashion_cnn_best.pt")


def load_model():
    if not os.path.exists(CKPT):
        sys.exit("Model bulunamadı. Önce 'python src/train.py' çalıştırın.")
    m = FashionCNN()
    m.load_state_dict(torch.load(CKPT, map_location="cpu"))
    m.eval()
    return m


def load_image(path):
    from PIL import Image
    img = Image.open(path).convert("L").resize((28, 28))
    arr = np.array(img, dtype=np.uint8)
    if arr.mean() > 127:  # Fashion-MNIST: koyu arka plan, açık nesne -> gerekirse ters çevir
        arr = 255 - arr
    return arr[None]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", help="Sınıflandırılacak görüntü dosyası")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    model = load_model()

    if args.image:
        probs = torch.softmax(model(to_tensor(load_image(args.image))), 1)[0]
        top = probs.argsort(descending=True)[:3]
        print(f"Görüntü: {args.image}")
        for i in top:
            print(f"  {CLASS_NAMES[i]:<15} {probs[i]:.2%}")
        return

    _, _, x_test, y_test = load_fashion_mnist(os.path.join(ROOT, "data"))
    idx = np.random.default_rng().choice(len(x_test), args.n, replace=False)
    probs = torch.softmax(model(to_tensor(x_test[idx])), 1)
    pred = probs.argmax(1).numpy()
    print(f"{'#':>5}  {'Gerçek':<15}{'Tahmin':<15}{'Güven':>7}")
    for k, i in enumerate(idx):
        ok = "✓" if pred[k] == y_test[i] else "✗"
        print(f"{i:>5}  {CLASS_NAMES[y_test[i]]:<15}{CLASS_NAMES[pred[k]]:<15}{probs[k].max():>7.1%} {ok}")

    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, args.n, figsize=(2 * args.n, 2.6))
        for ax, k, i in zip(np.atleast_1d(axes), range(args.n), idx):
            ax.imshow(x_test[i], cmap="gray")
            ax.axis("off")
            ax.set_title(f"{CLASS_NAMES[pred[k]]}\n{probs[k].max():.0%}", fontsize=9,
                         color="green" if pred[k] == y_test[i] else "red")
        plt.tight_layout()
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
