"""
Ana eğitim betiği
-----------------
Kullanım (proje klasöründen):
    python src/train.py                 # tam eğitim (varsayılan 15 epoch)
    python src/train.py --epochs 3      # hızlı deneme
    python src/train.py --no-baselines  # temel modelleri atla

Çıktılar:
    outputs/models/fashion_cnn_best.pt  -> en iyi CNN ağırlıkları
    outputs/history.json                -> epoch bazında kayıp/doğruluk
    outputs/metrics.json                -> test metrikleri (CNN + temel modeller)
    outputs/classification_report.txt   -> sınıf bazında kesinlik/duyarlılık/F1
    outputs/figures/*.png               -> tüm grafikler
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data_utils import CLASS_NAMES, augment_batch, get_loaders, load_fashion_mnist  # noqa: E402
from models import FashionCNN, SimpleMLP, count_params  # noqa: E402
import visualize  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, criterion, optimizer=None, augment=False, device="cpu"):
    train = optimizer is not None
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            if augment:
                xb = augment_batch(xb)
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            loss = criterion(out, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * xb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            n += xb.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def predict(model, loader, device="cpu"):
    model.eval()
    preds, probs, labels = [], [], []
    for xb, yb in loader:
        out = torch.softmax(model(xb.to(device)), dim=1).cpu()
        probs.append(out)
        preds.append(out.argmax(1))
        labels.append(yb)
    return torch.cat(preds).numpy(), torch.cat(probs).numpy(), torch.cat(labels).numpy()


def compute_metrics(y_true, y_pred):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro")),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
    }


def train_model(model, loaders, epochs, lr, augment, patience, device, ckpt_path, name):
    train_loader, val_loader, _ = loaders
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val, wait = float("inf"), 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, train_loader, criterion, optimizer, augment, device)
        vl, va = run_epoch(model, val_loader, criterion, device=device)
        scheduler.step(vl)
        for k, v in zip(hist, (tl, ta, vl, va, optimizer.param_groups[0]["lr"])):
            hist[k].append(v)
        flag = ""
        if vl < best_val:
            best_val, wait, flag = vl, 0, "  * en iyi model kaydedildi"
            torch.save(model.state_dict(), ckpt_path)
        else:
            wait += 1
        print(f"[{name}] Epoch {ep:02d}/{epochs} | eğitim kayıp {tl:.4f} doğ. {ta:.4f} | "
              f"doğrulama kayıp {vl:.4f} doğ. {va:.4f} | {time.time() - t0:.1f}s{flag}", flush=True)
        if wait >= patience:
            print(f"[{name}] Erken durdurma: {patience} epoch boyunca iyileşme yok.")
            break
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    return hist


def run_baselines(loaders, device, epochs_mlp, results):
    from sklearn.linear_model import LogisticRegression
    x_train, y_train, x_test, y_test = load_fashion_mnist(os.path.join(ROOT, "data"))
    xtr = x_train.reshape(len(x_train), -1) / 255.0
    xte = x_test.reshape(len(x_test), -1) / 255.0

    print("\n[Temel model 1] Lojistik Regresyon eğitiliyor...")
    t0 = time.time()
    import warnings
    warnings.filterwarnings("ignore")
    lr = LogisticRegression(max_iter=300)
    lr.fit(xtr, y_train)
    results["Lojistik Regresyon"] = compute_metrics(y_test, lr.predict(xte))
    results["Lojistik Regresyon"]["train_time_s"] = round(time.time() - t0, 1)
    results["Lojistik Regresyon"]["params"] = int(lr.coef_.size + lr.intercept_.size)
    print(f"  test doğruluğu: {results['Lojistik Regresyon']['accuracy']:.4f}")

    print(f"\n[Temel model 2] MLP eğitiliyor ({epochs_mlp} epoch)...")
    t0 = time.time()
    mlp = SimpleMLP().to(device)
    train_model(mlp, loaders, epochs_mlp, 1e-3, False, 99, device,
                os.path.join(ROOT, "outputs/models/mlp_best.pt"), "MLP")
    yp, _, yt = predict(mlp, loaders[2], device)
    results["MLP (2 gizli katman)"] = compute_metrics(yt, yp)
    results["MLP (2 gizli katman)"]["train_time_s"] = round(time.time() - t0, 1)
    results["MLP (2 gizli katman)"]["params"] = count_params(mlp)
    print(f"  test doğruluğu: {results['MLP (2 gizli katman)']['accuracy']:.4f}")


def main():
    ap = argparse.ArgumentParser(description="Fashion-MNIST CNN eğitimi")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--mlp-epochs", type=int, default=10)
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Cihaz: {device} | PyTorch {torch.__version__}")
    os.makedirs(os.path.join(ROOT, "outputs/figures"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "outputs/models"), exist_ok=True)

    loaders = get_loaders(os.path.join(ROOT, "data"), args.batch_size, seed=args.seed)
    print(f"Eğitim: {len(loaders[0].dataset)} | Doğrulama: {len(loaders[1].dataset)} | Test: {len(loaders[2].dataset)}")
    visualize.plot_dataset_samples(loaders[2].dataset, os.path.join(ROOT, "outputs/figures/veri_ornekleri.png"))

    # ---------------- CNN ----------------
    model = FashionCNN().to(device)
    print(f"\nFashionCNN parametre sayısı: {count_params(model):,}")
    print(model)
    t0 = time.time()
    hist = train_model(model, loaders, args.epochs, args.lr, not args.no_augment, args.patience,
                       device, os.path.join(ROOT, "outputs/models/fashion_cnn_best.pt"), "CNN")
    cnn_time = time.time() - t0

    y_pred, y_prob, y_true = predict(model, loaders[2], device)
    results = {"CNN (bu proje)": compute_metrics(y_true, y_pred)}
    results["CNN (bu proje)"]["train_time_s"] = round(cnn_time, 1)
    results["CNN (bu proje)"]["params"] = count_params(model)
    results["CNN (bu proje)"]["epochs_run"] = len(hist["train_loss"])
    print(f"\nCNN TEST: {json.dumps(results['CNN (bu proje)'], ensure_ascii=False)}")

    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    print("\n" + report)
    with open(os.path.join(ROOT, "outputs/classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)
    cm = confusion_matrix(y_true, y_pred)
    np.savetxt(os.path.join(ROOT, "outputs/confusion_matrix.csv"), cm, fmt="%d", delimiter=",")

    # Grafikler
    fig = os.path.join(ROOT, "outputs/figures")
    visualize.plot_history(hist, os.path.join(fig, "dogruluk_kayip.png"))
    visualize.plot_confusion(cm, os.path.join(fig, "karmasiklik_matrisi.png"))
    visualize.plot_predictions(loaders[2].dataset, y_pred, y_prob, y_true, fig)
    visualize.plot_per_class(y_true, y_pred, os.path.join(fig, "sinif_bazli_f1.png"))

    # ---------------- Temel modeller ----------------
    if not args.no_baselines:
        run_baselines(loaders, device, args.mlp_epochs, results)
        visualize.plot_comparison(results, os.path.join(fig, "model_karsilastirma.png"))

    with open(os.path.join(ROOT, "outputs/history.json"), "w") as f:
        json.dump(hist, f, indent=2)
    with open(os.path.join(ROOT, "outputs/metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nTamamlandı. Sonuçlar 'outputs/' klasöründe.")


if __name__ == "__main__":
    main()
