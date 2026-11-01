# Fashion-MNIST Giysi Sınıflandırması – Evrişimli Sinir Ağı (CNN)

**LBLM368 – Derin Öğrenme ve Sınıflandırma Teknikleri – Dönem Projesi**

**Öğrenci:** Hirbod Yaghoubi

Bu proje, Fashion-MNIST veri setindeki 28×28 gri tonlamalı giysi görüntülerini
10 sınıfa ayıran bir CNN modelini PyTorch ile eğitir ve Lojistik Regresyon / MLP
temel modelleriyle karşılaştırır.

## Klasör yapısı

```
FashionMNIST_CNN_Projesi/
├── data/                     Fashion-MNIST dosyaları (.gz) – hazır, internet gerekmez
├── src/
│   ├── data_utils.py         Veri yükleme, normalizasyon, veri artırma
│   ├── models.py             FashionCNN ve SimpleMLP modelleri
│   ├── train.py              Eğitim + değerlendirme + grafikler (ANA BETİK)
│   ├── visualize.py          Grafik fonksiyonları
│   └── predict.py            Eğitilmiş modelle tahmin
├── outputs/
│   ├── models/fashion_cnn_best.pt   Eğitilmiş model (hazır)
│   ├── figures/                     Tüm grafikler
│   ├── metrics.json                 Test metrikleri
│   ├── classification_report.txt    Sınıf bazında metrikler
│   └── egitim_log.txt               Eğitim çıktısı
├── Fashion_MNIST_CNN.ipynb   Aynı projenin Jupyter Notebook sürümü
└── requirements.txt
```

## Kurulum

Python 3.9+ gereklidir.

```bash
pip install -r requirements.txt
```

## Çalıştırma

Tüm komutlar proje klasörünün içinden çalıştırılır.

```bash
# 1) Hazır eğitilmiş modelle hemen tahmin (birkaç saniye)
python src/predict.py

# 2) Modeli baştan eğitmek (CPU'da ~15-20 dk; GPU varsa otomatik kullanılır)
python src/train.py

# Hızlı deneme (3 epoch, temel modeller olmadan)
python src/train.py --epochs 3 --no-baselines

# Kendi görüntünüzü sınıflandırmak
python src/predict.py --image ornek.png
```

Windows'ta `python` yerine `py` kullanmanız gerekebilir.

Alternatif: `Fashion_MNIST_CNN.ipynb` dosyasını Jupyter Notebook / VS Code / Google Colab
ile açıp hücreleri sırayla çalıştırabilirsiniz.

## Sınıflar

0 T-shirt/üst · 1 Pantolon · 2 Kazak · 3 Elbise · 4 Ceket · 5 Sandalet · 6 Gömlek · 7 Spor ayakkabı · 8 Çanta · 9 Bot
