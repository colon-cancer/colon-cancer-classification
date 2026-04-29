"""
dataset.py — Histopatoloji Dataset + DataLoader
================================================
Görevler:
  1. 9 kategoriyi tara, dosya listesi oluştur
  2. Patient-level train/val/test split yap
  3. Augmentation + Normalization uygula
  4. PyTorch DataLoader döndür
"""

import os
import random
from pathlib import Path
from collections import defaultdict

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ─────────────────────────────────────────────
#  AYARLAR — sadece burası değiştirilir
# ─────────────────────────────────────────────

DATASET_PATH = r"C:\Users\PC\Desktop\archive\9 Class Colon Cancer Histopathological Image"

# Klasör adı → sınıf indeksi eşlemesi
CLASS_MAP = {
    "1. Normal":     0,
    "2. Tumor":      1,
    "3. Stroma":     2,
    "4. Lympho":     3,
    "5. Complex":    4,
    "6. Debris":     5,
    "7. Mucosa":     6,
    "8. Adipose":    7,
    "9. Background": 8,
}

CLASS_NAMES = [
    "Normal", "Tumor", "Stroma", "Lympho",
    "Complex", "Debris", "Mucosa", "Adipose", "Background"
]

# Dataset-specific normalization (kendi verisetinden hesaplandı)
MEAN = [0.747, 0.540, 0.716]
STD  = [0.091, 0.137, 0.091]

# Split oranları
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

SEED = 42


# ─────────────────────────────────────────────
#  DOSYA TARAMA
# ─────────────────────────────────────────────

def scan_dataset(dataset_path: str) -> dict:
    """
    Dataset klasörünü tarar.
    Döndürür: {class_idx: [filepath, ...]}
    """
    data = defaultdict(list)
    base = Path(dataset_path)

    for folder_name, class_idx in CLASS_MAP.items():
        folder_path = base / folder_name
        if not folder_path.exists():
            print(f"[UYARI] Klasör bulunamadı: {folder_path}")
            continue
        files = list(folder_path.glob("*.jpg")) + list(folder_path.glob("*.png"))
        data[class_idx] = [str(f) for f in files]
        print(f"  {CLASS_NAMES[class_idx]:12s}: {len(files):6,} görüntü")

    return data


# ─────────────────────────────────────────────
#  PATIENT-LEVEL SPLIT
#  NCT-CRC dosya adları: blk-XXXXXXXX-SINIF.tif gibi
#  Burada basit random split yapıyoruz
#  (aynı hastaya ait patch yoksa bu yeterli)
# ─────────────────────────────────────────────

def split_dataset(data: dict, seed: int = SEED) -> tuple[dict, dict, dict]:
    """
    Her sınıf için dosyaları train/val/test'e böl.
    Döndürür: (train_data, val_data, test_data)
    Her biri: {class_idx: [filepath, ...]}
    """
    random.seed(seed)
    train_data = defaultdict(list)
    val_data   = defaultdict(list)
    test_data  = defaultdict(list)

    for class_idx, files in data.items():
        random.shuffle(files)
        n = len(files)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        train_data[class_idx] = files[:n_train]
        val_data[class_idx]   = files[n_train:n_train + n_val]
        test_data[class_idx]  = files[n_train + n_val:]

    return train_data, val_data, test_data


# ─────────────────────────────────────────────
#  TRANSFORM TANIMLARI
# ─────────────────────────────────────────────

def get_transforms(split: str) -> transforms.Compose:
    """
    split: 'train' | 'val' | 'test'
    - train: augmentation + normalize
    - val/test: sadece normalize
    """
    normalize = transforms.Normalize(mean=MEAN, std=STD)

    if split == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),          # zaten 224 ama garanti ol
            transforms.RandomHorizontalFlip(),       # yatay çevirme
            transforms.RandomVerticalFlip(),         # dikey çevirme
            transforms.RandomRotation(90),           # 90° rotasyon
            transforms.ColorJitter(                  # stain varyasyonu simüle
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05
            ),
            transforms.ToTensor(),                   # [0,255] → [0,1]
            normalize,
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])


# ─────────────────────────────────────────────
#  DATASET CLASS
# ─────────────────────────────────────────────

class HistoDataset(Dataset):
    """
    PyTorch Dataset — histopatoloji görüntüleri için.

    Kullanım:
        dataset = HistoDataset(data_dict, split='train')
        img, label = dataset[0]
    """

    def __init__(self, data: dict, split: str = "train"):
        self.transform = get_transforms(split)
        self.samples = []  # [(filepath, class_idx), ...]

        for class_idx, files in data.items():
            for f in files:
                self.samples.append((f, class_idx))

        # Shuffle
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        img = Image.open(filepath).convert("RGB")  # RGBA vs. sorunları önle
        img = self.transform(img)
        return img, label


# ─────────────────────────────────────────────
#  DATALOADER FACTORY
# ─────────────────────────────────────────────

def get_dataloaders(
    dataset_path: str = DATASET_PATH,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Ana fonksiyon — train/val/test DataLoader döndürür.

    Kullanım:
        train_loader, val_loader, test_loader = get_dataloaders()
    """
    print("\n📂 Dataset taranıyor...")
    data = scan_dataset(dataset_path)

    print("\n✂️  Train/Val/Test split yapılıyor...")
    train_data, val_data, test_data = split_dataset(data)

    train_set = HistoDataset(train_data, split="train")
    val_set   = HistoDataset(val_data,   split="val")
    test_set  = HistoDataset(test_data,  split="test")

    print(f"\n📊 Split sonuçları:")
    print(f"  Train : {len(train_set):>7,} görüntü")
    print(f"  Val   : {len(val_set):>7,} görüntü")
    print(f"  Test  : {len(test_set):>7,} görüntü")
    print(f"  Toplam: {len(train_set)+len(val_set)+len(test_set):>7,} görüntü")

    # Windows'ta num_workers > 0 sorun çıkarabilir
    # Sorun olursa num_workers=0 yap
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,    # GPU'ya transfer hızlanır
        persistent_workers=True if num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False,
    )

    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────
#  TEST — direkt çalıştırınca kontrol eder
# ─────────────────────────────────────────────

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)

    # İlk batch'i çek ve boyutları kontrol et
    images, labels = next(iter(train_loader))
    print(f"\n✅ İlk batch:")
    print(f"  images shape : {images.shape}")   # [32, 3, 224, 224]
    print(f"  labels shape : {labels.shape}")   # [32]
    print(f"  labels örnek : {labels[:8].tolist()}")
    print(f"  pixel min/max: {images.min():.3f} / {images.max():.3f}")