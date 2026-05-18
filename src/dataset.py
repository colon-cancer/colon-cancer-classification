import os
import random
from pathlib import Path
from collections import defaultdict

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

DATASET_PATH = r"C:\Users\PC\Desktop\bitirme\archive\NCT-CRC-HE-100K"

CLASS_MAP = {
    "NORM": 0,
    "TUM":  1,
    "STR":  2,
    "LYM":  3,
    "MUS":  4,
    "DEB":  5,
    "MUC":  6,
    "ADI":  7,
    "BACK": 8,
}

CLASS_NAMES = [
    "Normal", "Tümör", "Stroma", "Lenfosit",
    "Düz Kas", "Debris", "Mukosa", "Adipoz", "Arka Plan"
]

# ImageNet normalizasyonu — EfficientNet-B0 pretrained backbone için
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

SEED = 42


def scan_dataset(dataset_path: str) -> dict:
    data = defaultdict(list)
    base = Path(dataset_path)

    for folder_name, class_idx in CLASS_MAP.items():
        folder_path = base / folder_name
        if not folder_path.exists():
            print(f"[UYARI] Klasör bulunamadı: {folder_path}")
            continue
        files = list(folder_path.glob("*.tif"))
        data[class_idx] = [str(f) for f in files]
        print(f"  {CLASS_NAMES[class_idx]:12s}: {len(files):6,} görüntü")

    return data


def split_dataset(data: dict, seed: int = SEED) -> tuple[dict, dict, dict]:
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


def get_transforms(split: str) -> transforms.Compose:
    normalize = transforms.Normalize(mean=MEAN, std=STD)

    if split == "train":
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.1,
            ),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])


class HistoDataset(Dataset):

    def __init__(self, data: dict, split: str = "train"):
        self.transform = get_transforms(split)
        self.samples = []

        for class_idx, files in data.items():
            for f in files:
                self.samples.append((f, class_idx))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        img = Image.open(filepath).convert("RGB")  # RGBA vs. sorunları önle
        img = self.transform(img)
        return img, label


def get_dataloaders(
    dataset_path: str = DATASET_PATH,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    print("\nDataset taranıyor...")
    data = scan_dataset(dataset_path)

    print("\nTrain/Val/Test split yapılıyor...")
    train_data, val_data, test_data = split_dataset(data)

    train_set = HistoDataset(train_data, split="train")
    val_set   = HistoDataset(val_data,   split="val")
    test_set  = HistoDataset(test_data,  split="test")

    print(f"\nSplit sonuçları:")
    print(f"  Train : {len(train_set):>7,} görüntü")
    print(f"  Val   : {len(val_set):>7,} görüntü")
    print(f"  Test  : {len(test_set):>7,} görüntü")
    print(f"  Toplam: {len(train_set)+len(val_set)+len(test_set):>7,} görüntü")

    # Windows'ta num_workers > 0 sorun çıkarabilir
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
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


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)

    images, labels = next(iter(train_loader))
    print(f"\nIlk batch:")
    print(f"  images shape : {images.shape}")
    print(f"  labels shape : {labels.shape}")
    print(f"  labels örnek : {labels[:8].tolist()}")
    print(f"  pixel min/max: {images.min():.3f} / {images.max():.3f}")
