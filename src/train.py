"""
train.py — Eğitim Döngüsü
==========================
Özellikler:
  - Train / Val döngüsü
  - Class-weighted CrossEntropyLoss
  - ReduceLROnPlateau scheduler
  - Early stopping
  - En iyi modeli checkpoint olarak kaydet
  - Loss & Accuracy grafiği
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import Counter

from dataset import get_dataloaders, CLASS_NAMES
from model import SimpleCancerNet


# ─────────────────────────────────────────────
#  AYARLAR
# ─────────────────────────────────────────────

CONFIG = {
    "epochs"        : 50,
    "batch_size"    : 32,
    "lr"            : 1e-3,
    "weight_decay"  : 1e-4,
    "dropout"       : 0.4,
    "num_workers"   : 4,
    "patience"      : 7,          # early stopping
    "checkpoint_dir": r"C:\Users\PC\Desktop\bitirme\outputs\checkpoints",
    "plots_dir"     : r"C:\Users\PC\Desktop\bitirme\outputs\plots",
}


# ─────────────────────────────────────────────
#  CLASS WEIGHT HESAPLA
#  Az örneği olan sınıflara daha fazla ağırlık ver
# ─────────────────────────────────────────────

def compute_class_weights(train_loader, num_classes: int, device):
    print("⚖️  Class weight hesaplanıyor...")
    label_counts = Counter()
    for _, labels in train_loader:
        label_counts.update(labels.tolist())

    total = sum(label_counts.values())
    weights = torch.zeros(num_classes)
    for cls in range(num_classes):
        count = label_counts.get(cls, 1)
        weights[cls] = total / (num_classes * count)

    weights = weights / weights.sum() * num_classes  # normalize
    print(f"  Class weights: {[round(w, 3) for w in weights.tolist()]}")
    return weights.to(device)


# ─────────────────────────────────────────────
#  TEK EPOCH — TRAIN
# ─────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

        # Her 100 batch'te bir ilerleme göster
        if (batch_idx + 1) % 100 == 0:
            print(f"    Batch {batch_idx+1}/{len(loader)} | "
                  f"Loss: {loss.item():.4f}", end="\r")

    return total_loss / total, correct / total


# ─────────────────────────────────────────────
#  TEK EPOCH — VALIDATION
# ─────────────────────────────────────────────

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────
#  GRAFİK KAYDET
# ─────────────────────────────────────────────

def save_plots(train_losses, val_losses, train_accs, val_accs, plots_dir):
    os.makedirs(plots_dir, exist_ok=True)
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    ax1.plot(epochs, train_losses, "b-o", label="Train Loss", markersize=4)
    ax1.plot(epochs, val_losses,   "r-o", label="Val Loss",   markersize=4)
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, [a*100 for a in train_accs], "b-o", label="Train Acc", markersize=4)
    ax2.plot(epochs, [a*100 for a in val_accs],   "r-o", label="Val Acc",   markersize=4)
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(plots_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  📊 Grafik kaydedildi: {path}")


# ─────────────────────────────────────────────
#  ANA EĞİTİM FONKSİYONU
# ─────────────────────────────────────────────

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}\n")

    # Klasörleri oluştur
    os.makedirs(CONFIG["checkpoint_dir"], exist_ok=True)
    os.makedirs(CONFIG["plots_dir"], exist_ok=True)

    # ── DataLoader ────────────────────────────
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["num_workers"],
    )

    # ── Model ─────────────────────────────────
    model = SimpleCancerNet(num_classes=9, dropout=CONFIG["dropout"]).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model: SimpleCancerNet | {total_params:,} parametre\n")

    # ── Loss ──────────────────────────────────
    class_weights = compute_class_weights(train_loader, num_classes=9, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ── Optimizer ─────────────────────────────
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
    )

    # ── Scheduler ─────────────────────────────
    # Val loss 3 epoch iyileşmezse lr'yi yarıya indir
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # ── Eğitim Döngüsü ────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs,   val_accs   = [], []

    print("─" * 65)
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | "
          f"{'Val Loss':>8} | {'Val Acc':>7} | {'LR':>8} | {'Time':>6}")
    print("─" * 65)

    for epoch in range(1, CONFIG["epochs"] + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc*100:>8.2f}% | "
              f"{val_loss:>8.4f} | {val_acc*100:>6.2f}% | {current_lr:>8.6f} | {elapsed:>5.1f}s")

        # ── Checkpoint ────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_path = os.path.join(CONFIG["checkpoint_dir"], "best_model.pt")
            torch.save({
                "epoch"      : epoch,
                "model_state": model.state_dict(),
                "optimizer"  : optimizer.state_dict(),
                "val_loss"   : val_loss,
                "val_acc"    : val_acc,
                "config"     : CONFIG,
            }, ckpt_path)
            print(f"         ✅ Checkpoint kaydedildi (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= CONFIG["patience"]:
                print(f"\n⏹️  Early stopping: {CONFIG['patience']} epoch iyileşme yok.")
                break

    print("─" * 65)
    print(f"\n🏁 Eğitim tamamlandı!")
    print(f"   En iyi val_loss : {best_val_loss:.4f}")
    print(f"   Checkpoint      : {CONFIG['checkpoint_dir']}\\best_model.pt")

    save_plots(train_losses, val_losses, train_accs, val_accs, CONFIG["plots_dir"])


# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Windows multiprocessing için gerekli
    import multiprocessing
    multiprocessing.freeze_support()
    train()