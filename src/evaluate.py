"""
evaluate.py — Test Seti Değerlendirme
======================================
Yükler  : outputs/checkpoints/best_model.pt
Hesaplar: Accuracy, Confusion Matrix, Per-class F1
Kaydeder: outputs/plots/confusion_matrix.png
          outputs/plots/per_class_f1.png
          outputs/plots/classification_report.txt
"""

import sys
import os
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# src/ içindeyiz, dataset ve model doğrudan import edilebilir
from dataset import get_dataloaders, CLASS_NAMES
from model import SimpleCancerNet

# ─────────────────────────────────────────────
#  AYARLAR
# ─────────────────────────────────────────────

CHECKPOINT_PATH = Path(__file__).parent.parent / "outputs" / "checkpoints" / "best_model.pt"
PLOTS_DIR       = Path(__file__).parent.parent / "outputs" / "plots"
BATCH_SIZE      = 64
NUM_WORKERS     = 0   # Windows'ta güvenli


# ─────────────────────────────────────────────
#  MODEL YÜKLEME
# ─────────────────────────────────────────────

def load_model(checkpoint_path: Path, device: torch.device) -> SimpleCancerNet:
    if not checkpoint_path.exists():
        sys.exit(f"[HATA] Checkpoint bulunamadı: {checkpoint_path}")

    ckpt    = torch.load(checkpoint_path, map_location=device)
    config  = ckpt.get("config", {})
    dropout = config.get("dropout", 0.4)

    model = SimpleCancerNet(num_classes=9, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    val_acc = ckpt.get("val_acc", None)
    epoch   = ckpt.get("epoch", "?")
    if val_acc is not None:
        print(f"  Checkpoint : epoch {epoch}, val_acc={val_acc:.4f}")

    return model


# ─────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────

def run_inference(model: SimpleCancerNet, test_loader, device: torch.device):
    all_preds  = []
    all_labels = []

    total   = len(test_loader.dataset)
    done    = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            logits = model(images)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

            done += len(labels)
            print(f"  İlerleme : {done:>6,} / {total:,}", end="\r")

    print()  # satır sonu
    return np.array(all_labels), np.array(all_preds)


# ─────────────────────────────────────────────
#  CONFUSION MATRIX PLOTU
# ─────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, save_path: Path):
    cm = confusion_matrix(y_true, y_pred)

    # Satır normalize (recall per class)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # — Ham sayılar —
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[0],
        linewidths=0.4,
    )
    axes[0].set_title("Confusion Matrix (Sayı)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Tahmin Edilen Sınıf", fontsize=11)
    axes[0].set_ylabel("Gerçek Sınıf", fontsize=11)
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].tick_params(axis="y", rotation=0)

    # — Normalize (%) —
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[1],
        vmin=0,
        vmax=1,
        linewidths=0.4,
    )
    axes[1].set_title("Confusion Matrix (Normalize)", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Tahmin Edilen Sınıf", fontsize=11)
    axes[1].set_ylabel("Gerçek Sınıf", fontsize=11)
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].tick_params(axis="y", rotation=0)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Kaydedildi: {save_path}")


# ─────────────────────────────────────────────
#  PER-CLASS F1 BAR CHART
# ─────────────────────────────────────────────

def plot_per_class_f1(y_true, y_pred, save_path: Path):
    f1_scores = f1_score(y_true, y_pred, average=None, labels=list(range(9)))
    macro_f1  = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")

    colors = ["#d62728" if s < 0.70 else "#2ca02c" for s in f1_scores]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(CLASS_NAMES, f1_scores, color=colors, edgecolor="white", linewidth=0.8)

    # Değer etiketleri
    for bar, score in zip(bars, f1_scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Makro F1 referans çizgisi
    ax.axhline(macro_f1, color="#1f77b4", linestyle="--", linewidth=1.5,
               label=f"Makro F1 = {macro_f1:.3f}")
    ax.axhline(weighted_f1, color="#ff7f0e", linestyle=":", linewidth=1.5,
               label=f"Ağırlıklı F1 = {weighted_f1:.3f}")

    ax.set_ylim(0, 1.12)
    ax.set_title("Per-class F1 Skoru", fontsize=13, fontweight="bold")
    ax.set_xlabel("Sınıf", fontsize=11)
    ax.set_ylabel("F1 Skoru", fontsize=11)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Kaydedildi: {save_path}")

    return f1_scores, macro_f1, weighted_f1


# ─────────────────────────────────────────────
#  RAPOR KAYDETME
# ─────────────────────────────────────────────

def save_report(y_true, y_pred, accuracy, save_path: Path):
    report = classification_report(
        y_true, y_pred,
        target_names=CLASS_NAMES,
        digits=4,
    )
    with open(save_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  TEST SETİ DEĞERLENDİRME RAPORU\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Genel Accuracy : {accuracy:.4f}  ({accuracy*100:.2f}%)\n\n")
        f.write(report)
    print(f"  Kaydedildi: {save_path}")


# ─────────────────────────────────────────────
#  ANA AKIŞ
# ─────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice    : {device}")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Model yükle
    print("\nModel yükleniyor...")
    model = load_model(CHECKPOINT_PATH, device)

    # Test loader al
    print("\nTest seti hazırlanıyor...")
    _, _, test_loader = get_dataloaders(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)
    print(f"  Test örnekleri: {len(test_loader.dataset):,}")

    # Inference
    print("\nInference çalışıyor...")
    y_true, y_pred = run_inference(model, test_loader, device)

    # Metrikler
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\n{'='*50}")
    print(f"  Test Accuracy : {accuracy:.4f}  ({accuracy*100:.2f}%)")
    print(f"{'='*50}")

    # Plotlar
    print("\nPlotlar oluşturuluyor...")
    plot_confusion_matrix(
        y_true, y_pred,
        save_path=PLOTS_DIR / "confusion_matrix.png",
    )
    f1_scores, macro_f1, weighted_f1 = plot_per_class_f1(
        y_true, y_pred,
        save_path=PLOTS_DIR / "per_class_f1.png",
    )
    save_report(
        y_true, y_pred, accuracy,
        save_path=PLOTS_DIR / "classification_report.txt",
    )

    # Konsol özeti
    print(f"\n{'─'*42}")
    print(f"  {'Sınıf':<12} F1 Skoru")
    print(f"{'─'*42}")
    for name, score in zip(CLASS_NAMES, f1_scores):
        flag = " ⚠" if score < 0.70 else ""
        print(f"  {name:<12} {score:.4f}{flag}")
    print(f"{'─'*42}")
    print(f"  {'Makro F1':<12} {macro_f1:.4f}")
    print(f"  {'Ağırlıklı F1':<12} {weighted_f1:.4f}")
    print(f"{'─'*42}")
    print(f"\nTüm çıktılar outputs/plots/ klasörüne kaydedildi.\n")


if __name__ == "__main__":
    main()
