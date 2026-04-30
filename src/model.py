"""
model.py — EfficientNet-B0 Tabanlı Kanser Sınıflandırıcı
=========================================================
Mimari:
  Backbone : EfficientNet-B0 (ImageNet pretrained)
             features → AdaptiveAvgPool2d(1,1) → 1280-dim
  Head     : Dropout(0.4) → Linear(1280, 9) → raw logit

İki aşamalı eğitim:
  Faz 1 — freeze_backbone(): sadece head eğitilir
  Faz 2 — unfreeze_all():    tüm model ince ayar yapılır
"""

import torch
import torch.nn as nn
from torchvision import models


# ─────────────────────────────────────────────
#  ANA MODEL
# ─────────────────────────────────────────────

class EfficientCancerNet(nn.Module):
    """
    9 sınıflı histopatoloji sınıflandırıcı.
    EfficientNet-B0 pretrained backbone + özel sınıflandırıcı baş.

    Kullanım:
        model = EfficientCancerNet(num_classes=9)
        output = model(images)  # [batch, 9] logit döner
    """

    def __init__(self, num_classes: int = 9, dropout: float = 0.4):
        super().__init__()

        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        self.backbone   = base.features    # Conv blokları — 1280 kanal çıkış
        self.avgpool    = base.avgpool     # AdaptiveAvgPool2d(1, 1)

        # Orijinal baş atılıyor, yenisi ekleniyor
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, num_classes),  # raw logit — CrossEntropyLoss bunu bekler
        )

        nn.init.xavier_uniform_(self.classifier[1].weight)
        nn.init.zeros_(self.classifier[1].bias)

    def freeze_backbone(self):
        """Faz 1: backbone dondurulur, sadece head güncellenir."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        """Faz 2: tüm model açılır, düşük lr ile ince ayar yapılır."""
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x):
        x = self.backbone(x)        # [B, 1280, 7, 7]
        x = self.avgpool(x)         # [B, 1280, 1, 1]
        x = torch.flatten(x, 1)     # [B, 1280]
        return self.classifier(x)   # [B, 9]


# ─────────────────────────────────────────────
#  INFERENCE YARDIMCILARI
# ─────────────────────────────────────────────

CLASS_NAMES = [
    "Normal", "Tümör", "Stroma", "Lenfosit",
    "Düz Kas", "Debris", "Mukosa", "Adipoz", "Arka Plan"
]

# Klinik gruplama — MUS (Düz Kas) benign doku, Normal grubunda
CANCER_CLASSES      = {1}              # Tümör
NORMAL_CLASSES      = {0, 2, 3, 4, 6}  # Normal, Stroma, Lenfosit, Düz Kas, Mukosa
NONCLINICAL_CLASSES = {5, 7, 8}        # Debris, Adipoz, Arka Plan

CONFIDENCE_THRESHOLD = 0.70            # altında → "Belirsiz"


def predict(model: nn.Module, image_tensor: torch.Tensor, device: torch.device):
    """
    Tek görüntü için tahmin yapar.

    Args:
        model        : eğitilmiş EfficientCancerNet
        image_tensor : [1, 3, 224, 224] normalize edilmiş tensor
        device       : cpu veya cuda

    Returns:
        dict: {
            class_name, class_idx, confidence,
            all_probs, clinical_group
        }
    """
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.to(device))   # [1, 9]
        probs  = torch.softmax(logits, dim=1)[0]  # [9]
        conf, pred = probs.max(dim=0)
        conf = conf.item()
        pred = pred.item()

    if conf < CONFIDENCE_THRESHOLD:
        clinical_group = "Belirsiz"
    elif pred in CANCER_CLASSES:
        clinical_group = "🔴 Kanser Şüphesi"
    elif pred in NORMAL_CLASSES:
        clinical_group = "🟢 Normal Doku"
    else:
        clinical_group = "⚪ Klinik Dışı"

    return {
        "class_name"    : CLASS_NAMES[pred],
        "class_idx"     : pred,
        "confidence"    : round(conf * 100, 2),
        "all_probs"     : {CLASS_NAMES[i]: round(p.item() * 100, 2) for i, p in enumerate(probs)},
        "clinical_group": clinical_group,
    }


# ─────────────────────────────────────────────
#  TEST — direkt çalıştırınca model özetini göster
# ─────────────────────────────────────────────

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = EfficientCancerNet(num_classes=9).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Toplam parametre   : {total_params:,}")
    print(f"Eğitilebilir param : {trainable:,}")

    # Freeze test
    model.freeze_backbone()
    frozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Faz 1 (frozen) param: {frozen_trainable:,}")

    model.unfreeze_all()

    # Dummy forward pass
    dummy = torch.randn(4, 3, 224, 224).to(device)
    out   = model(dummy)
    print(f"\nInput  shape: {dummy.shape}")
    print(f"Output shape: {out.shape}")  # [4, 9]
