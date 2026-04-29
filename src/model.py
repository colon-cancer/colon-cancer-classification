"""
model.py — SimpleCancerNet CNN Mimarisi
=======================================
Mimari:
  Input  : 224×224×3
  Block1 : Conv(32, 5×5) → BN → ReLU → MaxPool   → 112×112×32
  Block2 : Conv(64, 3×3) → BN → ReLU → MaxPool   →  56×56×64
  Block3 : Conv(128,3×3) → BN → ReLU → MaxPool   →  28×28×128
  Block4 : Conv(256,3×3) → BN → ReLU → MaxPool   →  14×14×256
  GAP    : GlobalAveragePooling                   →  256
  Head   : Dense(256) → ReLU → Dropout(0.4)
           Dense(9)   → Softmax (inference'ta)
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────
#  CONV BLOK — tekrar eden yapı
# ─────────────────────────────────────────────

class ConvBlock(nn.Module):
    """
    Tek conv bloğu:
      Conv2d → BatchNorm → ReLU → MaxPool
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,   # padding='same' etkisi
                bias=False,                 # BN varken bias gereksiz
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # boyutu yarıya indir
        )

    def forward(self, x):
        return self.block(x)


# ─────────────────────────────────────────────
#  ANA MODEL
# ─────────────────────────────────────────────

class SimpleCancerNet(nn.Module):
    """
    9 sınıflı histopatoloji CNN classifier.

    Kullanım:
        model = SimpleCancerNet(num_classes=9)
        output = model(images)  # [batch, 9] logit döner
    """

    def __init__(self, num_classes: int = 9, dropout: float = 0.4):
        super().__init__()

        # ── Feature Extractor ──────────────────
        self.features = nn.Sequential(
            ConvBlock(3,   32,  kernel_size=5),  # 224→112  | geniş kernel: doku pattern
            ConvBlock(32,  64,  kernel_size=3),  # 112→56
            ConvBlock(64,  128, kernel_size=3),  #  56→28
            ConvBlock(128, 256, kernel_size=3),  #  28→14
        )

        # ── Global Average Pooling ─────────────
        # Flatten yerine GAP: parametre sayısını düşürür, overfit azalır
        # 14×14×256 → 256
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── Classifier Head ───────────────────
        self.classifier = nn.Sequential(
            nn.Flatten(),                        # [B, 256, 1, 1] → [B, 256]
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),         # logit çıktı (softmax yok)
        )

        # ── Weight Initialization ─────────────
        self._init_weights()

    def _init_weights(self):
        """He initialization — ReLU için optimal."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)   # conv bloklar
        x = self.gap(x)        # global average pool
        x = self.classifier(x) # dense head
        return x               # raw logit — CrossEntropyLoss bunu bekler


# ─────────────────────────────────────────────
#  INFERENCE YARDIMCILARI
# ─────────────────────────────────────────────

CLASS_NAMES = [
    "Normal", "Tumor", "Stroma", "Lympho",
    "Complex", "Debris", "Mucosa", "Adipose", "Background"
]

# Klinik gruplama
CANCER_CLASSES    = {1, 4}          # Tumor, Complex
NORMAL_CLASSES    = {0, 2, 3, 6}    # Normal, Stroma, Lympho, Mucosa
NONCLINICAL_CLASSES = {5, 7, 8}     # Debris, Adipose, Background

CONFIDENCE_THRESHOLD = 0.70         # altında → "Belirsiz"


def predict(model: nn.Module, image_tensor: torch.Tensor, device: torch.device):
    """
    Tek görüntü için tahmin yapar.

    Args:
        model        : eğitilmiş SimpleCancerNet
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
        logits = model(image_tensor.to(device))          # [1, 9]
        probs  = torch.softmax(logits, dim=1)[0]         # [9]
        conf, pred = probs.max(dim=0)
        conf  = conf.item()
        pred  = pred.item()

    # Confidence threshold
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

    model = SimpleCancerNet(num_classes=9).to(device)

    # Parametre sayısı
    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Toplam parametre   : {total_params:,}")
    print(f"Eğitilebilir param : {trainable:,}")

    # Dummy forward pass
    dummy = torch.randn(4, 3, 224, 224).to(device)
    out   = model(dummy)
    print(f"\nInput  shape: {dummy.shape}")
    print(f"Output shape: {out.shape}")   # [4, 9]

    # Katman özeti
    print("\n── Model Mimarisi ──")
    print(model)