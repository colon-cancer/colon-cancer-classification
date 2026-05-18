import torch
import torch.nn as nn
from torchvision import models


class EfficientCancerNet(nn.Module):

    def __init__(self, num_classes: int = 9, dropout: float = 0.4):
        super().__init__()

        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)

        self.backbone   = base.features
        self.avgpool    = base.avgpool

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, num_classes),  # raw logit — CrossEntropyLoss bunu bekler
        )

        nn.init.xavier_uniform_(self.classifier[1].weight)
        nn.init.zeros_(self.classifier[1].bias)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.classifier.parameters():
            p.requires_grad = True

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


CLASS_NAMES = [
    "Normal", "Tümör", "Stroma", "Lenfosit",
    "Düz Kas", "Debris", "Mukosa", "Adipoz", "Arka Plan"
]

# Klinik gruplama — MUS (Düz Kas) benign doku, Normal grubunda
CANCER_CLASSES      = {1}
NORMAL_CLASSES      = {0, 2, 3, 4, 6}
NONCLINICAL_CLASSES = {5, 7, 8}

CONFIDENCE_THRESHOLD = 0.70


def predict(model: nn.Module, image_tensor: torch.Tensor, device: torch.device):
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probs  = torch.softmax(logits, dim=1)[0]
        conf, pred = probs.max(dim=0)
        conf = conf.item()
        pred = pred.item()

    if conf < CONFIDENCE_THRESHOLD:
        clinical_group = "Belirsiz"
    elif pred in CANCER_CLASSES:
        clinical_group = "Kanser Şüphesi"
    elif pred in NORMAL_CLASSES:
        clinical_group = "Normal Doku"
    else:
        clinical_group = "Klinik Dışı"

    return {
        "class_name"    : CLASS_NAMES[pred],
        "class_idx"     : pred,
        "confidence"    : round(conf * 100, 2),
        "all_probs"     : {CLASS_NAMES[i]: round(p.item() * 100, 2) for i, p in enumerate(probs)},
        "clinical_group": clinical_group,
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = EfficientCancerNet(num_classes=9).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Toplam parametre   : {total_params:,}")
    print(f"Eğitilebilir param : {trainable:,}")

    model.freeze_backbone()
    frozen_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Faz 1 (frozen) param: {frozen_trainable:,}")

    model.unfreeze_all()

    dummy = torch.randn(4, 3, 224, 224).to(device)
    out   = model(dummy)
    print(f"\nInput  shape: {dummy.shape}")
    print(f"Output shape: {out.shape}")
