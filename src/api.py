"""
api.py — Histopatoloji Inference API
=====================================
Çalıştır : uvicorn api:app --reload
Endpoint  : POST /predict  (multipart/form-data, alan adı: file)
Döner     : JSON — tahmin, olasılıklar, klinik gruplama
"""

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms

from model import SimpleCancerNet, CLASS_NAMES, CANCER_CLASSES, NORMAL_CLASSES

# ─────────────────────────────────────────────
#  AYARLAR
# ─────────────────────────────────────────────

CHECKPOINT_PATH = Path(__file__).parent.parent / "outputs" / "checkpoints" / "best_model.pt"
CONFIDENCE_THRESHOLD = 0.70

MEAN = [0.747, 0.540, 0.716]
STD  = [0.091, 0.137, 0.091]

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/tiff", "image/bmp"}

# ─────────────────────────────────────────────
#  UYGULAMA DURUMU  (model tek sefer yüklenir)
# ─────────────────────────────────────────────

class AppState:
    model: SimpleCancerNet = None
    device: torch.device   = None


state = AppState()

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──
    state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(f"Checkpoint bulunamadı: {CHECKPOINT_PATH}")

    ckpt    = torch.load(CHECKPOINT_PATH, map_location=state.device)
    config  = ckpt.get("config", {})
    dropout = config.get("dropout", 0.4)

    state.model = SimpleCancerNet(num_classes=9, dropout=dropout).to(state.device)
    state.model.load_state_dict(ckpt["model_state"])
    state.model.eval()

    val_acc = ckpt.get("val_acc")
    print(f"Model yüklendi — device={state.device}"
          + (f", val_acc={val_acc:.4f}" if val_acc else ""))
    yield
    # ── Shutdown ── (temizlik gerekmez)


# ─────────────────────────────────────────────
#  FASTAPI UYGULAMASI
# ─────────────────────────────────────────────

app = FastAPI(
    title="Histopatoloji Sınıflandırma API",
    description="9 sınıflı kolon kanseri doku analizi — SimpleCancerNet",
    version="1.0.0",
    lifespan=lifespan,
)

_cors_origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
#  YARDIMCI
# ─────────────────────────────────────────────

def clinical_group(class_idx: int, confidence: float) -> str:
    if confidence < CONFIDENCE_THRESHOLD:
        return "Belirsiz"
    if class_idx in CANCER_CLASSES:
        return "Kanser Şüphesi"
    if class_idx in NORMAL_CLASSES:
        return "Normal Doku"
    return "Klinik Dışı"


# ─────────────────────────────────────────────
#  ENDPOINT'LER
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": state.model is not None,
        "device": str(state.device),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(..., description="JPG/PNG histopatoloji görüntüsü")):
    # ── Dosya tipi kontrolü ──
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Desteklenmeyen dosya tipi: {file.content_type}. "
                   f"Kabul edilenler: {', '.join(ALLOWED_CONTENT_TYPES)}",
        )

    # ── Görüntü yükleme ──
    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Görüntü okunamadı: {exc}")

    # ── Preprocessing ──
    tensor = _transform(img).unsqueeze(0).to(state.device)  # [1, 3, 224, 224]

    # ── Inference ──
    with torch.no_grad():
        logits = state.model(tensor)                    # [1, 9]
        probs  = F.softmax(logits, dim=1)[0]            # [9]

    confidence, pred_idx = probs.max(dim=0)
    confidence = confidence.item()
    pred_idx   = pred_idx.item()

    # ── Yanıt ──
    return JSONResponse({
        "prediction": {
            "class_name":      CLASS_NAMES[pred_idx],
            "class_index":     pred_idx,
            "confidence":      round(confidence * 100, 2),
            "clinical_group":  clinical_group(pred_idx, confidence),
        },
        "all_probabilities": {
            CLASS_NAMES[i]: round(probs[i].item() * 100, 2)
            for i in range(len(CLASS_NAMES))
        },
        "meta": {
            "image_size":  f"{img.width}x{img.height}",
            "device":      str(state.device),
            "threshold":   CONFIDENCE_THRESHOLD,
        },
    })
