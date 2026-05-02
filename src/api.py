"""
api.py — Histopatoloji Inference API
=====================================
Çalıştır : uvicorn api:app --reload
Endpoint  : POST /predict  (multipart/form-data, alan adı: file)
Döner     : JSON — tahmin, olasılıklar, klinik gruplama
"""

import io
import os
import uuid
import copy
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
from torchvision import transforms
from supabase import create_client, Client

from model import EfficientCancerNet, CLASS_NAMES, CANCER_CLASSES, NORMAL_CLASSES

# ─────────────────────────────────────────────
#  AYARLAR
# ─────────────────────────────────────────────

CHECKPOINT_PATH = Path(__file__).parent.parent / "outputs" / "checkpoints" / "best_model.pt"
CONFIDENCE_THRESHOLD = 0.70

# ImageNet normalizasyonu — EfficientNet-B0 pretrained için
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/tiff", "image/bmp"}

# ─────────────────────────────────────────────
#  SUPABASE & FEEDBACK AYARLARI
# ─────────────────────────────────────────────

SUPABASE_URL       = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY       = os.environ.get("SUPABASE_KEY", "")
FEEDBACK_THRESHOLD = int(os.environ.get("FEEDBACK_THRESHOLD", "10"))
FINE_TUNE_LR       = 1e-4
FINE_TUNE_EPOCHS   = 10
BUCKET_NAME        = "feedback-images"

_fine_tuning  = False          # eş zamanlı fine-tune önleme
_model_lock   = threading.Lock()  # inference ↔ fine-tune güvenliği

def get_supabase() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(503, "Supabase yapılandırılmamış")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

# ─────────────────────────────────────────────
#  UYGULAMA DURUMU  (model tek sefer yüklenir)
# ─────────────────────────────────────────────

class AppState:
    model: EfficientCancerNet = None
    device: torch.device      = None


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

    state.model = EfficientCancerNet(num_classes=9, dropout=dropout).to(state.device)
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
    description="9 sınıflı kolon kanseri doku analizi — EfficientNet-B0",
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


# ─────────────────────────────────────────────
#  FEEDBACK ENDPOINT'LERİ
# ─────────────────────────────────────────────

@app.post("/feedback")
async def feedback(
    file: UploadFile = File(...),
    correct_label: int = Form(...),
):
    """
    Kullanıcı düzeltmesini alır, Supabase'e kaydeder.
    FEEDBACK_THRESHOLD dolunca arka planda fine-tune başlatır.
    """
    global _fine_tuning

    if correct_label not in range(len(CLASS_NAMES)):
        raise HTTPException(400, f"Geçersiz etiket: {correct_label} (0-{len(CLASS_NAMES)-1} arası olmalı)")

    sb = get_supabase()

    # Görüntüyü JPEG'e dönüştür ve Supabase Storage'a yükle
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    buf.seek(0)

    img_name = f"{uuid.uuid4()}.jpg"
    sb.storage.from_(BUCKET_NAME).upload(
        path=img_name,
        file=buf.read(),
        file_options={"content-type": "image/jpeg"},
    )

    # Metadata'yı DB'ye kaydet
    sb.table("feedback").insert({
        "image_path": img_name,
        "correct_label": correct_label,
        "fine_tuned": False,
    }).execute()

    # Bekleyen örnek sayısını al
    result  = sb.table("feedback").select("id", count="exact").eq("fine_tuned", False).execute()
    pending = result.count or 0

    # Eşiğe ulaşıldıysa ve fine-tune yoksa başlat
    if pending >= FEEDBACK_THRESHOLD and not _fine_tuning:
        t = threading.Thread(target=_run_fine_tune, daemon=True)
        t.start()
        return JSONResponse({
            "status": "fine_tune_başlatıldı",
            "pending": pending,
            "threshold": FEEDBACK_THRESHOLD,
        })

    return JSONResponse({
        "status": "kaydedildi",
        "pending": pending,
        "threshold": FEEDBACK_THRESHOLD,
        "remaining": max(0, FEEDBACK_THRESHOLD - pending),
    })


@app.get("/feedback/status")
def feedback_status():
    """Kaç geri bildirim toplandı, fine-tune durumu."""
    sb = get_supabase()
    pending = sb.table("feedback").select("id", count="exact").eq("fine_tuned", False).execute().count or 0
    total   = sb.table("feedback").select("id", count="exact").execute().count or 0
    return {
        "pending":    pending,
        "total":      total,
        "threshold":  FEEDBACK_THRESHOLD,
        "fine_tuning": _fine_tuning,
        "class_names": CLASS_NAMES,
    }


# ─────────────────────────────────────────────
#  FINE-TUNE FONKSİYONU (arka plan thread)
# ─────────────────────────────────────────────

def _run_fine_tune():
    global _fine_tuning
    if _fine_tuning:
        return
    _fine_tuning = True

    try:
        sb = get_supabase()

        # İşlenmemiş geri bildirimleri al
        rows = sb.table("feedback").select("*").eq("fine_tuned", False).execute().data
        if not rows:
            return

        print(f"🔄 Fine-tune başlıyor: {len(rows)} örnek")

        images, labels, ids = [], [], []
        for row in rows:
            try:
                img_bytes = sb.storage.from_(BUCKET_NAME).download(row["image_path"])
                img    = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                tensor = _transform(img)
                images.append(tensor)
                labels.append(row["correct_label"])
                ids.append(row["id"])
            except Exception as e:
                print(f"  [UYARI] Görüntü yüklenemedi {row['image_path']}: {e}")

        if not images:
            return

        # Model kopyası üzerinde çalış — inference kesilmesin
        model_copy = copy.deepcopy(state.model)
        model_copy.freeze_backbone()          # sadece head güncellenir
        model_copy.to(state.device)

        X = torch.stack(images).to(state.device)
        y = torch.tensor(labels, dtype=torch.long).to(state.device)

        optimizer = torch.optim.Adam(model_copy.classifier.parameters(), lr=FINE_TUNE_LR)
        criterion = nn.CrossEntropyLoss()

        model_copy.train()
        for epoch in range(FINE_TUNE_EPOCHS):
            optimizer.zero_grad()
            loss = criterion(model_copy(X), y)
            loss.backward()
            optimizer.step()
            print(f"  Epoch {epoch+1}/{FINE_TUNE_EPOCHS}  loss={loss.item():.4f}")

        model_copy.eval()
        model_copy.unfreeze_all()

        # Modeli atomik olarak değiştir
        with _model_lock:
            state.model = model_copy

        # Checkpoint'i güncelle
        torch.save({
            "model_state": state.model.state_dict(),
            "val_acc":  None,
            "val_loss": None,
            "epoch":    -1,
            "config":   {"dropout": 0.4},
        }, CHECKPOINT_PATH)

        print("✅ Fine-tune tamamlandı, model güncellendi")

        # DB'de işaretle ve Supabase'den görüntüleri sil
        for row_id in ids:
            sb.table("feedback").update({"fine_tuned": True}).eq("id", row_id).execute()
        paths = [r["image_path"] for r in rows if r["id"] in ids]
        sb.storage.from_(BUCKET_NAME).remove(paths)
        print(f"🗑️  {len(paths)} görüntü Supabase'den silindi")

    except Exception as e:
        print(f"❌ Fine-tune hatası: {e}")
    finally:
        _fine_tuning = False


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
    with _model_lock:
        with torch.no_grad():
            logits = state.model(tensor)                # [1, 9]
            probs  = F.softmax(logits, dim=1)[0]        # [9]

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
