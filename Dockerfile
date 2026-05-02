FROM python:3.12-slim

WORKDIR /app

# Sistem bağımlılıkları
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# CPU-only torch önce yükle (requirements.txt'teki CUDA index'i devre dışı bırakır)
RUN pip install --no-cache-dir \
    torch==2.2.2 \
    torchvision==0.17.2 \
    --index-url https://download.pytorch.org/whl/cpu

# Geri kalan bağımlılıklar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyaları
COPY src/ ./src/

# Model dosyası (repoda mevcut)
COPY outputs/checkpoints/best_model.pt ./outputs/checkpoints/best_model.pt

WORKDIR /app/src

ENV CORS_ORIGINS="*"

EXPOSE 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]
