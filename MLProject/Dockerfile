# Dockerfile custom untuk serving model prediksi risiko diabetes.
# Dibuat manual (bukan lewat `mlflow models build-docker`) karena tool bawaan
# tersebut sedang bermasalah akibat perubahan eksternal (kebijakan channel
# Anaconda & deprecated bootstrap pip untuk Python 3.9 di base image lama).
# Pendekatan ini lebih ringan, cepat, dan sepenuhnya bisa dikontrol.

FROM python:3.11-slim

WORKDIR /app

# Install dependency terlebih dahulu supaya layer di-cache oleh Docker
COPY requirements_docker.txt .
RUN pip install --no-cache-dir -r requirements_docker.txt

# Salin file aplikasi serving
COPY serve.py .

# model.joblib dan scaler.joblib disalin oleh workflow CI setelah proses
# training selesai (lihat .github/workflows/ci.yml), jadi tidak perlu
# COPY manual di sini kalau belum ada saat build lokal -- lihat README.

EXPOSE 8080

CMD ["python", "serve.py"]
