"""
serve.py

Serving API untuk model prediksi risiko diabetes, dijalankan di dalam
Docker image yang dibangun oleh GitHub Actions (lihat ../.github/workflows/ci.yml).

Endpoint:
    GET  /health   -> health check
    POST /predict  -> prediksi risiko diabetes
    GET  /metrics  -> metrik Prometheus (agar image ini juga bisa langsung
                      dipakai untuk kriteria 4 - monitoring & logging)
"""

import time
import os
import joblib
import psutil
import pandas as pd
from flask import Flask, request, jsonify, Response
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

FEATURE_ORDER = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
_start_load = time.time()
model = joblib.load(os.path.join(BASE_DIR, "model.joblib"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.joblib"))
MODEL_LOAD_TIME = time.time() - _start_load
SERVICE_START_TIME = time.time()

REQUEST_TOTAL = Counter("ml_request_total", "Total request masuk ke endpoint /predict")
REQUEST_SUCCESS = Counter("ml_request_success_total", "Total request yang berhasil diproses")
REQUEST_ERROR = Counter("ml_request_error_total", "Total request yang gagal/error")
PREDICTION_TOTAL = Counter("ml_prediction_total", "Total prediksi per kelas", ["predicted_class"])
REQUEST_LATENCY = Histogram("ml_request_latency_seconds", "Distribusi latensi request /predict")
PREDICTION_PROBA = Histogram("ml_prediction_probability", "Distribusi probabilitas prediksi kelas positif",
                              buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
ACTIVE_REQUESTS = Gauge("ml_active_requests", "Jumlah request yang sedang diproses saat ini")
CPU_USAGE = Gauge("ml_cpu_usage_percent", "Penggunaan CPU sistem (%)")
MEM_USAGE_PERCENT = Gauge("ml_memory_usage_percent", "Penggunaan memori sistem (%)")
MEM_USAGE_BYTES = Gauge("ml_memory_usage_bytes", "Penggunaan memori sistem (bytes)")
MODEL_LOAD_TIME_GAUGE = Gauge("ml_model_load_time_seconds", "Waktu load model saat startup")
UPTIME_GAUGE = Gauge("ml_uptime_seconds", "Lama service berjalan sejak start (detik)")
MODEL_LOAD_TIME_GAUGE.set(MODEL_LOAD_TIME)


def refresh_system_metrics():
    CPU_USAGE.set(psutil.cpu_percent(interval=0.1))
    mem = psutil.virtual_memory()
    MEM_USAGE_PERCENT.set(mem.percent)
    MEM_USAGE_BYTES.set(mem.used)
    UPTIME_GAUGE.set(time.time() - SERVICE_START_TIME)


@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_TOTAL.inc()
    ACTIVE_REQUESTS.inc()
    start_time = time.time()
    try:
        payload = request.get_json(force=True)
        row = [payload[col] for col in FEATURE_ORDER]
        df = pd.DataFrame([row], columns=FEATURE_ORDER)
        df_scaled = pd.DataFrame(scaler.transform(df), columns=FEATURE_ORDER)

        proba = model.predict_proba(df_scaled)[0][1]
        pred_class = int(model.predict(df_scaled)[0])

        PREDICTION_TOTAL.labels(predicted_class=str(pred_class)).inc()
        PREDICTION_PROBA.observe(proba)
        REQUEST_SUCCESS.inc()

        return jsonify({
            "prediction": pred_class,
            "probability_diabetes": round(float(proba), 4),
            "label": "Diabetes" if pred_class == 1 else "Tidak Diabetes"
        })
    except Exception as e:
        REQUEST_ERROR.inc()
        return jsonify({"error": str(e)}), 400
    finally:
        REQUEST_LATENCY.observe(time.time() - start_time)
        ACTIVE_REQUESTS.dec()


@app.route("/metrics", methods=["GET"])
def metrics():
    refresh_system_metrics()
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print(f"Model loaded in {MODEL_LOAD_TIME:.4f}s. Serving on http://0.0.0.0:8080")
    app.run(host="0.0.0.0", port=8080)
