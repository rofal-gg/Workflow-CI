# Workflow-CI

Repository ini merupakan bagian dari submission "Membangun Sistem Machine
Learning" (Dicoding) — Kriteria 3: Membuat Workflow CI.

**Nama:** Muhammad Ainur Rofal Achsony

## Struktur
```
Workflow-CI/
├── .github/workflows/ci.yml     # CI: retraining otomatis + build & push Docker image
└── MLProject/
    ├── MLProject                # Definisi MLflow Project
    ├── conda.yaml                # Environment dependencies
    ├── modelling.py              # Skrip training (parametrized untuk CI)
    ├── diabetes_preprocessing/   # Dataset siap latih (dari kriteria 1)
    └── DockerHub.txt             # Tautan & instruksi Docker Hub
```

## Menjalankan MLProject secara lokal
```bash
pip install mlflow==2.19.0 pandas numpy scikit-learn==1.5.2 joblib
cd MLProject
mlflow run . --env-manager=local
```

## Setup GitHub Actions
1. Push repo ini ke GitHub: `https://github.com/rofal-gg/Workflow-CI`
2. Buka **Settings > Secrets and variables > Actions**, tambahkan:
   - `DOCKERHUB_USERNAME` = `achsony`
   - `DOCKERHUB_TOKEN` = (access token dari hub.docker.com/settings/security)
3. Setiap push ke folder `MLProject/`, workflow otomatis:
   - Menjalankan `mlflow run` (retraining)
   - Menyimpan artefak model & mlruns ke GitHub (kriteria Skilled)
   - Build & push Docker image ke Docker Hub via `mlflow models build-docker` (kriteria Advance)

## Cara push ke GitHub
```bash
git init
git add .
git commit -m "Kriteria 3: workflow CI untuk retraining dan build docker image"
git branch -M main
git remote add origin https://github.com/rofal-gg/Workflow-CI.git
git push -u origin main
```
