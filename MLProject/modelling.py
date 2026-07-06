"""
modelling.py (versi MLProject untuk Workflow CI)

Melatih model RandomForest untuk prediksi risiko diabetes.
Didesain untuk dijalankan lewat `mlflow run` sesuai definisi di file MLProject,
sehingga menerima parameter melalui argparse (bukan hardcode) agar CI-friendly.
"""

import argparse
import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_dataset(data_path):
    train_df = pd.read_csv(os.path.join(data_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(data_path, "test.csv"))

    X_train = train_df.drop(columns=["Outcome"])
    y_train = train_df["Outcome"]
    X_test = test_df.drop(columns=["Outcome"])
    y_test = test_df["Outcome"]
    return X_train, X_test, y_train, y_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="diabetes_preprocessing")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=7)
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_dataset(args.data_path)

    mlflow.sklearn.autolog()

    with mlflow.start_run(run_name="CI_RandomForest"):
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("test_accuracy", acc)
        mlflow.log_metric("test_precision", prec)
        mlflow.log_metric("test_recall", rec)
        mlflow.log_metric("test_f1_score", f1)

        print(f"accuracy={acc:.4f} precision={prec:.4f} recall={rec:.4f} f1={f1:.4f}")

        # Simpan juga model ke folder lokal agar mudah diambil job CI berikutnya
        os.makedirs("model_output", exist_ok=True)
        mlflow.sklearn.save_model(model, "model_output/model")

        # Simpan juga sebagai .joblib (dipakai oleh Dockerfile custom, lebih ringan
        # dan tidak bergantung pada mekanisme docker-build bawaan mlflow yang
        # kadang bermasalah karena isu eksternal conda/pyenv)
        joblib.dump(model, "model_output/model.joblib")


if __name__ == "__main__":
    main()
