"""Entrenamiento reproducible del baseline TC5061 con MLflow Tracking.

La seleccion utiliza CV en entrenamiento; el holdout se abre en run_experiments.py.
"""
import os

# Antes de importar NumPy: un solo hilo evita variacion por reducciones paralelas.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
    average_precision_score, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
EXPERIMENT = "TC5061_BreastCancer_LogisticRegression"
NUMERIC = ["edad", "tamano_tumor_mm", "ganglios_examinados", "ganglios_positivos",
    "estadio_T", "estadio_N", "grado", "estadio_A_distante",
    "receptor_estrogeno_positivo", "receptor_progesterona_positivo"]
CATEGORICAL = ["raza", "estado_civil"]
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", type=Path, default=ROOT / "data/Breast_Cancer.csv", help="CSV original SEER")
    p.add_argument("--C", type=float, default=1.0, help="Inverso de la intensidad de regularizacion L2")
    p.add_argument("--class-weight", choices=["balanced", "none"], default="balanced", help="Ponderacion de las clases")
    p.add_argument("--solver", choices=["lbfgs", "liblinear"], default="lbfgs", help="Algoritmo de optimizacion")
    p.add_argument("--max-iter", type=int, default=1000, help="Maximo de iteraciones")
    p.add_argument("--tol", type=float, default=1e-4, help="Tolerancia de convergencia")
    p.add_argument("--seed", type=int, default=42, help="Semilla de NumPy, Python, split, CV y modelo")
    p.add_argument("--test-size", type=float, default=0.2, help="Fraccion de prueba reservada")
    p.add_argument("--cv-folds", type=int, default=5, help="Numero de pliegues estratificados")
    p.add_argument("--experiment", default=EXPERIMENT, help="Experimento MLflow")
    p.add_argument("--tracking-uri", default=f"sqlite:///{ROOT / 'mlflow.db'}", help="URI del backend MLflow")
    p.add_argument("--run-name", default=None, help="Nombre opcional de la corrida")
    p.add_argument("--role", choices=["candidate", "reproducibility"], default="candidate", help="Uso de la corrida")
    a = p.parse_args(argv)
    if not np.isfinite(a.C) or a.C <= 0 or not np.isfinite(a.tol) or a.tol <= 0:
        p.error("C y tol deben ser finitos y positivos")
    if a.max_iter < 1 or a.cv_folds < 2 or not 0 < a.test_size < 1 or not 0 <= a.seed < 2**32:
        p.error("Verifica max-iter, cv-folds, test-size y seed")
    return a


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_data(path):
    """Replica las conversiones fijas del notebook, sin aprender del holdout."""
    raw = pd.read_csv(path)
    raw.columns = raw.columns.str.strip()
    required = ["Age", "Tumor Size", "Regional Node Examined", "Reginol Node Positive",
        "T Stage", "N Stage", "Grade", "A Stage", "Estrogen Status",
        "Progesterone Status", "Race", "Marital Status", "Status"]
    missing = set(required) - set(raw.columns)
    if missing:
        raise ValueError(f"Columnas faltantes: {sorted(missing)}")
    for col in raw.select_dtypes(include="object"):
        raw[col] = raw[col].str.strip()
    df = raw.drop_duplicates().reset_index(drop=True)
    maps = {"Status": {"Alive": 0, "Dead": 1}, "A Stage": {"Regional": 0, "Distant": 1},
        "Estrogen Status": {"Negative": 0, "Positive": 1},
        "Progesterone Status": {"Negative": 0, "Positive": 1},
        "T Stage": {"T1": 1, "T2": 2, "T3": 3, "T4": 4},
        "N Stage": {"N1": 1, "N2": 2, "N3": 3},
        "Grade": {"1": 1, "2": 2, "3": 3, "anaplastic; Grade IV": 4}}
    for col, mapping in maps.items():
        df[col] = df[col].map(mapping)
    rename = dict(zip(required, NUMERIC[:4] + ["estadio_T", "estadio_N", "grado",
        "estadio_A_distante", "receptor_estrogeno_positivo", "receptor_progesterona_positivo",
        "raza", "estado_civil", "objetivo"]))
    clean = df[required].rename(columns=rename)
    if clean.isna().any().any():
        raise ValueError("Hay faltantes o categorias no reconocidas; revisa el CSV")
    clean[NUMERIC] = clean[NUMERIC].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(clean[NUMERIC].to_numpy()).all():
        raise ValueError("Se encontraron valores numericos no finitos")
    if set(clean.objetivo.unique()) != {0, 1}:
        raise ValueError("Se requieren ambas clases")
    return clean, {"rows_raw": len(raw), "rows_clean": len(clean), "duplicates_removed": len(raw)-len(clean)}


def split_data(clean, seed, test_size):
    return train_test_split(clean[NUMERIC + CATEGORICAL], clean.objetivo,
        test_size=test_size, stratify=clean.objetivo, random_state=seed)


def build_model(a):
    preprocessing = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)])
    model = LogisticRegression(C=a.C, class_weight=None if a.class_weight == "none" else "balanced",
        solver=a.solver, max_iter=a.max_iter, tol=a.tol, random_state=a.seed)
    return Pipeline([("preprocessing", preprocessing), ("classifier", model)])


def evaluate(y, probabilities):
    pred = (probabilities >= 0.5).astype(int)
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, probabilities), "average_precision": average_precision_score(y, probabilities)}


def save_confusion(y, probabilities, path, title):
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y, (probabilities >= .5).astype(int),
        display_labels=["Alive (0)", "Dead (1)"], ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def train(a):
    random.seed(a.seed)
    np.random.seed(a.seed)
    clean, data_info = load_data(a.data)
    X_train, X_test, y_train, y_test = split_data(clean, a.seed, a.test_size)
    if y_train.value_counts().min() < a.cv_folds:
        raise ValueError("Muy pocos ejemplos por clase para los pliegues solicitados")
    mlflow.set_tracking_uri(a.tracking_uri)
    client = mlflow.MlflowClient()
    if client.get_experiment_by_name(a.experiment) is None:
        client.create_experiment(a.experiment, artifact_location=(ROOT / "mlartifacts").as_uri())
    mlflow.set_experiment(a.experiment)
    with mlflow.start_run(run_name=a.run_name) as run:
        run_id = run.info.run_id
        out = ROOT / "results" / run_id
        out.mkdir(parents=True, exist_ok=True)
        params = {"C": a.C, "class_weight": a.class_weight, "solver": a.solver,
            "max_iter": a.max_iter, "tol": a.tol, "seed": a.seed, "test_size": a.test_size,
            "cv_folds": a.cv_folds, "threshold": .5, "regularization": "L2",
            "data_sha256": sha256(a.data), "code_sha256": sha256(__file__),
            "train_rows": len(X_train), "test_rows": len(X_test), **data_info}
        for name, value in params.items():
            mlflow.log_param(name, value)
        mlflow.set_tags({"role": a.role, "selection_metric": "cv_f1_mean", "positive_class": "Dead",
            "author": "Carlo Eugenio Leviaguirre Chavez", "student_id": "A01022264"})
        model = build_model(a)
        cv = StratifiedKFold(n_splits=a.cv_folds, shuffle=True, random_state=a.seed)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            scores = cross_validate(model, X_train, y_train, cv=cv, scoring=SCORING,
                n_jobs=1, error_score="raise")
            probabilities = cross_val_predict(model, X_train, y_train, cv=cv,
                method="predict_proba", n_jobs=1)[:, 1]
            model.fit(X_train, y_train)
        fold_scores = pd.DataFrame({m: scores[f"test_{m}"] for m in SCORING})
        metrics = {f"cv_{m}_{stat}": float(getattr(fold_scores[m], stat)(**({"ddof": 1} if stat == "std" else {})))
            for m in SCORING for stat in ["mean", "std"]}
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        fold_scores.to_csv(out / "cv_folds.csv", index=False)
        pd.DataFrame({"row_index": X_train.index, "y_true": y_train,
            "probability_dead": probabilities}).to_csv(out / "oof_predictions.csv", index=False)
        write_json(out / "metrics.json", metrics)
        write_json(out / "config.json", params)
        write_json(out / "split_indices.json", {"train": X_train.index.tolist(), "test": X_test.index.tolist()})
        versions = {n: importlib.metadata.version(n) for n in
            ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "mlflow", "cloudpickle"]}
        write_json(out / "environment.json", {"python": platform.python_version(), "platform": platform.platform(),
            "packages": versions, "numerical_threads": 1})
        save_confusion(y_train, probabilities, out / "confusion_matrix_oof.png", "Predicciones fuera de pliegue")
        for path in sorted(out.iterdir()):
            mlflow.log_artifact(str(path), artifact_path="evaluation")
        mlflow.log_artifact(__file__, artifact_path="source")
        mlflow.log_artifact(str(ROOT / "requirements.txt"), artifact_path="source")
        model_info = mlflow.sklearn.log_model(model, name="model",
            signature=infer_signature(X_train, model.predict(X_train)),
            input_example=X_train.head(3), pip_requirements=[f"{n}=={v}" for n,v in versions.items()])
        mlflow.set_tag("model_uri", model_info.model_uri)
        write_json(out / "run.json", {"run_id": run_id, "model_uri": model_info.model_uri,
            "experiment_id": run.info.experiment_id})
        print(json.dumps({"run_id": run_id, "cv_f1_mean": metrics["cv_f1_mean"]}))
        return run_id


if __name__ == "__main__":
    train(parse_args())
