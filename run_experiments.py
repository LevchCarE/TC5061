"""Ejecuta nueve candidatos, una repeticion y evalua el ganador en holdout."""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from train import ROOT, EXPERIMENT, load_data, split_data, evaluate, save_confusion, sha256, write_json


def execute(name, C, weight, solver="lbfgs", role="candidate"):
    command = [sys.executable, str(ROOT / "train.py"), "--C", str(C),
        "--class-weight", weight, "--solver", solver, "--max-iter", "1000",
        "--tol", "0.0001", "--seed", "42", "--run-name", name, "--role", role]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    record = json.loads(result.stdout.strip().splitlines()[-1])
    print(name, record, flush=True)
    return record["run_id"], " ".join(["python", "train.py", *command[2:]])


def main():
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")
    client = mlflow.MlflowClient()
    batch = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    configs = [(1, "balanced", "lbfgs"), (.01, "balanced", "lbfgs"),
        (.1, "balanced", "lbfgs"), (10, "balanced", "lbfgs"),
        (.01, "none", "lbfgs"), (.1, "none", "lbfgs"),
        (1, "none", "lbfgs"), (10, "none", "lbfgs"), (1, "balanced", "liblinear")]
    ids, commands = [], []
    for i, (C, weight, solver) in enumerate(configs, 1):
        name = f"E{i:02d}_C{C}_{weight}_{solver}"
        rid, command = execute(name, C, weight, solver)
        client.set_tag(rid, "batch", batch)
        ids.append(rid)
        commands.append(command)
    repeat, command = execute("R01_repeticion_E01", 1, "balanced", role="reproducibility")
    client.set_tag(repeat, "batch", batch)
    commands.append(command)
    first = client.get_run(ids[0])
    repeated = client.get_run(repeat)
    differences = {k: abs(v-repeated.data.metrics[k]) for k,v in first.data.metrics.items()}
    same_predictions = sha256(ROOT / "results" / ids[0] / "oof_predictions.csv") == sha256(ROOT / "results" / repeat / "oof_predictions.csv")
    reproducibility = {"original_run": ids[0], "repeat_run": repeat,
        "max_metric_abs_difference": max(differences.values()), "identical_oof_predictions": same_predictions,
        "same_split_indices": sha256(ROOT / "results" / ids[0] / "split_indices.json") == sha256(ROOT / "results" / repeat / "split_indices.json")}
    assert reproducibility["max_metric_abs_difference"] == 0 and same_predictions and reproducibility["same_split_indices"]
    rows = []
    for rid in ids:
        run = client.get_run(rid)
        rows.append({"name": run.data.tags["mlflow.runName"], "run_id": rid,
            **{k: run.data.params[k] for k in ["C", "class_weight", "solver", "max_iter", "tol", "seed"]},
            **run.data.metrics})
    table = pd.DataFrame(rows).sort_values(["cv_f1_mean", "cv_average_precision_mean", "name"], ascending=[False,False,True])
    best = client.get_run(table.iloc[0].run_id)
    model = mlflow.sklearn.load_model(best.data.tags["model_uri"])
    clean, info = load_data(ROOT / "data/Breast_Cancer.csv")
    X_train, X_test, y_train, y_test = split_data(clean, 42, .2)
    probabilities = model.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, probabilities)
    out = ROOT / "results"
    table.to_csv(out / "comparison.csv", index=False)
    write_json(out / "reproducibility.json", reproducibility)
    write_json(out / "test_metrics.json", test_metrics)
    save_confusion(y_test, probabilities, out / "confusion_matrix_test.png", "Holdout del modelo seleccionado")
    pd.DataFrame({"row_index": X_test.index, "y_true": y_test,
        "probability_dead": probabilities}).to_csv(out / "test_predictions.csv", index=False)
    (out / "commands.txt").write_text("\n".join(commands)+"\n", encoding="utf-8")
    summary = {"batch": batch, "experiment": EXPERIMENT, "experiment_id": best.info.experiment_id,
        "best_run_id": best.info.run_id, "best_name": best.data.tags["mlflow.runName"],
        "best_params": best.data.params, "best_cv_metrics": best.data.metrics,
        "test_metrics": test_metrics, "reproducibility": reproducibility, "data": info,
        "classes": clean.objetivo.value_counts().to_dict(), "candidate_runs": ids, "repeat_run": repeat}
    write_json(out / "summary.json", summary)
    with mlflow.start_run(run_id=best.info.run_id):
        mlflow.set_tag("selected", "best_cv_f1")
        for name,value in test_metrics.items():
            mlflow.log_metric(f"test_{name}", value)
        for name in ["test_metrics.json", "confusion_matrix_test.png", "test_predictions.csv",
            "comparison.csv", "reproducibility.json", "commands.txt", "summary.json"]:
            mlflow.log_artifact(str(out / name), artifact_path="final_selection")
        mlflow.log_artifact(__file__, artifact_path="source")
    print(table[["name", "cv_f1_mean", "cv_recall_mean", "cv_average_precision_mean"]].to_string(index=False))
    print("Prueba final:", test_metrics)


if __name__ == "__main__":
    main()
