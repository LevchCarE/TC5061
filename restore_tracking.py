"""Reubica las rutas de artefactos de la copia SQLite incluida en este paquete.

Ejecutar una vez despues de descomprimir y antes de iniciar MLflow. No cambia
parametros, metricas, identificadores ni modelos. Conserva una copia del backend.
"""
from pathlib import Path
import shutil
import sqlite3

ROOT = Path(__file__).resolve().parent


def main():
    database = ROOT / "mlflow.db"
    if not database.is_file():
        raise FileNotFoundError("No existe mlflow.db; ejecuta primero train.py")
    backup = ROOT / "mlflow.db.before_relocation"
    if not backup.exists():
        shutil.copy2(database, backup)
    with sqlite3.connect(database) as connection:
        for table, key, column in [("experiments", "experiment_id", "artifact_location"),
            ("runs", "run_uuid", "artifact_uri"), ("logged_models", "model_id", "artifact_location")]:
            # Identificadores de tabla y columna constantes, nunca recibidos del usuario.
            for ident, value in connection.execute(f"SELECT {key}, {column} FROM {table}").fetchall():
                if value and "/mlartifacts" in value:
                    suffix = value.split("/mlartifacts", 1)[1].lstrip("/")
                    new = (ROOT / "mlartifacts" / suffix).as_uri()
                    connection.execute(f"UPDATE {table} SET {column}=? WHERE {key}=?", (new, ident))
    print("Rutas actualizadas. Abre MLflow con: mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5001")


if __name__ == "__main__":
    main()
