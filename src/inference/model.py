"""Load model from MODEL_PATH (joblib) or MLflow (run_id or model_name)."""
import os
from pathlib import Path
from typing import Optional

import joblib
import mlflow
import mlflow.sklearn


def load_model(run_id: Optional[str] = None, model_name: Optional[str] = None):
    model_path = os.environ.get("MODEL_PATH")
    if model_path and Path(model_path).exists():
        return joblib.load(model_path)
    if run_id:
        model_uri = f"runs:/{run_id}/model"
        return mlflow.sklearn.load_model(model_uri)
    if model_name:
        model_uri = f"models:/{model_name}/latest"
        return mlflow.sklearn.load_model(model_uri)
    raise ValueError("Set MODEL_PATH, MLFLOW_RUN_ID, or MLFLOW_MODEL_NAME for inference.")


def get_feature_names():
    from src.config import load_config
    cfg = load_config()
    if cfg.get("ingestion", {}).get("dataset_name") == "iris":
        return ["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
    return [
        "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
        "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
        "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline",
    ]
