"""Train model, log to MLflow, register artifact."""
from pathlib import Path
import sys

import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config


def main():
    cfg = load_config()
    paths = cfg["paths"]
    processed_dir = Path(paths["data_processed"])
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"

    if not train_path.exists() or not test_path.exists():
        print("Run scripts/ingest.py first.")
        sys.exit(1)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    feature_cols = [c for c in train_df.columns if c != "target"]
    X_train = train_df[feature_cols]
    y_train = train_df["target"]
    X_test = test_df[feature_cols]
    y_test = test_df["target"]

    train_cfg = cfg.get("training", {})
    exp_name = train_cfg.get("experiment_name", "wine_quality")
    model_name = train_cfg.get("model_name", "sklearn_wine")
    hp = train_cfg.get("hyperparameters", {})

    mlflow.set_experiment(exp_name)
    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=hp.get("n_estimators", 100),
            max_depth=hp.get("max_depth", 6),
            random_state=hp.get("random_state", 42),
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")

        mlflow.log_params({
            "n_estimators": hp.get("n_estimators", 100),
            "max_depth": hp.get("max_depth", 6),
        })
        mlflow.log_metrics({"accuracy": accuracy, "f1_weighted": f1})
        mlflow.sklearn.log_model(model, "model", registered_model_name=model_name)

        model_dir = Path("/models/sklearn_wine")
        model_dir.mkdir(parents=True, exist_ok=True)
        pkl_path = model_dir / "model.pkl"
        joblib.dump(model, pkl_path)
        print(f"Model saved to {pkl_path}")

    print(f"Run finished. accuracy={accuracy:.4f} f1_weighted={f1:.4f}")
    print("Model logged to MLflow. Use MLflow UI or MLFLOW_RUN_ID for inference.")


if __name__ == "__main__":
    main()
