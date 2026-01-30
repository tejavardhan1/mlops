"""Fetch, validate, split data; write to config paths."""
from pathlib import Path
import sys

import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config


def main():
    cfg = load_config()
    paths = cfg["paths"]
    raw_dir = Path(paths["data_raw"])
    processed_dir = Path(paths["data_processed"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = cfg.get("ingestion", {}).get("dataset_name", "wine_quality")
    if dataset_name == "wine_quality":
        data = load_wine()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
    else:
        from sklearn.datasets import load_iris
        data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target

    assert not df.isnull().any().any(), "Nulls found in dataset"
    assert "target" in df.columns, "Missing target column"

    raw_path = raw_dir / "raw.csv"
    df.to_csv(raw_path, index=False)
    print(f"Wrote raw: {raw_path}")

    train_ratio = cfg.get("ingestion", {}).get("train_ratio", 0.8)
    random_state = cfg.get("ingestion", {}).get("random_state", 42)
    train_df, test_df = train_test_split(df, train_size=train_ratio, random_state=random_state)

    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Wrote train: {train_path} ({len(train_df)} rows)")
    print(f"Wrote test:  {test_path} ({len(test_df)} rows)")
    print("Ingestion done.")


if __name__ == "__main__":
    main()
