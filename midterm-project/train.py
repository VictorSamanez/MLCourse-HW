"""Train the final bank-marketing model and persist the artifact + metadata."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from pipeline import FEATURE_COLUMNS, build_pipeline, load_features_and_target

RANDOM_STATE = 42
THRESHOLD = 0.23

MODEL_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "min_child_weight": 1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "eval_metric": "auc",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(model, x_test, y_test, threshold: float) -> dict[str, float]:
    probability = model.predict_proba(x_test)[:, 1]
    prediction = (probability >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_test, probability)),
        "pr_auc": float(average_precision_score(y_test, probability)),
        "threshold": threshold,
        "precision": float(precision_score(y_test, prediction)),
        "recall": float(recall_score(y_test, prediction)),
        "f1": float(f1_score(y_test, prediction)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/bank-full.csv"))
    parser.add_argument("--model", type=Path, default=Path("model.bin"))
    parser.add_argument("--metadata", type=Path, default=Path("model_metadata.json"))
    args = parser.parse_args()

    features, target, raw = load_features_and_target(args.data)

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    holdout_model = build_pipeline(XGBClassifier(**MODEL_PARAMS))
    holdout_model.fit(x_train, y_train)
    holdout_metrics = evaluate(holdout_model, x_test, y_test, THRESHOLD)
    print("Holdout metrics:")
    print(json.dumps(holdout_metrics, indent=2))

    final_model = build_pipeline(XGBClassifier(**MODEL_PARAMS))
    final_model.fit(features, target)

    with args.model.open("wb") as model_file:
        pickle.dump(final_model, model_file, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = {
        "project": "bank-term-deposit-midterm",
        "dataset": args.data.name,
        "dataset_sha256": sha256(args.data),
        "rows": int(len(raw)),
        "train_rows": int(len(x_train)),
        "holdout_rows": int(len(x_test)),
        "features": FEATURE_COLUMNS,
        "target": "y",
        "positive_class": "yes",
        "model": "XGBClassifier",
        "hyperparameters": MODEL_PARAMS,
        "threshold": THRESHOLD,
        "holdout_metrics": holdout_metrics,
        "versions": {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}",
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "model_sha256": sha256(args.model),
    }
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {args.model} ({args.model.stat().st_size:,} bytes)")
    print(f"Saved {args.metadata}")


if __name__ == "__main__":
    main()
