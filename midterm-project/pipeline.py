"""Shared feature engineering and model-building utilities for the midterm project."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "y"
DROP_COLUMNS = ("duration",)

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "poutcome",
]

NUMERICAL_FEATURES = [
    "age",
    "balance",
    "day",
    "campaign",
    "pdays",
    "previous",
    "was_contacted_before",
]

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the project feature engineering and drop leakage/target columns."""

    features = df.copy()
    features["was_contacted_before"] = (features["pdays"] != -1).astype(int)
    drop = [column for column in (*DROP_COLUMNS, TARGET) if column in features.columns]
    return features.drop(columns=drop)


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the UCI bank-full.csv (semicolon separated)."""

    return pd.read_csv(path, sep=";")


def load_features_and_target(path: str | Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    raw = load_raw(path)
    target = raw[TARGET].map({"no": 0, "yes": 1}).astype(int)
    features = engineer_features(raw)
    return features, target, raw


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_pipeline(classifier) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", classifier),
        ]
    )
