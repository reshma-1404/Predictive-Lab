from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(data_path: str, target: str):
    data = pd.read_csv(Path(data_path))
    if target not in data.columns:
        raise ValueError(f"Target column '{target}' was not found in the dataset")
    data = data.dropna(subset=[target])
    return data.drop(columns=[target]), data[target]


def make_preprocessor(features):
    numeric = features.select_dtypes(include="number").columns
    categorical = features.select_dtypes(exclude="number").columns
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ])


def split_data(features, target, classification=False):
    return train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target if classification else None,
    )