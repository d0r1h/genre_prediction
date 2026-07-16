"""
models.py
---------
Factory functions for the traditional ML models used in this project.
Each function returns a fresh, un-fitted sklearn Pipeline combining the
shared preprocessor with a classifier, so models stay consistent and
reusable across train.py / evaluate.py.
"""
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from data_preprocessing import build_preprocessor


def get_model_registry(seed: int = 42) -> dict:
    """
    Returns a dict of {model_name: sklearn Pipeline} for every model
    supported by this project. Add new models here to extend the project.
    """
    registry = {
        "logistic_regression": Pipeline([
            ("preprocess", build_preprocessor()),
            ("clf", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=seed,
            )),
        ]),
        "random_forest": Pipeline([
            ("preprocess", build_preprocessor()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )),
        ]),
    }
    return registry


def get_model(name: str, seed: int = 42):
    registry = get_model_registry(seed)
    if name not in registry:
        raise ValueError(f"Unknown model '{name}'. Available: {list(registry.keys())}")
    return registry[name]
