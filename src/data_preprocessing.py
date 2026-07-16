"""
data_preprocessing.py
----------------------
Handles: loading the raw CSV, basic exploratory data analysis (EDA),
feature engineering, and building the sklearn ColumnTransformer used to
turn raw columns into a model-ready numeric matrix.

This module is intentionally free of any model-training code so it can be
reused by both train.py and evaluate.py.
"""
import os
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer

from genre_map import get_primary_broad_genre

sns.set_style("whitegrid")

NUMERIC_FEATURES = ["release_year", "duration_value", "cast_size", "years_since_release"]
CATEGORICAL_FEATURES = ["type", "rating", "platform", "country_main"]
TEXT_FEATURE = "description"
TARGET_COL = "genre"


# Loading
def load_data(csv_path: str) -> pd.DataFrame:
    """Load the raw TV-shows/movies CSV into a DataFrame."""
    df = pd.read_csv(csv_path)
    return df


# EDA
def run_eda(df: pd.DataFrame, output_dir: str) -> dict:
    """
    Produce basic EDA artifacts (printed stats + saved plots).
    Returns a dict summary of key EDA findings (useful for the report).
    """
    os.makedirs(output_dir, exist_ok=True)
    summary = {}

    summary["shape"] = df.shape
    summary["missing_values"] = df.isnull().sum().to_dict()
    summary["type_counts"] = df["type"].value_counts().to_dict()
    summary["platform_counts"] = df["platform"].value_counts().to_dict()

    # --- Missing values plot ---
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(x=missing.values, y=missing.index, color="#4C72B0")
    plt.title("Missing Values per Column")
    plt.xlabel("Number of Missing Rows")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "missing_values.png"), dpi=150)
    plt.close()

    # --- Movie vs TV Show distribution ---
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="type", hue="type", palette="Set2", legend=False)
    plt.title("Movie vs TV Show Distribution")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "type_distribution.png"), dpi=150)
    plt.close()

    # --- Release year distribution ---
    plt.figure(figsize=(8, 5))
    sns.histplot(df["release_year"].dropna(), bins=30, color="#55A868")
    plt.title("Release Year Distribution")
    plt.xlabel("Release Year")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "release_year_distribution.png"), dpi=150)
    plt.close()

    # --- Rating distribution ---
    plt.figure(figsize=(9, 5))
    top_ratings = df["rating"].value_counts().head(12)
    sns.barplot(x=top_ratings.values, y=top_ratings.index, color="#C44E52")
    plt.title("Top Content Ratings")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rating_distribution.png"), dpi=150)
    plt.close()

    # --- Genre (target) distribution, computed after mapping ---
    genres = df["listed_in"].astype(str).apply(get_primary_broad_genre)
    genre_counts = genres.value_counts()
    summary["genre_counts"] = genre_counts.to_dict()
    plt.figure(figsize=(9, 6))
    sns.barplot(x=genre_counts.values, y=genre_counts.index, color="#8172B2")
    plt.title("Target Class Distribution (Broad Genre)")
    plt.xlabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "genre_class_distribution.png"), dpi=150)
    plt.close()

    # --- Correlation heatmap for numeric engineered features ---
    df_num = engineer_features(df.copy())
    num_cols = [c for c in NUMERIC_FEATURES if c in df_num.columns]
    corr = df_num[num_cols].corr()
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Between Numeric Features")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "numeric_feature_correlation.png"), dpi=150)
    plt.close()

    return summary


# Feature engineering
def _parse_duration(row) -> float:
    """Convert duration strings ('90 min' / '2 Seasons') into a numeric value."""
    val = row["duration"]
    if pd.isna(val):
        return np.nan
    match = re.search(r"(\d+)", str(val))
    return float(match.group(1)) if match else np.nan


def _count_cast(cast_str) -> int:
    if pd.isna(cast_str):
        return 0
    return len([c for c in str(cast_str).split(",") if c.strip()])


def _main_country(country_str) -> str:
    if pd.isna(country_str):
        return "Unknown"
    return str(country_str).split(",")[0].strip()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived numeric/categorical columns used as model features."""
    df = df.copy()
    df["duration_value"] = df.apply(_parse_duration, axis=1)
    df["cast_size"] = df["cast"].apply(_count_cast)
    df["country_main"] = df["country"].apply(_main_country)
    df["years_since_release"] = 2021 - df["release_year"]  # dataset max year ~2021
    df["rating"] = df["rating"].fillna("Unknown")
    df["description"] = df["description"].fillna("")
    return df


def build_target(df: pd.DataFrame) -> pd.Series:
    """Derive the broad-genre target label from the `listed_in` column."""
    return df["listed_in"].astype(str).apply(get_primary_broad_genre)


def prepare_dataset(df: pd.DataFrame, min_class_count: int = 40):
    """
    Full preparation: engineer features, build target, drop rare classes
    (too few samples to learn/evaluate reliably), and return X (raw feature
    frame, before the ColumnTransformer) and y.
    """
    df = engineer_features(df)
    y = build_target(df)

    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= min_class_count].index
    mask = y.isin(valid_classes)

    X = df.loc[mask, NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE]].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)
    return X, y


# Preprocessing pipeline (ColumnTransformer)
def build_preprocessor(max_text_features: int = 300) -> ColumnTransformer:
    """
    Build the sklearn ColumnTransformer that:
      - median-imputes + scales numeric features
      - most-frequent-imputes + one-hot-encodes categorical features
      - TF-IDF vectorizes the free-text description
    """
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
    ])

    text_pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=max_text_features, stop_words="english")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
            ("text", text_pipe, TEXT_FEATURE),
        ]
    )
    return preprocessor
