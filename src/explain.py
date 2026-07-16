"""
explain.py
----------
Model explainability: extracts and visualizes which input features drive
predictions for each trained model.

- Random Forest -> built in impurity-based feature importances
- Logistic Regression -> per class coefficient magnitudes
- Both -> permutation importance (model-agnostic, computed on held-out test data)

Example:
    python src/explain.py --output_dir outputs
"""
import argparse
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection import permutation_importance

sns.set_style("whitegrid")


def parse_args():
    parser = argparse.ArgumentParser(description="Explain trained genre classifier(s).")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--top_n", type=int, default=20)
    return parser.parse_args()


def get_feature_names(preprocessor) -> np.ndarray:
    """Recover human-readable feature names after the ColumnTransformer."""
    return preprocessor.get_feature_names_out()


def plot_builtin_importance(model, model_name, fig_dir, top_n=20):
    preprocessor = model.named_steps["preprocess"]
    clf = model.named_steps["clf"]
    feature_names = get_feature_names(preprocessor)

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
        title = f"{model_name}: Top {top_n} Feature Importances (impurity-based)"
    elif hasattr(clf, "coef_"):
        # Average absolute coefficient magnitude across classes
        importances = np.abs(clf.coef_).mean(axis=0)
        title = f"{model_name}: Top {top_n} Feature Importances (|coef| avg across classes)"
    else:
        print(f"  No built-in importance available for {model_name}")
        return None

    order = np.argsort(importances)[::-1][:top_n]
    top_features = feature_names[order]
    top_values = importances[order]

    plt.figure(figsize=(9, 8))
    sns.barplot(x=top_values, y=top_features, color="#4C72B0")
    plt.title(title)
    plt.xlabel("Importance")
    plt.tight_layout()
    path = os.path.join(fig_dir, f"feature_importance_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_permutation_importance(model, model_name, X_test, y_test, fig_dir, top_n=15, n_repeats=5, seed=42):
    """
    Model-agnostic importance: shuffles each RAW input column (release_year,
    rating, type, description, etc.) and measures the drop in accuracy.
    This is more interpretable to a general audience than encoded/TF-IDF
    feature-level importance since it operates on the original columns.
    """
    result = permutation_importance(
        model, X_test, y_test, n_repeats=n_repeats, random_state=seed, scoring="accuracy", n_jobs=-1
    )
    importances = pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)
    importances = importances.head(top_n)

    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances.values, y=importances.index, color="#55A868")
    plt.title(f"{model_name}: Permutation Importance (raw input columns)")
    plt.xlabel("Mean Accuracy Drop When Shuffled")
    plt.tight_layout()
    path = os.path.join(fig_dir, f"permutation_importance_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path, importances


def plot_top_tfidf_terms_per_class(model, model_name, fig_dir, top_n=8):
    """For linear models, show the strongest TF-IDF text terms per genre class."""
    preprocessor = model.named_steps["preprocess"]
    clf = model.named_steps["clf"]
    if not hasattr(clf, "coef_"):
        return None

    feature_names = get_feature_names(preprocessor)
    text_mask = np.array([f.startswith("text__") for f in feature_names])
    text_feature_names = np.array([f.replace("text__", "") for f in feature_names[text_mask]])

    classes = clf.classes_
    n_classes = len(classes)
    fig, axes = plt.subplots(nrows=int(np.ceil(n_classes / 3)), ncols=3, figsize=(15, 4 * int(np.ceil(n_classes / 3))))
    axes = axes.flatten()

    for i, cls in enumerate(classes):
        coefs = clf.coef_[i][text_mask]
        order = np.argsort(coefs)[::-1][:top_n]
        top_terms = text_feature_names[order]
        top_vals = coefs[order]
        sns.barplot(x=top_vals, y=top_terms, ax=axes[i], color="#C44E52")
        axes[i].set_title(cls, fontsize=10)
        axes[i].set_xlabel("")

    for j in range(n_classes, len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"{model_name}: Top Description Keywords Driving Each Genre", y=1.00)
    plt.tight_layout()
    path = os.path.join(fig_dir, f"top_keywords_per_genre_{model_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def main():
    args = parse_args()
    fig_dir = os.path.join(args.output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    split = joblib.load(os.path.join(args.output_dir, "train_test_split.pkl"))
    X_test, y_test = split["X_test"], split["y_test"]

    import glob
    model_paths = sorted(glob.glob(os.path.join(args.output_dir, "models", "*.pkl")))

    for path in model_paths:
        model_name = os.path.splitext(os.path.basename(path))[0]
        print(f"\n=== Explaining {model_name} ===")
        model = joblib.load(path)

        p1 = plot_builtin_importance(model, model_name, fig_dir, top_n=args.top_n)
        if p1:
            print(f"  Saved built-in importance plot -> {p1}")

        p2, importances = plot_permutation_importance(model, model_name, X_test, y_test, fig_dir)
        print(f"  Saved permutation importance plot -> {p2}")
        print(f"  Top raw features by permutation importance:\n{importances}")

        p3 = plot_top_tfidf_terms_per_class(model, model_name, fig_dir)
        if p3:
            print(f"  Saved per-genre keyword plot -> {p3}")


if __name__ == "__main__":
    main()
