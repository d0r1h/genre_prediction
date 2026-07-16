"""
evaluate.py
-----------
CLI entrypoint to evaluate one or more trained models on the held-out test
split, print metrics, and save comparison plots + a confusion matrix.

Example:
    python src/evaluate.py --model_path outputs/models/random_forest.pkl --data_path data/tv-shows.csv
    python src/evaluate.py --model_path all --data_path data/tv-shows.csv   # evaluate every saved model
"""
import argparse
import os
import glob
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize

sns.set_style("whitegrid")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained genre classifier(s).")
    parser.add_argument("--model_path", type=str, default="all",
                         help="Path to a single .pkl model, or 'all' to evaluate every model in outputs/models/.")
    parser.add_argument("--data_path", type=str, default="data/tv-shows.csv",
                         help="Kept for CLI-interface consistency; the saved train/test split is used directly.")
    parser.add_argument("--output_dir", type=str, default="outputs")
    return parser.parse_args()


def compute_metrics(y_test, y_pred, y_proba, classes) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted"),
        "precision_macro": precision_score(y_test, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, y_pred, average="macro", zero_division=0),
    }
    try:
        y_test_bin = label_binarize(y_test, classes=classes)
        metrics["roc_auc_macro_ovr"] = roc_auc_score(
            y_test_bin, y_proba, average="macro", multi_class="ovr"
        )
    except Exception as e:
        metrics["roc_auc_macro_ovr"] = None
        print(f"  (ROC-AUC could not be computed: {e})")
    return metrics


def plot_confusion_matrix(y_test, y_pred, classes, model_name, output_dir):
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(output_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_model_comparison(all_metrics: dict, output_dir):
    """all_metrics: {model_name: {metric: value}}"""
    df = pd.DataFrame(all_metrics).T
    plot_cols = ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
    df_plot = df[plot_cols]

    ax = df_plot.plot(kind="bar", figsize=(10, 6), colormap="viridis")
    plt.title("Model Performance Comparison")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def evaluate_single_model(model_path, X_test, y_test, output_dir):
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    print(f"\n=== Evaluating {model_name} ===")
    model = joblib.load(model_path)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    classes = model.classes_

    metrics = compute_metrics(y_test, y_pred, y_proba, classes)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n  Classification report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    cm_path = plot_confusion_matrix(y_test, y_pred, classes, model_name, fig_dir)
    print(f"  Saved confusion matrix -> {cm_path}")

    return metrics


def main():
    args = parse_args()
    split_path = os.path.join(args.output_dir, "train_test_split.pkl")
    split = joblib.load(split_path)
    X_test, y_test = split["X_test"], split["y_test"]

    if args.model_path == "all":
        model_paths = sorted(glob.glob(os.path.join(args.output_dir, "models", "*.pkl")))
    else:
        model_paths = [args.model_path]

    all_metrics = {}
    for path in model_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        all_metrics[name] = evaluate_single_model(path, X_test, y_test, args.output_dir)

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nSaved metrics summary -> {metrics_path}")

    if len(all_metrics) > 1:
        fig_dir = os.path.join(args.output_dir, "figures")
        cmp_path = plot_model_comparison(all_metrics, fig_dir)
        print(f"Saved model comparison plot -> {cmp_path}")


if __name__ == "__main__":
    main()
