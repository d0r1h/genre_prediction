"""
train.py
--------
CLI entrypoint to train a genre-prediction model end to end.

Example:
    python src/train.py --data_path data/tv-shows.csv --model random_forest --seed 42
    python src/train.py --data_path data/tv-shows.csv --model logistic_regression --seed 42
    python src/train.py --data_path data/tv-shows.csv --model all --seed 42   # trains every model
"""
import argparse
import os
import joblib
from sklearn.model_selection import train_test_split

from data_preprocessing import load_data, run_eda, prepare_dataset
from models import get_model, get_model_registry


def parse_args():
    parser = argparse.ArgumentParser(description="Train a TV/Movie genre classifier.")
    parser.add_argument("--data_path", type=str, default="data/tv-shows.csv")
    parser.add_argument("--model", type=str, default="all",
                         help="Model name (logistic_regression, random_forest) or 'all'.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--run_eda", action="store_true", default=True)
    return parser.parse_args()


def train_and_save(model_name: str, X_train, y_train, seed: int, output_dir: str):
    print(f"\n=== Training model: {model_name} ===")
    model = get_model(model_name, seed=seed)
    model.fit(X_train, y_train)

    model_dir = os.path.join(output_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.pkl")
    joblib.dump(model, model_path)
    print(f"Saved trained model -> {model_path}")
    return model_path


def main():
    args = parse_args()

    print(f"Loading data from {args.data_path} ...")
    df = load_data(args.data_path)
    print(f"Raw shape: {df.shape}")

    if args.run_eda:
        eda_dir = os.path.join(args.output_dir, "figures")
        print(f"Running EDA, saving plots to {eda_dir} ...")
        summary = run_eda(df, eda_dir)
        print(f"Missing values: {summary['missing_values']}")
        print(f"Type counts: {summary['type_counts']}")
        print(f"Target class counts: {summary['genre_counts']}")

    X, y = prepare_dataset(df)
    print(f"Prepared feature matrix: {X.shape}, target classes: {sorted(y.unique())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

    # Persist the split so evaluate.py can reuse the exact same test set.
    split_path = os.path.join(args.output_dir, "train_test_split.pkl")
    os.makedirs(args.output_dir, exist_ok=True)
    joblib.dump(
        {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test},
        split_path,
    )
    print(f"Saved train/test split -> {split_path}")

    if args.model == "all":
        model_names = list(get_model_registry().keys())
    else:
        model_names = [args.model]

    for name in model_names:
        train_and_save(name, X_train, y_train, args.seed, args.output_dir)

    print("\nTraining complete. Run evaluate.py next to score the model(s).")


if __name__ == "__main__":
    main()
