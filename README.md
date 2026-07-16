# TV Show / Movie Genre Prediction

Predicts the genre of a Netflix/Disney title (Movie or TV Show) from its
metadata (type, cast, country, rating, duration, release year, platform)
and text description, using traditional machine learning models.

## 1. Project Structure

```
genre_prediction/
├── data/
│   └── tv-shows.csv              # input dataset
├── src/
│   ├── genre_map.py              # raw -> broad genre label mapping
│   ├── data_preprocessing.py     # loading, EDA, feature engineering, ColumnTransformer
│   ├── models.py                 # model factory (Logistic Regression, Random Forest)
│   ├── train.py                  # CLI: train model(s), save train/test split + models
│   ├── evaluate.py               # CLI: evaluate model(s), metrics + confusion matrices
│   └── explain.py                # CLI: feature importance / explainability plots
├── outputs/
│   ├── figures/                  # all PNG plots (EDA, evaluation, explainability)
│   ├── models/                   # saved .pkl model pipelines
│   ├── train_test_split.pkl      # fixed train/test split (shared by train/evaluate/explain)
│   └── metrics.json              # evaluation metrics summary
├── requirements.txt
└── README.md
```

## 2. Setup & Run

```bash
pip install -r requirements.txt

# Train both models (Logistic Regression + Random Forest), runs EDA as a side effect
python src/train.py --data_path data/tv-shows.csv --model all --seed 42

# Evaluate all trained models against the held-out test set
python src/evaluate.py --model_path all --data_path data/tv-shows.csv

# Generate explainability plots (feature importance, permutation importance, keyword analysis)
python src/explain.py --output_dir outputs
```

To train/evaluate a single model instead of both:
```bash
python src/train.py --data_path data/tv-shows.csv --model random_forest --seed 42
python src/evaluate.py --model_path outputs/models/random_forest.pkl
```

All scripts are run from the `genre_prediction/` root directory.

## 3. Approach

### 3.1 Target definition
The raw dataset's `listed_in` column is a **comma-separated, multi-label**
list of very granular genre tags (84 unique values, e.g. "TV Dramas",
"Dramas", "Action & Adventure", "Action-Adventure" — the same concept
split by naming convention). To turn this into a learnable, single-label
classification problem:

1. The **first-listed genre** in `listed_in` is taken as the primary genre
   for each title (the dataset orders tags by relevance).
2. That raw tag is mapped to one of **~13 broad genre classes** (Drama,
   Comedy, Action & Adventure, Documentary, Kids & Family, Anime &
   Animation, Crime, Horror & Thriller, International, Reality & Talk
   Show, Romance, Sci-Fi & Fantasy, Other) via keyword matching
   (`src/genre_map.py`).
3. Classes with fewer than 40 samples (Romance, Sci-Fi & Fantasy) are
   dropped - too few examples to learn or evaluate reliably — leaving
   **11 target classes**.

### 3.2 Features
| Feature | Type | Notes |
|---|---|---|
| `release_year`, `years_since_release` | numeric | scaled |
| `duration_value` | numeric | minutes (movies) or number of seasons (TV), parsed from `duration` |
| `cast_size` | numeric | number of listed cast members |
| `type` | categorical | Movie / TV Show |
| `rating` | categorical | e.g. TV-MA, PG-13 |
| `platform` | categorical | Netflix / Disney |
| `country_main` | categorical | first listed country |
| `description` | text | TF-IDF vectorized (300 terms, English stop words removed) |

Preprocessing (`build_preprocessor` in `data_preprocessing.py`) is a single
`ColumnTransformer`: median imputation + scaling for numeric columns,
most-frequent imputation + one-hot encoding for categoricals, and TF-IDF
for the description combined into one reusable pipeline object.

### 3.3 Models
Two traditional ML models were trained, both wrapped in the same
preprocessing pipeline for a fair comparison:
- **Logistic Regression** (`class_weight="balanced"`, `max_iter=2000`)
- **Random Forest** (`n_estimators=300`, `class_weight="balanced"`)

Both use `class_weight="balanced"` to compensate for the mild class
imbalance (Comedy/Drama vs. Reality & Talk Show).

### 3.4 Evaluation
80/20 stratified train/test split (`seed=42`). Metrics: accuracy, macro/weighted
F1, macro precision/recall, and macro one-vs-rest ROC-AUC  plus a full
per-class classification report and confusion matrix for each model.
See `outputs/metrics.json` and `outputs/figures/model_comparison.png`.

### 3.5 Explainability
- **Built-in importance**: Random Forest impurity based importances;
  Logistic Regression average absolute coefficient magnitude per feature.
- **Permutation importance**: model-agnostic shuffles each *raw* input
  column and measures the resulting drop in test accuracy. This is the
  most interpretable view since it operates on original columns (`rating`,
  `type`, `description`, ...) rather than encoded/TF-IDF features.
- **Per-genre keyword analysis**: for the linear model, the TF-IDF terms
  with the strongest positive coefficient for each genre class are
  plotted, showing which words in a title's description push it toward
  that genre.

## 4. Results Summary

See the full **Model Evaluation & Explainability Report** (`.docx`) for
metrics tables, comparison charts, confusion matrices, and a written
recommendation.
