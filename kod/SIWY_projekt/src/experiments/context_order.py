import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from tabpfn import TabPFNRegressor

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.train_model import generate_tsp_dataset

def run_experiment(
    n_cities: int,
    random_state: int,
    sample_size: int | None = None,
) -> None:

    X, y, routes = generate_tsp_dataset(
        n_cities=n_cities,
        random_state=random_state,
        sample_size=sample_size,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=random_state,
    )

    rng = np.random.default_rng(123)
    order = rng.permutation(len(X_train))

    X_train_shuffled = X_train[order]
    y_train_shuffled = y_train[order]


    params = dict(
        categorical_features_indices=list(range(X_train.shape[1])),
        random_state=42,
    )

    model_original = TabPFNRegressor(**params)
    model_original.fit(X_train, y_train)
    pred_original = model_original.predict(X_test)

    model_shuffled = TabPFNRegressor(**params)
    model_shuffled.fit(X_train_shuffled, y_train_shuffled)
    pred_shuffled = model_shuffled.predict(X_test)

    difference = np.abs(pred_original - pred_shuffled)

    print(f"MAE — pierwotna kolejność: {mean_absolute_error(y_test, pred_original):.4f}")
    print(f"MAE — przetasowana kolejność: {mean_absolute_error(y_test, pred_shuffled):.4f}")
    print(f"Średnia różnica predykcji: {difference.mean():.4f}")
    print(f"Maksymalna różnica predykcji: {difference.max():.4f}\n")

def main() -> None:
    print(("============= 7 miast ============="))
    # Compare increasing sample sizes from the same 7-city route population.
    for sample_size in (40, 200, 360):
        run_experiment(7, 42, sample_size)

    print(("============= 9 miast ============="))
    # Compare increasing sample sizes from the same 7-city route population.
    for sample_size in (100, 250, 400):
        run_experiment(9, 42, sample_size)

if __name__ == "__main__":
    main()
