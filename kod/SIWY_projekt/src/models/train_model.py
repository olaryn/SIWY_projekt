import argparse

from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier, TabPFNRegressor


def test_classification():
    features, target = make_classification(
        n_samples=120,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=42,
        stratify=target,
    )

    model = TabPFNClassifier()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Test klasyfikacji TabPFN")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(f"Pierwsze przewidywania: {predictions[:5]}")


def test_regression():
    features, target = make_regression(
        n_samples=120,
        n_features=6,
        n_informative=4,
        noise=10,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.25,
        random_state=42,
    )

    model = TabPFNRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Test regresji TabPFN")
    print(f"MAE: {mean_absolute_error(y_test, predictions):.3f}")
    print(f"Pierwsze przewidywania: {predictions[:5]}")


def main():
    parser = argparse.ArgumentParser(
        description="Uruchamia demonstracyjny test modelu TabPFN."
    )
    parser.add_argument(
        "--task",
        choices=("classification", "regression", "both"),
        default="classification",
        help="Rodzaj testu (domyślnie: classification).",
    )
    task = parser.parse_args().task

    if task in ("classification", "both"):
        test_classification()
    if task in ("regression", "both"):
        test_regression()


if __name__ == "__main__":
    main()
