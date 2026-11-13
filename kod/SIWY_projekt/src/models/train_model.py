from itertools import permutations
from math import factorial

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from tabpfn import TabPFNRegressor


def generate_tsp_dataset(
    n_cities: int,
    random_state: int,
    sample_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample unique tours for one Euclidean TSP instance.

    Each feature row is a successor permutation: column i contains the city
    visited immediately after city i. The target is the total closed-tour
    distance. Reversed versions of symmetric tours are included only once.
    """
    rng = np.random.default_rng(random_state)
    # Generate random coordinates for the cities in 2D space
    coordinates = rng.random((n_cities, 2))
    distances = np.zeros((n_cities, n_cities))
    for city in range(n_cities):
        for other_city in range(n_cities):
            x_difference = coordinates[city, 0] - coordinates[other_city, 0]
            y_difference = coordinates[city, 1] - coordinates[other_city, 1]
            distances[city, other_city] = np.sqrt(
                x_difference**2 + y_difference**2
            )

    features: list[list[int]] = []
    costs: list[float] = []
    routes: list[list[int]] = []
    all_routes = [
        route
        for route in permutations(range(1, n_cities))
        if route[0] < route[-1]
    ]
    route_order = rng.permutation(len(all_routes))
    sampled_routes = [all_routes[index] for index in route_order[:sample_size]]

    for remaining_cities in sampled_routes:
        route = (0, *remaining_cities)
        successors = [0] * n_cities
        route_cost = 0.0
        for city, next_city in zip(route, route[1:] + route[:1]):
            successors[city] = next_city
            route_cost += distances[city, next_city]

        features.append(successors)
        costs.append(route_cost)
        routes.append(list(route))

    return (
        np.asarray(features, dtype=int),
        np.asarray(costs, dtype=float),
        np.asarray(routes, dtype=int),
    )


def run_experiment(
    n_cities: int,
    random_state: int,
    sample_size: int | None = None,
) -> None:
    features, costs, routes = generate_tsp_dataset(
        n_cities,
        random_state,
        sample_size,
    )
    population_size = factorial(n_cities - 1) // 2
    (
        features_train,
        features_test,
        costs_train,
        costs_test,
        _,
        routes_test,
    ) = train_test_split(
        features,
        costs,
        routes,
        test_size=0.25,
        random_state=random_state,
    )

    tree_regressor = make_pipeline(
        ColumnTransformer(
            [
                (
                    "one_hot",
                    OneHotEncoder(handle_unknown="ignore"),
                    list(range(n_cities)),
                )
            ]
        ),
        DecisionTreeRegressor(random_state=random_state),
    )
    tree_regressor.fit(features_train, costs_train)
    tree_predictions = tree_regressor.predict(features_test)

    linear_regression = make_pipeline(
        ColumnTransformer(
            [
                (
                    "one_hot",
                    OneHotEncoder(handle_unknown="ignore"),
                    list(range(n_cities)),
                )
            ]
        ),
        LinearRegression(),
    )
    linear_regression.fit(features_train, costs_train)
    linear_predictions = linear_regression.predict(features_test)

    model = TabPFNRegressor(
        categorical_features_indices=list(range(n_cities)),
        random_state=random_state,
    )
    model.fit(features_train, costs_train)
    predictions = model.predict(features_test)

    print("Regresja kosztu tras dla problemu komiwojażera (TabPFN)")
    print(f"Liczba miast: {n_cities}")
    print(f"Liczba różnych tras w populacji: {population_size}")
    print(f"Liczba wylosowanych tras: {len(features)}")
    print(f"Liczba przykładów treningowych: {len(features_train)}")
    print(f"Liczba przykładów testowych: {len(features_test)}")
    print(f"MAE TabPFN: {mean_absolute_error(costs_test, predictions):.4f}")
    print(f"R² TabPFN: {r2_score(costs_test, predictions):.4f}")
    print(
        "MAE drzewa decyzyjnego: "
        f"{mean_absolute_error(costs_test, tree_predictions):.4f}"
    )
    print(
        "R² drzewa decyzyjnego: "
        f"{r2_score(costs_test, tree_predictions):.4f}"
    )
    print(
        "MAE regresji liniowej: "
        f"{mean_absolute_error(costs_test, linear_predictions):.4f}"
    )
    print(
        "R² regresji liniowej: "
        f"{r2_score(costs_test, linear_predictions):.4f}"
    )
    print(
        "Przykładowe trasy: koszt rzeczywisty -> TabPFN "
        "-> drzewo -> regresja liniowa"
    )
    for route, actual, predicted, tree_predicted, linear_predicted in zip(
        routes_test[:5],
        costs_test[:5],
        predictions[:5],
        tree_predictions[:5],
        linear_predictions[:5],
    ):
        displayed_route = " -> ".join(map(str, route))
        print(
            f"{displayed_route} -> {route[0]}: "
            f"{actual:.4f} -> {predicted:.4f} -> "
            f"{tree_predicted:.4f} -> {linear_predicted:.4f}"
        )


def main() -> None:
    # Compare increasing sample sizes from the same 7-city route population.
    for sample_size in (40, 100, 200, 360):
        run_experiment(7, 42, sample_size)


if __name__ == "__main__":
    main()
