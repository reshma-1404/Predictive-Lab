import argparse

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from common import load_data, make_preprocessor, split_data


def main():
    parser = argparse.ArgumentParser(description="Tune a house price regression model")
    parser.add_argument("data", help="CSV dataset path")
    parser.add_argument("--target", default="price", help="Target column")
    args = parser.parse_args()

    features, target = load_data(args.data, args.target)
    x_train, x_test, y_train, y_test = split_data(features, target)
    pipeline = Pipeline([
        ("preprocess", make_preprocessor(features)),
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
    ])
    search = RandomizedSearchCV(
        pipeline,
        {
            "model__n_estimators": [100, 200, 400],
            "model__max_depth": [None, 10, 20, 30],
            "model__min_samples_split": [2, 5, 10],
        },
        n_iter=10,
        cv=5,
        scoring="neg_mean_absolute_error",
        random_state=42,
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    predictions = search.predict(x_test)
    print(f"Best parameters: {search.best_params_}")
    print(f"MAE: {mean_absolute_error(y_test, predictions):.2f}")
    print(f"R2: {r2_score(y_test, predictions):.3f}")


if __name__ == "__main__":
    main()