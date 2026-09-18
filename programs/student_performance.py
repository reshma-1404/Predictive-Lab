import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from common import load_data, make_preprocessor, split_data


def main():
    parser = argparse.ArgumentParser(description="Tune a student performance classifier")
    parser.add_argument("data", help="CSV dataset path")
    parser.add_argument("--target", default="performance", help="Target column")
    args = parser.parse_args()

    features, target = load_data(args.data, args.target)
    x_train, x_test, y_train, y_test = split_data(features, target, classification=True)
    pipeline = Pipeline([
        ("preprocess", make_preprocessor(features)),
        ("model", RandomForestClassifier(random_state=42, class_weight="balanced")),
    ])
    search = GridSearchCV(
        pipeline,
        {"model__n_estimators": [100, 200], "model__max_depth": [None, 10, 20]},
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )
    search.fit(x_train, y_train)
    predictions = search.predict(x_test)
    print(f"Best parameters: {search.best_params_}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    main()