import argparse

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

from common import load_data, make_preprocessor, split_data


def main():
    parser = argparse.ArgumentParser(description="Predict student admission outcomes")
    parser.add_argument("data", help="CSV dataset path")
    parser.add_argument("--target", default="admitted", help="Target column")
    args = parser.parse_args()

    features, target = load_data(args.data, args.target)
    x_train, x_test, y_train, y_test = split_data(features, target, classification=True)
    pipeline = Pipeline([
        ("preprocess", make_preprocessor(features)),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    main()