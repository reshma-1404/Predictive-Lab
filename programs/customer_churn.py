import argparse

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline

from common import load_data, make_preprocessor, split_data


def main():
    parser = argparse.ArgumentParser(description="Predict customer churn")
    parser.add_argument("data", help="CSV dataset path")
    parser.add_argument("--target", default="Churn", help="Churn label column")
    args = parser.parse_args()

    features, target = load_data(args.data, args.target)
    x_train, x_test, y_train, y_test = split_data(features, target, classification=True)
    pipeline = Pipeline([
        ("preprocess", make_preprocessor(features)),
        ("model", RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.3f}")
    print(classification_report(y_test, predictions))


if __name__ == "__main__":
    main()