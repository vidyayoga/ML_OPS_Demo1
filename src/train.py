import argparse
from pathlib import Path

import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# =====================================================
# MLflow Configuration
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MLFLOW_DB = PROJECT_ROOT / "mlflow.db"

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB}"
)


def train_model(C, max_iter, test_size):

    # =================================================
    # 1. Load Dataset
    # =================================================

    DATA_PATH = PROJECT_ROOT / "data" / "breast_cancer.csv"

    RANDOM_STATE = 42

    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")


    # =================================================
    # 2. Separate Features & Target
    # =================================================

    X = df.drop("target", axis=1)

    y = df["target"]


    # =================================================
    # 3. Train-Test Split
    # =================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y
    )


    # =================================================
    # 4. Create ML Pipeline
    # =================================================

    model = Pipeline([
        ("scaler", StandardScaler()),

        (
            "classifier",

            LogisticRegression(
                C=C,
                max_iter=max_iter
            )
        )
    ])


    # =================================================
    # 5. Train Model
    # =================================================

    model.fit(X_train, y_train)


    # =================================================
    # 6. Predictions
    # =================================================

    predictions = model.predict(X_test)


    # =================================================
    # 7. Calculate Metrics
    # =================================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions
    )

    recall = recall_score(
        y_test,
        predictions
    )

    f1 = f1_score(
        y_test,
        predictions
    )


    # =================================================
    # 8. Log Parameters to MLflow
    # =================================================

    mlflow.log_param(
        "model",
        "LogisticRegression"
    )

    mlflow.log_param(
        "C",
        C
    )

    mlflow.log_param(
        "max_iter",
        max_iter
    )

    mlflow.log_param(
        "test_size",
        test_size
    )

    mlflow.log_param(
        "random_state",
        RANDOM_STATE
    )


    # =================================================
    # 9. Log Metrics to MLflow
    # =================================================

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )


    # =================================================
    # 10. Log Model to MLflow
    # =================================================

    mlflow.sklearn.log_model(
        model,
        name="breast_cancer_model"
    )


    # =================================================
    # 11. Print Results
    # =================================================

    print("\nModel Results")

    print("----------------------")

    print(f"C         : {C}")

    print(f"Accuracy  : {accuracy:.4f}")

    print(f"Precision : {precision:.4f}")

    print(f"Recall    : {recall:.4f}")

    print(f"F1 Score  : {f1:.4f}")


def main():

    # =================================================
    # Create MLflow Experiment
    # =================================================

    mlflow.set_experiment(
        "Breast Cancer Classification"
    )


    # =================================================
    # Command Line Arguments
    # =================================================

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--C",
        type=float,
        default=1.0
    )


    parser.add_argument(
        "--max_iter",
        type=int,
        default=500
    )


    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2
    )


    args = parser.parse_args()


    # =================================================
    # Start MLflow Run
    # =================================================

    with mlflow.start_run():

        train_model(
            C=args.C,
            max_iter=args.max_iter,
            test_size=args.test_size
        )


if __name__ == "__main__":
    main()
