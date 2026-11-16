"""A small, interpretable baseline model for the synthetic labels."""

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEATURE_COLUMNS


MODEL_NAMES = ["Logistic Regression", "Random Forest", "Gradient Boosting"]


def build_models() -> dict[str, object]:
    """Create three contrasting models for a small classroom comparison."""
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.08,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        ),
    }


def train_model_suite(data: pd.DataFrame) -> tuple[dict, pd.DataFrame, dict]:
    """Compare models on a shared holdout and 5-fold validation of demo labels."""
    x = data[FEATURE_COLUMNS]
    y = data["risk_label"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=42, stratify=y
    )
    labels = sorted(y.unique())
    models = build_models()
    comparisons = []
    diagnostics = {}

    for name, model in models.items():
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        fold_scores = cross_val_score(
            build_models()[name],
            x,
            y,
            cv=5,
            scoring="f1_weighted",
            n_jobs=1,
        )
        importance = permutation_importance(
            model,
            x_test,
            y_test,
            scoring="f1_weighted",
            n_repeats=5,
            random_state=42,
            n_jobs=1,
        )
        comparisons.append(
            {
                "Model": name,
                "Holdout accuracy": accuracy_score(y_test, predicted),
                "Balanced accuracy": balanced_accuracy_score(y_test, predicted),
                "Weighted F1": f1_score(y_test, predicted, average="weighted"),
                "5-fold F1 mean": fold_scores.mean(),
                "5-fold F1 spread": fold_scores.std(),
            }
        )
        diagnostics[name] = {
            "confusion": pd.DataFrame(
                confusion_matrix(y_test, predicted, labels=labels),
                index=[f"Actual {label}" for label in labels],
                columns=[f"Predicted {label}" for label in labels],
            ),
            "report": pd.DataFrame(
                classification_report(
                    y_test, predicted, labels=labels, output_dict=True, zero_division=0
                )
            ).T,
            "importance": pd.DataFrame(
                {
                    "Input feature": FEATURE_COLUMNS,
                    "Importance": importance.importances_mean,
                    "Variation": importance.importances_std,
                }
            ).sort_values("Importance", ascending=False),
            "test_rows": len(y_test),
        }

    model_table = pd.DataFrame(comparisons).sort_values("Weighted F1", ascending=False)
    return models, model_table, diagnostics


def predict_risk(model, values: dict) -> tuple[str, dict[str, float]]:
    """Return the most likely demo class and each class probability."""
    row = pd.DataFrame([{column: values[column] for column in FEATURE_COLUMNS}])
    probabilities = model.predict_proba(row)[0]
    by_class = dict(zip(model.classes_, probabilities))
    best_class = max(by_class, key=by_class.get)
    return best_class, {label: float(score) for label, score in by_class.items()}
