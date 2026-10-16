"""Train, tune, evaluate and track candidate models; package the best one.

Usage:
    python -m src.train
    python -m src.train --cv-folds 5 --tracking-uri sqlite:///mlflow.db
"""
import argparse
import json
import logging
import os
import platform
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import sklearn  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (  # noqa: E402
    GridSearchCV, StratifiedKFold, cross_validate, train_test_split,
)
from sklearn.pipeline import Pipeline  # noqa: E402

from src import config  # noqa: E402
from src.data import load_clean, prepare_clean_dataset, split_xy  # noqa: E402
from src.features import build_preprocessor, get_feature_names  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train")

CV_SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def candidate_models() -> dict:
    """Model name -> (estimator, hyper-parameter grid)."""
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE),
            {"clf__C": [0.01, 0.1, 1, 10]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=config.RANDOM_STATE),
            {
                "clf__n_estimators": [100, 300],
                "clf__max_depth": [3, 5, None],
                "clf__min_samples_leaf": [1, 3],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=config.RANDOM_STATE),
            {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.05, 0.1],
                "clf__max_depth": [2, 3],
            },
        ),
    }


def setup_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Point MLflow at the tracking store and keep artifacts in <repo>/mlartifacts."""
    mlflow.set_tracking_uri(tracking_uri)
    if tracking_uri.startswith("sqlite") and mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            experiment_name, artifact_location=(config.ROOT / "mlartifacts").as_uri()
        )
    mlflow.set_experiment(experiment_name)


def make_pipeline(estimator) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("clf", estimator)])


def evaluate_on_test(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "test_accuracy": accuracy_score(y_test, pred),
        "test_precision": precision_score(y_test, pred),
        "test_recall": recall_score(y_test, pred),
        "test_f1": f1_score(y_test, pred),
        "test_roc_auc": roc_auc_score(y_test, proba),
    }


def save_plots(name, model, X_test, y_test, out_dir):
    """ROC curve, confusion matrix and feature importance for one model."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_title(f"ROC curve - {name}")
    p = out_dir / f"roc_{name}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test, ax=ax, cmap="Blues",
        display_labels=["No disease", "Disease"], colorbar=False,
    )
    ax.set_title(f"Confusion matrix - {name}")
    p = out_dir / f"confusion_{name}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)

    clf = model.named_steps["clf"]
    names = get_feature_names(model.named_steps["preprocess"])
    if hasattr(clf, "feature_importances_"):
        values, label = clf.feature_importances_, "Importance"
    else:
        values, label = clf.coef_.ravel(), "Coefficient"
    order = np.argsort(np.abs(values))[::-1][:12]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(names[order][::-1], values[order][::-1], color="#2a6f97")
    ax.set_xlabel(label)
    ax.set_title(f"Top features - {name}")
    p = out_dir / f"features_{name}.png"
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)
    paths.append(p)
    return paths


def run(args) -> dict:
    if not config.CLEAN_PATH.exists():
        logger.info("Clean dataset missing; building it from raw data")
        prepare_clean_dataset()
    df = load_clean()
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    logger.info("Train %s | Test %s | positive rate %.3f", X_train.shape, X_test.shape, y.mean())

    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=config.RANDOM_STATE)

    setup_mlflow(args.tracking_uri, config.MLFLOW_EXPERIMENT)

    results, fitted = {}, {}
    with mlflow.start_run(run_name="model-comparison") as parent:
        mlflow.log_params({
            "n_rows": len(df), "n_features": X.shape[1], "test_size": config.TEST_SIZE,
            "cv_folds": args.cv_folds, "random_state": config.RANDOM_STATE,
            "sklearn_version": sklearn.__version__,
        })
        mlflow.log_artifact(str(config.CLEAN_PATH), artifact_path="data")

        for name, (estimator, grid) in candidate_models().items():
            with mlflow.start_run(run_name=name, nested=True):
                logger.info("Tuning %s", name)
                search = GridSearchCV(
                    make_pipeline(estimator), grid, scoring="roc_auc", cv=cv,
                    n_jobs=-1, refit=True,
                )
                search.fit(X_train, y_train)
                best = search.best_estimator_

                # CV metrics of the tuned model on the training split
                cv_res = cross_validate(best, X_train, y_train, cv=cv, scoring=CV_SCORING)
                cv_metrics = {}
                for k in CV_SCORING:
                    cv_metrics[f"cv_{k}_mean"] = float(cv_res[f"test_{k}"].mean())
                    cv_metrics[f"cv_{k}_std"] = float(cv_res[f"test_{k}"].std())
                held_out = evaluate_on_test(best, X_test, y_test)

                mlflow.log_param("model_type", name)
                mlflow.log_params({k.replace("clf__", ""): v
                                   for k, v in search.best_params_.items()})
                mlflow.log_metric("grid_best_cv_roc_auc", search.best_score_)
                mlflow.log_metrics(cv_metrics)
                mlflow.log_metrics(held_out)
                for p in save_plots(name, best, X_test, y_test, config.FIGURES_DIR):
                    mlflow.log_artifact(str(p), artifact_path="plots")

                signature = infer_signature(X_test, best.predict_proba(X_test))
                mlflow.sklearn.log_model(
                    best, name="model", signature=signature, input_example=X_test.head(3),
                    serialization_format="cloudpickle",
                )
                results[name] = {
                    "best_params": search.best_params_,
                    "cv": cv_metrics, "test": held_out,
                    "run_id": mlflow.active_run().info.run_id,
                }
                fitted[name] = best
                logger.info("%s | CV AUC %.3f | test AUC %.3f", name,
                            cv_metrics["cv_roc_auc_mean"], held_out["test_roc_auc"])

        # model selection uses CV ROC-AUC only; the test set is never used to choose
        best_name = max(results, key=lambda n: results[n]["cv"]["cv_roc_auc_mean"])
        mlflow.log_param("selected_model", best_name)
        mlflow.log_metric("selected_test_roc_auc", results[best_name]["test"]["test_roc_auc"])

        fig, ax = plt.subplots(figsize=(6, 5))
        for name, model in fitted.items():
            RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.set_title("ROC comparison (held-out test set)")
        fig.tight_layout()
        cmp_path = config.FIGURES_DIR / "roc_comparison.png"
        fig.savefig(cmp_path, dpi=150)
        plt.close(fig)
        mlflow.log_artifact(str(cmp_path), artifact_path="plots")

        final = fitted[best_name]
        signature = infer_signature(X_test, final.predict_proba(X_test))
        info = mlflow.sklearn.log_model(
            final, name="final_model", signature=signature,
            input_example=X_test.head(3), serialization_format="cloudpickle",
            registered_model_name=config.MLFLOW_MODEL_NAME,
        )

        config.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final, config.MODEL_PATH)
        metadata = {
            "model_name": best_name,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "mlflow_run_id": parent.info.run_id,
            "mlflow_model_uri": info.model_uri,
            "feature_columns": config.FEATURE_COLUMNS,
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
            "best_params": results[best_name]["best_params"],
            "cv_metrics": results[best_name]["cv"],
            "test_metrics": results[best_name]["test"],
            "all_models": results,
        }
        config.METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str))
        mlflow.log_artifact(str(config.METADATA_PATH))
        logger.info("Selected %s; saved %s", best_name, config.MODEL_PATH)
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-folds", type=int, default=config.CV_FOLDS)
    parser.add_argument(
        "--tracking-uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{config.ROOT / 'mlflow.db'}"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
