"""Train baseline models, log every run to MLflow, and compare them on the test set."""
import hashlib
import json

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from config import (TRAIN_CSV, TEST_CSV, TARGET, RANDOM_STATE, MODEL_EXCLUDE,
                    MLFLOW_DB, EXPERIMENT_NAME, METRICS_JSON)


def file_md5(path):
    """Fingerprint of a data file, so each run records exactly which data it used."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def build_preprocessor(X, scale):
    """Fill missing values and turn text columns into numbers.
    Fitted on training data only, then reused unchanged on the test data."""
    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:                                   # only logistic regression needs scaling
        num_steps.append(("scale", StandardScaler()))

    return ColumnTransformer([
        ("num", Pipeline(num_steps), num_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ])


def get_models(scale_pos_weight):
    """name -> (model, needs_scaling). Class weights handle the 24% / 76% imbalance."""
    return {
        "dummy": (DummyClassifier(strategy="prior"), False),
        "logreg": (LogisticRegression(max_iter=1000, class_weight="balanced"), True),
        "random_forest": (RandomForestClassifier(
            n_estimators=200, min_samples_leaf=20, class_weight="balanced_subsample",
            n_jobs=-1, random_state=RANDOM_STATE), False),
        "xgboost": (XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            n_jobs=-1, random_state=RANDOM_STATE), False),
    }


def evaluate(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]      # probability of default
    pred = (proba >= 0.5).astype(int)              # class decision at 0.5
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    train = pd.read_csv(TRAIN_CSV)
    test = pd.read_csv(TEST_CSV)
    features = [c for c in train.columns if c not in [TARGET] + MODEL_EXCLUDE]
    X_train, y_train = train[features], train[TARGET]
    X_test, y_test = test[features], test[TARGET]

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    print(f"Features: {len(features)} | train rows: {len(train):,} | test rows: {len(test):,}")

    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    all_metrics = {}
    for name, (estimator, scale) in get_models(neg / pos).items():
        pipe = Pipeline([("prep", build_preprocessor(X_train, scale)), ("model", estimator)])
        with mlflow.start_run(run_name=name):
            pipe.fit(X_train, y_train)
            metrics = evaluate(pipe, X_test, y_test)

            mlflow.log_param("model", name)
            mlflow.log_param("n_features", len(features))
            mlflow.log_param("excluded_features", ",".join(MODEL_EXCLUDE))
            mlflow.log_param("train_csv_md5", file_md5(TRAIN_CSV))
            mlflow.log_params({f"model__{k}": v for k, v in estimator.get_params().items()
                               if v is not None})
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            mlflow.sklearn.log_model(pipe, "model",
                                     serialization_format="cloudpickle")

        all_metrics[name] = metrics
        print(f"{name:14s} AUC={metrics['roc_auc']:.3f}  PR-AUC={metrics['pr_auc']:.3f}  "
              f"precision={metrics['precision']:.3f}  recall={metrics['recall']:.3f}  "
              f"f1={metrics['f1']:.3f}  accuracy={metrics['accuracy']:.3f}")

    METRICS_JSON.parent.mkdir(exist_ok=True)
    METRICS_JSON.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()