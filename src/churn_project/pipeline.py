from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "bank-additional"
FULL_DATA_PATH = DATA_DIR / "bank-additional-full.csv"
SAMPLE_DATA_PATH = DATA_DIR / "bank-additional.csv"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "churn_model.joblib"

TARGET_COLUMN = "y"
LEAKAGE_COLUMNS = ["duration"]
RANDOM_STATE = 42


@dataclass
class ModelBundle:
    pipeline: Pipeline
    metrics: dict[str, Any]
    feature_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    shap_summary: pd.DataFrame
    test_sample: pd.DataFrame
    test_labels: pd.Series
    test_probabilities: np.ndarray
    assumptions: dict[str, Any]


def load_bank_data(prefer_full: bool = True) -> pd.DataFrame:
    """Load the semicolon-delimited bank marketing data."""
    data_path = FULL_DATA_PATH if prefer_full and FULL_DATA_PATH.exists() else SAMPLE_DATA_PATH
    if not data_path.exists():
        raise FileNotFoundError(f"No bank dataset found at {FULL_DATA_PATH} or {SAMPLE_DATA_PATH}")
    return pd.read_csv(data_path, sep=";")


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
    """Create a leakage-aware feature matrix and churn-risk target."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Expected target column '{TARGET_COLUMN}' was not found.")

    dropped = [column for column in LEAKAGE_COLUMNS if column in df.columns]
    feature_columns = [column for column in df.columns if column not in dropped + [TARGET_COLUMN]]
    X = df[feature_columns].copy()

    # Portfolio framing: "no" means the customer did not convert/retain in the campaign.
    y = df[TARGET_COLUMN].str.lower().eq("no").astype(int)
    assumptions = {
        "target_column": TARGET_COLUMN,
        "positive_class": "y = no",
        "positive_class_business_name": "churn/non-conversion risk",
        "excluded_leakage_columns": dropped,
        "source_rows": int(len(df)),
        "feature_count": len(feature_columns),
    }
    return X, y, assumptions


def make_pipeline(X: pd.DataFrame) -> tuple[Pipeline, list[str], list[str]]:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    classifier = RandomForestClassifier(
        n_estimators=180,
        max_depth=10,
        min_samples_leaf=25,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", classifier)]), numeric_features, categorical_features


def evaluate_model(y_true: pd.Series, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = (probabilities >= 0.5).astype(int)
    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "confusion_matrix": cm.tolist(),
    }


def transformed_feature_names(pipeline: Pipeline) -> list[str]:
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    return preprocessor.get_feature_names_out().tolist()


def compute_shap_summary(pipeline: Pipeline, X_sample: pd.DataFrame) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]
    transformed = preprocessor.transform(X_sample)
    feature_names = transformed_feature_names(pipeline)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed)
    class_values = shap_values[1] if isinstance(shap_values, list) else shap_values[:, :, 1]
    importance = np.abs(class_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .head(25)
        .reset_index(drop=True)
    )


def train_and_save(
    prefer_full: bool = True,
    sample_rows: int | None = None,
    save_artifact: bool = True,
) -> ModelBundle:
    df = load_bank_data(prefer_full=prefer_full)
    if sample_rows and len(df) > sample_rows:
        df = df.sample(sample_rows, random_state=RANDOM_STATE)

    X, y, assumptions = build_feature_frame(df)
    pipeline, numeric_features, categorical_features = make_pipeline(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = evaluate_model(y_test, probabilities)
    shap_sample = X_test.sample(min(700, len(X_test)), random_state=RANDOM_STATE)
    shap_summary = compute_shap_summary(pipeline, shap_sample)
    test_sample = X_test.assign(churn_risk=probabilities, actual_churn_risk=y_test.values)

    bundle = ModelBundle(
        pipeline=pipeline,
        metrics=metrics,
        feature_columns=X.columns.tolist(),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        shap_summary=shap_summary,
        test_sample=test_sample.reset_index(drop=True),
        test_labels=y_test.reset_index(drop=True),
        test_probabilities=probabilities,
        assumptions=assumptions,
    )
    if save_artifact:
        ARTIFACT_DIR.mkdir(exist_ok=True)
        joblib.dump(bundle, ARTIFACT_PATH)
    return bundle


def load_or_train() -> ModelBundle:
    if ARTIFACT_PATH.exists():
        return joblib.load(ARTIFACT_PATH)
    return train_and_save(prefer_full=True)


def get_feature_importance(bundle: ModelBundle) -> pd.DataFrame:
    model = bundle.pipeline.named_steps["model"]
    names = transformed_feature_names(bundle.pipeline)
    return (
        pd.DataFrame({"feature": names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(25)
        .reset_index(drop=True)
    )


def individual_explanation(bundle: ModelBundle, row: pd.DataFrame, top_n: int = 8) -> tuple[float, pd.DataFrame]:
    probability = float(bundle.pipeline.predict_proba(row[bundle.feature_columns])[:, 1][0])
    preprocessor = bundle.pipeline.named_steps["preprocessor"]
    model = bundle.pipeline.named_steps["model"]
    transformed = preprocessor.transform(row[bundle.feature_columns])
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed)
    class_values = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0, :, 1]
    explanation = (
        pd.DataFrame({"feature": transformed_feature_names(bundle.pipeline), "shap_value": class_values})
        .assign(direction=lambda frame: np.where(frame["shap_value"] >= 0, "raises risk", "reduces risk"))
        .reindex(columns=["feature", "shap_value", "direction"])
    )
    explanation["abs_shap"] = explanation["shap_value"].abs()
    explanation = explanation.sort_values("abs_shap", ascending=False).head(top_n).drop(columns="abs_shap")
    return probability, explanation.reset_index(drop=True)


def risk_band(probability: float) -> str:
    if probability >= 0.75:
        return "High"
    if probability >= 0.55:
        return "Medium"
    return "Low"


def recommend_actions(row: pd.Series, probability: float, explanation: pd.DataFrame | None = None) -> list[str]:
    actions: list[str] = []
    band = risk_band(probability)
    if band == "High":
        actions.append("Prioritize for proactive retention outreach with a personalized offer and senior-agent handling.")
    elif band == "Medium":
        actions.append("Place into a monitored nurture journey with tailored messaging and a follow-up trigger.")
    else:
        actions.append("Keep in a low-cost engagement stream and avoid over-contacting the customer.")

    if row.get("poutcome") == "success":
        actions.append("Reference the successful prior campaign and make the next offer easy to accept.")
    if row.get("pdays", 999) != 999:
        actions.append("Use recent-contact context to time the next interaction instead of restarting the conversation.")
    if row.get("campaign", 0) >= 4:
        actions.append("Reduce contact pressure; switch to a higher-value channel or a clearer offer.")
    if row.get("contact") == "telephone":
        actions.append("Test a cellular or digital follow-up channel where available.")
    if row.get("default") == "unknown":
        actions.append("Resolve missing credit-default information before assigning costly incentives.")
    if explanation is not None and not explanation.empty:
        drivers = explanation.query("shap_value > 0").head(3)["feature"].str.replace("cat__", "", regex=False)
        if len(drivers) > 0:
            actions.append("Audit the top risk drivers for this customer: " + ", ".join(drivers.tolist()) + ".")

    return actions[:6]


def dataset_profile(df: pd.DataFrame) -> dict[str, Any]:
    duplicate_rows = int(df.duplicated().sum())
    unknown_counts = {
        column: int((df[column] == "unknown").sum())
        for column in df.select_dtypes(include=["object"]).columns
        if (df[column] == "unknown").any()
    }
    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "duplicates": duplicate_rows,
        "missing_cells": int(df.isna().sum().sum()),
        "unknown_counts": unknown_counts,
    }
