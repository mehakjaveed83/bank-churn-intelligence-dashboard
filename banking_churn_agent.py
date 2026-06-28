from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "y"
LEAKAGE_COLUMNS = ["duration"]


@dataclass
class ModelArtifacts:
    model: Pipeline
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_pred: np.ndarray
    y_proba: np.ndarray
    metrics: dict[str, Any]
    feature_names: list[str]
    encoded_sample: pd.DataFrame


class BankingChurnAgent:
    """End-to-end churn-style analysis agent for the bank marketing dataset."""

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)

    def load_data(self) -> pd.DataFrame:
        df = pd.read_csv(self.data_path, sep=";")
        df.columns = [col.strip() for col in df.columns]
        return df

    def prepare_features(
        self, df: pd.DataFrame, include_duration: bool = False
    ) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
        feature_df = df.drop(columns=[TARGET]).copy()
        if not include_duration and "duration" in feature_df.columns:
            feature_df = feature_df.drop(columns=LEAKAGE_COLUMNS)

        y = df[TARGET].map({"no": 0, "yes": 1}).astype(int)
        numeric_features = feature_df.select_dtypes(include=np.number).columns.tolist()
        categorical_features = [
            col for col in feature_df.columns if col not in numeric_features
        ]
        return feature_df, y, numeric_features, categorical_features

    def profile(self, df: pd.DataFrame) -> dict[str, Any]:
        y_rate = float((df[TARGET] == "yes").mean())
        unknown_counts = {
            col: int((df[col] == "unknown").sum())
            for col in df.select_dtypes(include="object").columns
            if col != TARGET
        }
        return {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "positive_rate": y_rate,
            "negative_rate": 1 - y_rate,
            "duplicates": int(df.duplicated().sum()),
            "unknown_counts": unknown_counts,
        }

    def _preprocessor(
        self, numeric_features: list[str], categorical_features: list[str]
    ) -> ColumnTransformer:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, numeric_features),
                ("cat", categorical_pipe, categorical_features),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    def train_prediction_model(
        self,
        df: pd.DataFrame,
        include_duration: bool = False,
        n_estimators: int = 220,
        max_depth: int | None = 8,
        random_state: int = 42,
    ) -> ModelArtifacts:
        X, y, numeric_features, categorical_features = self.prepare_features(
            df, include_duration=include_duration
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=random_state, stratify=y
        )

        preprocessor = self._preprocessor(numeric_features, categorical_features)
        classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )
        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", classifier),
            ]
        )
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        threshold = 0.35
        y_pred = (y_proba >= threshold).astype(int)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        metrics = {
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "average_precision": float(average_precision_score(y_test, y_proba)),
            "threshold": threshold,
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": report,
        }

        fitted_preprocessor = model.named_steps["preprocessor"]
        feature_names = fitted_preprocessor.get_feature_names_out().tolist()
        encoded = fitted_preprocessor.transform(X_test.head(1200))
        encoded_sample = pd.DataFrame(encoded, columns=feature_names)

        return ModelArtifacts(
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            metrics=metrics,
            feature_names=feature_names,
            encoded_sample=encoded_sample,
        )

    def cluster_customers(
        self,
        df: pd.DataFrame,
        include_duration: bool = False,
        n_clusters: int = 5,
        sample_size: int = 9000,
        random_state: int = 42,
    ) -> dict[str, Any]:
        X, y, numeric_features, categorical_features = self.prepare_features(
            df, include_duration=include_duration
        )
        sampled = X.sample(
            n=min(sample_size, len(X)), random_state=random_state
        ).sort_index()
        sampled_y = y.loc[sampled.index]

        preprocessor = self._preprocessor(numeric_features, categorical_features)
        encoded = preprocessor.fit_transform(sampled)
        feature_names = preprocessor.get_feature_names_out().tolist()
        kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state)
        labels = kmeans.fit_predict(encoded)

        pca = PCA(n_components=2, random_state=random_state)
        coords = pca.fit_transform(encoded)
        clustered = sampled.copy()
        clustered["cluster"] = labels.astype(str)
        clustered["target"] = sampled_y.map({0: "no", 1: "yes"})
        clustered["pca_1"] = coords[:, 0]
        clustered["pca_2"] = coords[:, 1]

        summary = (
            clustered.assign(target_flag=(clustered["target"] == "yes").astype(int))
            .groupby("cluster")
            .agg(
                customers=("cluster", "size"),
                subscription_rate=("target_flag", "mean"),
                median_age=("age", "median"),
                median_campaign=("campaign", "median"),
                previous_contacts=("previous", "mean"),
            )
            .reset_index()
            .sort_values("subscription_rate", ascending=False)
        )

        return {
            "clustered": clustered,
            "summary": summary,
            "silhouette": float(silhouette_score(encoded, labels)),
            "explained_variance": pca.explained_variance_ratio_.tolist(),
            "feature_names": feature_names,
        }

    def shap_summary(
        self, artifacts: ModelArtifacts, max_rows: int = 600
    ) -> dict[str, Any]:
        encoded = artifacts.encoded_sample.head(max_rows)
        classifier = artifacts.model.named_steps["classifier"]
        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(encoded)

        if isinstance(shap_values, list):
            positive_values = shap_values[1]
        elif shap_values.ndim == 3:
            positive_values = shap_values[:, :, 1]
        else:
            positive_values = shap_values

        mean_abs = np.abs(positive_values).mean(axis=0)
        importance = (
            pd.DataFrame(
                {"feature": artifacts.feature_names, "mean_abs_shap": mean_abs}
            )
            .sort_values("mean_abs_shap", ascending=False)
            .head(20)
        )
        long_values = pd.DataFrame(positive_values, columns=artifacts.feature_names)
        return {
            "importance": importance,
            "values": long_values[importance["feature"].head(12).tolist()],
            "sample": encoded[importance["feature"].head(12).tolist()],
        }

    def recommendation_table(
        self, artifacts: ModelArtifacts, top_n: int = 250
    ) -> pd.DataFrame:
        scored = artifacts.X_test.copy()
        scored["actual"] = artifacts.y_test.map({0: "no", 1: "yes"}).values
        scored["subscription_probability"] = artifacts.y_proba
        scored["priority"] = pd.cut(
            scored["subscription_probability"],
            bins=[-0.01, 0.25, 0.5, 0.75, 1.0],
            labels=["Low", "Watch", "High", "Critical"],
        )
        scored["recommendation"] = scored.apply(self._recommend_row, axis=1)
        return scored.sort_values("subscription_probability", ascending=False).head(top_n)

    @staticmethod
    def _recommend_row(row: pd.Series) -> str:
        if row.get("poutcome") == "success":
            return "Fast-track with a loyalty term-deposit offer and minimal friction."
        if row.get("previous", 0) > 0 and row.get("poutcome") == "failure":
            return "Change message and channel; avoid repeating the previous campaign script."
        if row.get("campaign", 0) >= 4:
            return "Reduce contact pressure and switch to a lower-frequency nurturing sequence."
        if row.get("contact") == "cellular":
            return "Prioritize a mobile-first follow-up with a concise rate/value proposition."
        return "Use a consultative offer focused on trust, timing, and deposit flexibility."
