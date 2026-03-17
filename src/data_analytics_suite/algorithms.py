from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class AlgorithmResult:
    data: pd.DataFrame
    metadata: dict[str, Any]


class Preprocessor:
    """基础预处理算法集合。"""

    def fillna_mean(self, df: pd.DataFrame) -> AlgorithmResult:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        result = df.copy()
        result[numeric_cols] = result[numeric_cols].fillna(result[numeric_cols].mean())
        return AlgorithmResult(result, {"method": "fillna_mean", "columns": list(numeric_cols)})

    def standardize(self, df: pd.DataFrame, columns: list[str]) -> AlgorithmResult:
        scaler = StandardScaler()
        result = df.copy()
        result[columns] = scaler.fit_transform(result[columns])
        return AlgorithmResult(result, {"method": "standardize", "columns": columns})

    def minmax_scale(self, df: pd.DataFrame, columns: list[str]) -> AlgorithmResult:
        scaler = MinMaxScaler()
        result = df.copy()
        result[columns] = scaler.fit_transform(result[columns])
        return AlgorithmResult(result, {"method": "minmax_scale", "columns": columns})


class FeatureExtractor:
    """特征提取算法集合。"""

    def pca(self, df: pd.DataFrame, columns: list[str], n_components: int = 2) -> AlgorithmResult:
        model = PCA(n_components=n_components)
        transformed = model.fit_transform(df[columns])
        component_names = [f"pca_{i+1}" for i in range(n_components)]
        pca_df = pd.DataFrame(transformed, columns=component_names, index=df.index)
        result = pd.concat([df.drop(columns=columns), pca_df], axis=1)
        return AlgorithmResult(
            result,
            {
                "method": "pca",
                "columns": columns,
                "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
            },
        )


class AnomalyDetector:
    """异常检测算法集合。"""

    def isolation_forest(self, df: pd.DataFrame, columns: list[str], contamination: float = 0.05) -> AlgorithmResult:
        model = IsolationForest(contamination=contamination, random_state=42)
        predictions = model.fit_predict(df[columns])
        scores = model.decision_function(df[columns])

        result = df.copy()
        result["anomaly_label"] = np.where(predictions == -1, "anomaly", "normal")
        result["anomaly_score"] = scores
        return AlgorithmResult(
            result,
            {
                "method": "isolation_forest",
                "columns": columns,
                "contamination": contamination,
            },
        )


class Predictor:
    """预测算法集合。"""

    def linear_regression(self, df: pd.DataFrame, features: list[str], target: str) -> AlgorithmResult:
        model = LinearRegression()
        model.fit(df[features], df[target])

        result = df.copy()
        result["prediction"] = model.predict(df[features])
        return AlgorithmResult(
            result,
            {
                "method": "linear_regression",
                "features": features,
                "target": target,
                "coef": model.coef_.tolist(),
                "intercept": float(model.intercept_),
            },
        )

    def random_forest_regressor(
        self,
        df: pd.DataFrame,
        features: list[str],
        target: str,
        n_estimators: int = 100,
    ) -> AlgorithmResult:
        model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        model.fit(df[features], df[target])

        result = df.copy()
        result["prediction"] = model.predict(df[features])
        return AlgorithmResult(
            result,
            {
                "method": "random_forest_regressor",
                "features": features,
                "target": target,
                "n_estimators": n_estimators,
                "feature_importances": model.feature_importances_.tolist(),
            },
        )
