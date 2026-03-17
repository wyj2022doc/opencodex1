from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable
import json

import pandas as pd

from .algorithms import AnomalyDetector, FeatureExtractor, Predictor, Preprocessor
from .plugins import PythonPluginRunner


@dataclass
class PipelineStep:
    name: str
    step_type: str
    params: dict[str, Any] = field(default_factory=dict)


class AnalyticsPipeline:
    """串联预处理、特征提取、异常检测、预测与插件步骤。"""

    def __init__(self) -> None:
        self.preprocessor = Preprocessor()
        self.feature_extractor = FeatureExtractor()
        self.anomaly_detector = AnomalyDetector()
        self.predictor = Predictor()
        self.plugin_runner = PythonPluginRunner()
        self.history: list[PipelineStep] = []

    def _record(self, name: str, step_type: str, params: dict[str, Any]) -> None:
        self.history.append(PipelineStep(name=name, step_type=step_type, params=params))

    def apply(self, df: pd.DataFrame, steps: list[PipelineStep]) -> pd.DataFrame:
        current = df.copy()
        handlers: dict[str, Callable[..., Any]] = {
            "fillna_mean": self.preprocessor.fillna_mean,
            "standardize": self.preprocessor.standardize,
            "minmax_scale": self.preprocessor.minmax_scale,
            "pca": self.feature_extractor.pca,
            "isolation_forest": self.anomaly_detector.isolation_forest,
            "linear_regression": self.predictor.linear_regression,
            "random_forest_regressor": self.predictor.random_forest_regressor,
        }

        for step in steps:
            if step.step_type == "plugin":
                current = self.plugin_runner.run(
                    current,
                    file_path=step.params["file_path"],
                    context=step.params.get("context", {}),
                )
                self._record(step.name, step.step_type, step.params)
                continue

            if step.step_type not in handlers:
                raise ValueError(f"未知步骤类型: {step.step_type}")
            result = handlers[step.step_type](current, **step.params)
            current = result.data
            self._record(step.name, step.step_type, result.metadata)

        return current

    def export_spec(self, output_path: str) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"steps": [asdict(step) for step in self.history]}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)
