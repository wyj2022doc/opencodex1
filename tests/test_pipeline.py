from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from data_analytics_suite.generation import DataGenerator, GenerationRule
from data_analytics_suite.llm_embed import LLMPipelineEmbedder
from data_analytics_suite.pipeline import AnalyticsPipeline, PipelineStep


def test_pipeline_and_export(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "feature1": [1.0, 2.0, 3.0, 4.0],
            "feature2": [2.0, 3.0, 4.0, 5.0],
            "target": [3.0, 5.0, 7.0, 9.0],
        }
    )

    pipeline = AnalyticsPipeline()
    result = pipeline.apply(
        df,
        [
            PipelineStep(name="fill", step_type="fillna_mean", params={}),
            PipelineStep(name="pred", step_type="linear_regression", params={"features": ["feature1", "feature2"], "target": "target"}),
        ],
    )

    assert "prediction" in result.columns
    spec_path = tmp_path / "spec.json"
    pipeline.export_spec(str(spec_path))
    exported = json.loads(spec_path.read_text(encoding="utf-8"))
    assert len(exported["steps"]) == 2


def test_generation_and_llm_embed() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [4.0, 5.0, 6.0]})
    generator = DataGenerator()
    generated = generator.bootstrap_with_noise(df, n_samples=5, noise_scale=0.0)
    assert len(generated) == 5

    ruled = generator.apply_rules(generated, [GenerationRule(name="sum_xy", expression="x + y")])
    assert "sum_xy" in ruled.columns

    payload = LLMPipelineEmbedder(model_name="test-llm").build_payload({"steps": []}, business_goal="测试")
    assert payload["model"] == "test-llm"
