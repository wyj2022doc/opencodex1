from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .generation import DataGenerator, GenerationRule
from .llm_embed import LLMPipelineEmbedder
from .pipeline import AnalyticsPipeline, PipelineStep
from .visualization import plot_histogram, plot_scatter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="数据分析软件 CLI")
    parser.add_argument("--input", required=True, help="输入 CSV 文件")
    parser.add_argument("--output", default="outputs/result.csv", help="输出 CSV 文件")
    parser.add_argument("--config", required=True, help="流程配置 JSON")
    parser.add_argument("--export-spec", default="outputs/pipeline_spec.json", help="导出的流程规范")
    parser.add_argument("--business-goal", default="数据洞察", help="用于大模型嵌入的业务目标")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    df = pd.read_csv(args.input)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))

    generator = DataGenerator()
    if "generate" in config:
        generate_cfg = config["generate"]
        df = generator.bootstrap_with_noise(
            df,
            n_samples=generate_cfg.get("n_samples", len(df)),
            noise_scale=generate_cfg.get("noise_scale", 0.0),
        )
        rules = [GenerationRule(**item) for item in generate_cfg.get("rules", [])]
        if rules:
            df = generator.apply_rules(df, rules)

    steps = [PipelineStep(**item) for item in config.get("steps", [])]
    pipeline = AnalyticsPipeline()
    result = pipeline.apply(df, steps)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    pipeline.export_spec(args.export_spec)

    viz_cfg = config.get("visualization", {})
    if "histogram" in viz_cfg:
        plot_histogram(result, viz_cfg["histogram"]["column"], viz_cfg["histogram"].get("output", "outputs/hist.png"))
    if "scatter" in viz_cfg:
        plot_scatter(
            result,
            viz_cfg["scatter"]["x"],
            viz_cfg["scatter"]["y"],
            viz_cfg["scatter"].get("output", "outputs/scatter.png"),
        )

    embedder = LLMPipelineEmbedder(model_name=config.get("llm_model", "custom-llm"))
    payload = embedder.build_payload(
        pipeline_spec=json.loads(Path(args.export_spec).read_text(encoding="utf-8")),
        business_goal=args.business_goal,
    )
    Path("outputs/llm_payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
