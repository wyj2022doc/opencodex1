from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class LLMPipelineEmbedder:
    """将整套数据处理算法导出并嵌入到大模型接口。"""

    model_name: str = "custom-llm"

    def build_payload(self, pipeline_spec: dict[str, Any], business_goal: str) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "task": "pipeline_execution",
            "business_goal": business_goal,
            "pipeline_spec": pipeline_spec,
            "instructions": [
                "按步骤执行 pipeline_spec。",
                "输出每一步的中间统计信息。",
                "返回最终数据摘要与可解释结论。",
            ],
        }
