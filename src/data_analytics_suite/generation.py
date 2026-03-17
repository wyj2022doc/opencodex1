from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class GenerationRule:
    name: str
    expression: str


class DataGenerator:
    """根据原始数据生成满足要求的新数据。"""

    def bootstrap_with_noise(
        self,
        df: pd.DataFrame,
        n_samples: int,
        noise_scale: float = 0.01,
        random_state: int = 42,
    ) -> pd.DataFrame:
        sampled = df.sample(n=n_samples, replace=True, random_state=random_state).reset_index(drop=True)
        numeric_cols = sampled.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0 and noise_scale > 0:
            rng = np.random.default_rng(random_state)
            noise = rng.normal(0, noise_scale, size=(len(sampled), len(numeric_cols)))
            sampled.loc[:, numeric_cols] = sampled[numeric_cols].to_numpy() + noise
        return sampled

    def apply_rules(self, df: pd.DataFrame, rules: list[GenerationRule], extra_context: dict[str, Any] | None = None) -> pd.DataFrame:
        context: dict[str, Any] = {"np": np, "pd": pd}
        if extra_context:
            context.update(extra_context)

        result = df.copy()
        for rule in rules:
            result[rule.name] = result.eval(rule.expression, local_dict=context)
        return result
