from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pandas as pd


class PythonPluginRunner:
    """加载并执行用户自定义 Python 算法。"""

    def load_module(self, file_path: str) -> ModuleType:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"插件文件不存在: {file_path}")

        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载插件: {file_path}")

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def run(self, df: pd.DataFrame, file_path: str, context: dict[str, Any] | None = None) -> pd.DataFrame:
        module = self.load_module(file_path)
        if not hasattr(module, "run"):
            raise AttributeError("插件必须提供 run(df, context) 函数")

        context = context or {}
        output = module.run(df.copy(), context)
        if not isinstance(output, pd.DataFrame):
            raise TypeError("插件返回值必须是 pandas.DataFrame")
        return output
