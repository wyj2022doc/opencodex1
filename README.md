# 数据分析软件（Data Analytics Suite）

这是一个可扩展的数据分析软件原型，满足以下能力：

1. **基础算法能力**：内置预处理、特征提取、异常检测、预测算法。
2. **算法后可视化**：流程执行后自动输出直方图、散点图。
3. **自定义 Python 算法接入**：用户可通过插件文件接入 `run(df, context)` 算法。
4. **数据生成**：基于原始数据进行重采样、加噪声、表达式规则生成新字段。
5. **整套流程导出 + 大模型嵌入**：导出 pipeline JSON，并生成可嵌入大模型接口的 payload。

## 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 目录结构

```text
src/data_analytics_suite/
  algorithms.py      # 预处理、特征提取、异常检测、预测
  pipeline.py        # 流程引擎 + 导出
  generation.py      # 新数据生成
  plugins.py         # Python 插件加载运行
  visualization.py   # 可视化输出
  llm_embed.py       # 大模型接口 payload 生成
  cli.py             # 命令行入口
tests/
```

## 快速开始

1) 准备输入数据 `data/input.csv`。

2) 准备流程配置 `config/demo_pipeline.json`：

```json
{
  "generate": {
    "n_samples": 120,
    "noise_scale": 0.05,
    "rules": [
      { "name": "feature_sum", "expression": "feature1 + feature2" }
    ]
  },
  "steps": [
    { "name": "fill_missing", "step_type": "fillna_mean", "params": {} },
    { "name": "scale", "step_type": "standardize", "params": { "columns": ["feature1", "feature2"] } },
    { "name": "anomaly", "step_type": "isolation_forest", "params": { "columns": ["feature1", "feature2"], "contamination": 0.08 } },
    { "name": "predict", "step_type": "linear_regression", "params": { "features": ["feature1", "feature2"], "target": "target" } }
  ],
  "visualization": {
    "histogram": { "column": "prediction", "output": "outputs/pred_hist.png" },
    "scatter": { "x": "feature1", "y": "prediction", "output": "outputs/pred_scatter.png" }
  },
  "llm_model": "custom-llm"
}
```

3) 运行：

```bash
python -m data_analytics_suite.cli \
  --input data/input.csv \
  --config config/demo_pipeline.json \
  --output outputs/result.csv \
  --export-spec outputs/pipeline_spec.json \
  --business-goal "产线质量预测"
```

执行后将得到：
- `outputs/result.csv`：处理结果
- `outputs/pipeline_spec.json`：完整算法链路
- `outputs/llm_payload.json`：大模型接口载荷
- `outputs/*.png`：图表结果

## 自定义 Python 算法接入

在配置中添加插件步骤：

```json
{
  "name": "custom_rule",
  "step_type": "plugin",
  "params": {
    "file_path": "plugins/my_algo.py",
    "context": { "threshold": 10 }
  }
}
```

插件示例 `plugins/my_algo.py`：

```python
import pandas as pd

def run(df: pd.DataFrame, context: dict) -> pd.DataFrame:
    threshold = context.get("threshold", 0)
    df["flag"] = (df["feature1"] > threshold).astype(int)
    return df
```
