from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_histogram(df: pd.DataFrame, column: str, output: str) -> str:
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    df[column].hist(bins=20)
    plt.title(f"Histogram: {column}")
    plt.xlabel(column)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    return output


def plot_scatter(df: pd.DataFrame, x: str, y: str, output: str) -> str:
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 4))
    plt.scatter(df[x], df[y], alpha=0.7)
    plt.title(f"Scatter: {x} vs {y}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    return output
