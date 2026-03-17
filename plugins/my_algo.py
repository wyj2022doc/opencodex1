import pandas as pd


def run(df: pd.DataFrame, context: dict) -> pd.DataFrame:
    threshold = context.get("threshold", 10)
    df["high_feature1"] = (df["feature1"] > threshold).astype(int)
    return df
