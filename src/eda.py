import pandas as pd
import numpy as np


def missing_values_table(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum()
    pct = missing / len(df) * 100
    return (
        pd.DataFrame({"Ausentes": missing, "Percentual (%)": pct.round(2)})
        .sort_values("Ausentes", ascending=False)
        .query("Ausentes > 0")
    )


def detect_outliers_iqr(df: pd.DataFrame, col: str) -> pd.Series:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    return (df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)


def outliers_summary(df: pd.DataFrame) -> pd.DataFrame:
    numericas = df.select_dtypes(include=[np.number]).columns
    rows = []
    for col in numericas:
        mask = detect_outliers_iqr(df, col)
        n = mask.sum()
        if n > 0:
            rows.append({"Coluna": col, "Outliers": n, "Percentual (%)": round(n / len(df) * 100, 2)})
    return pd.DataFrame(rows).sort_values("Outliers", ascending=False)


def high_correlations(df: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    numericas = df.select_dtypes(include=[np.number])
    corr = numericas.corr()
    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if abs(val) >= threshold:
                pairs.append({"Var1": cols[i], "Var2": cols[j], "Correlação": round(val, 4)})
    return pd.DataFrame(pairs).sort_values("Correlação", key=abs, ascending=False)


def categorical_frequencies(df: pd.DataFrame) -> None:
    for col in df.select_dtypes(include=["object", "category"]).columns:
        print(f"\n{col}:\n{df[col].value_counts()}")
