import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def encode_binary(df: pd.DataFrame, col: str, positive_value) -> pd.DataFrame:
    df = df.copy()
    df[col] = (df[col] == positive_value).astype(int)
    return df


def one_hot_encode(df: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    df = df.copy()
    dummies = pd.get_dummies(df[col], prefix=prefix, drop_first=False)
    df = pd.concat([df, dummies], axis=1).drop(col, axis=1)
    return df


def booleans_to_int(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bool_cols = df.select_dtypes(include=["bool"]).columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def standardize(df: pd.DataFrame, exclude_cols: list) -> tuple[pd.DataFrame, StandardScaler]:
    df = df.copy()
    cols = [c for c in df.columns if c not in exclude_cols]
    scaler = StandardScaler()
    df[cols] = scaler.fit_transform(df[cols])
    return df, scaler


def full_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = encode_binary(df, "sex", "Male")
    df = one_hot_encode(df, "chest_pain_type", "chest_pain")
    df = one_hot_encode(df, "smoker_status", "smoker")
    df = booleans_to_int(df)
    return df
