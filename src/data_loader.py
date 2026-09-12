import pandas as pd


def load_raw(path: str = "data/raw/dataset.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def load_clean(path: str = "data/processed/dataset_clean.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def load_scaled(path: str = "data/processed/dataset_scaled.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def load_regression(path: str = "data/processed/dataset_regression.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def load_classification(path: str = "data/processed/dataset_classification.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def dataset_summary(df: pd.DataFrame) -> None:
    print(f"Shape: {df.shape}")
    print(f"Colunas: {list(df.columns)}")
    print(f"\nTipos:\n{df.dtypes}")
    print(f"\nValores ausentes:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
