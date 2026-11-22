"""Configuração compartilhada: caminhos, sementes, divisão treino/validação/teste e padronização.

Protocolo (igual para todos os experimentos):
- mesma divisão treino/teste 80/20 das Atividades 1 e 2 (random_state=42, estratificada por
  has_heart_disease, mesmos índices nos dois problemas);
- validação = 20% do treino, também estratificada e com semente fixa (usada para comparar
  hiperparâmetros, early stopping e escolha do limiar);
- StandardScaler ajustado só no subconjunto de treino; na regressão o target também é
  padronizado com média e desvio do treino.
"""
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED_SPLIT = 42            # mesma semente da divisão das atividades anteriores
TEST_SIZE = 0.2
VAL_SIZE = 0.2             # fração do treino reservada para validação
SEMENTES = [0, 1, 2, 3, 4] # sementes de inicialização/embaralhamento das redes

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"
for _p in (MODELS, LOGS, FIGURES, TABLES):
    _p.mkdir(parents=True, exist_ok=True)

TARGETS = {"regressao": "max_heart_rate_achieved", "classificacao": "has_heart_disease"}
ARQUIVOS = {"regressao": "dataset_regression.csv", "classificacao": "dataset_classification.csv"}
CLASSES = ["sem_doenca", "com_doenca"]

# Fator de emissão do Brasil apresentado em aula (gCO2/kWh).
FATOR_EMISSAO_BR = 45.8
# Potência usada na estimativa de energia: TDP do AMD Ryzen 5 5600X (W), 6 núcleos.
POTENCIA_CPU_W = 65.0
NUCLEOS_FISICOS = 6


def ler_dataset(problema):
    """Lê o CSV do problema com os nomes de coluna usados desde a Atividade 2
    (espaços e hífens trocados por '_', ex.: chest_pain_Typical_Angina)."""
    df = pd.read_csv(DATA / ARQUIVOS[problema])
    df.columns = df.columns.str.replace(r"[ -]", "_", regex=True)
    return df


def indices_split(df):
    """Índices de treino, validação e teste. A divisão treino/teste é a mesma das atividades
    anteriores; a validação sai do treino (20%, estratificada)."""
    estrato = df[TARGETS["classificacao"]]
    idx_treino_total, idx_teste = train_test_split(
        df.index, test_size=TEST_SIZE, random_state=SEED_SPLIT, stratify=estrato)
    idx_treino, idx_val = train_test_split(
        idx_treino_total, test_size=VAL_SIZE, random_state=SEED_SPLIT,
        stratify=estrato.loc[idx_treino_total])
    return idx_treino, idx_val, idx_teste, idx_treino_total


@dataclass
class Dados:
    """Conjuntos já padronizados (float32) + os parâmetros de padronização do treino."""
    problema: str
    features: list
    X_tr: np.ndarray
    y_tr: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_te: np.ndarray
    y_te: np.ndarray
    x_media: np.ndarray
    x_desvio: np.ndarray
    y_media: float = 0.0
    y_desvio: float = 1.0
    # versões na escala original (para métricas e inferência)
    y_tr_orig: np.ndarray = None
    y_val_orig: np.ndarray = None
    y_te_orig: np.ndarray = None
    X_te_orig: pd.DataFrame = None

    def para_original(self, y_padronizado):
        """Desfaz a padronização do target (regressão)."""
        return np.asarray(y_padronizado) * self.y_desvio + self.y_media


def preparar(problema, idx_treino=None, idx_val=None, idx_teste=None, padronizar=True):
    """Monta os conjuntos de um problema. Por padrão usa a divisão oficial; índices podem ser
    passados para a validação cruzada K-fold. A padronização é sempre ajustada só no treino.

    `padronizar=False` é usado apenas no estudo opcional 'com e sem StandardScaler'."""
    df = ler_dataset(problema)
    if idx_treino is None:
        idx_treino, idx_val, idx_teste, _ = indices_split(df)
    alvo = TARGETS[problema]
    X = df.drop(columns=["patient_id", alvo])
    y = df[alvo].astype(float)
    feats = list(X.columns)

    scaler = StandardScaler().fit(X.loc[idx_treino])
    if padronizar:
        tx = lambda idx: scaler.transform(X.loc[idx]).astype(np.float32)
        media, desvio = scaler.mean_, scaler.scale_
    else:
        tx = lambda idx: X.loc[idx].to_numpy(np.float32)
        media, desvio = np.zeros(len(feats)), np.ones(len(feats))

    y_media, y_desvio = 0.0, 1.0
    if problema == "regressao":
        y_media, y_desvio = float(y.loc[idx_treino].mean()), float(y.loc[idx_treino].std(ddof=0))
    ty = lambda idx: ((y.loc[idx].to_numpy() - y_media) / y_desvio).astype(np.float32)

    return Dados(
        problema=problema, features=feats,
        X_tr=tx(idx_treino), y_tr=ty(idx_treino),
        X_val=tx(idx_val), y_val=ty(idx_val),
        X_te=tx(idx_teste) if idx_teste is not None else None,
        y_te=ty(idx_teste) if idx_teste is not None else None,
        x_media=np.asarray(media, dtype=np.float64), x_desvio=np.asarray(desvio, dtype=np.float64),
        y_media=y_media, y_desvio=y_desvio,
        y_tr_orig=y.loc[idx_treino].to_numpy(), y_val_orig=y.loc[idx_val].to_numpy(),
        y_te_orig=y.loc[idx_teste].to_numpy() if idx_teste is not None else None,
        X_te_orig=X.loc[idx_teste] if idx_teste is not None else None,
    )


def salvar_fig(nome):
    """Salva a figura atual em reports/figures/<nome>.png."""
    plt.savefig(FIGURES / f"{nome}.png", dpi=150, bbox_inches="tight")


def salvar_tabela(df, nome, index=False):
    """Salva um DataFrame em reports/tables/<nome>.csv."""
    df.to_csv(TABLES / f"{nome}.csv", index=index)
