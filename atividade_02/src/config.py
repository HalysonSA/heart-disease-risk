"""Configuração compartilhada: caminhos, semente, carga dos dados e divisão treino/teste.

Todos os notebooks usam `carregar_split` para garantir que os três tipos de modelo
(sistema especialista, árvore e ensembles) sejam avaliados nos MESMOS conjuntos de
treino e teste, nos dois problemas.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
TEST_SIZE = 0.2

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
RULES = ROOT / "rules"
LOGS = ROOT / "logs"
MODELS = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"
TABLES = ROOT / "reports" / "tables"
for _p in (MODELS, FIGURES, TABLES):
    _p.mkdir(parents=True, exist_ok=True)

# Mesmos targets da atividade anterior.
TARGETS = {"regressao": "max_heart_rate_achieved", "classificacao": "has_heart_disease"}
ARQUIVOS = {"regressao": "dataset_regression.csv", "classificacao": "dataset_classification.csv"}
ARQUIVO_REGRAS = {"regressao": RULES / "regras_regressao.txt",
                  "classificacao": RULES / "regras_classificacao.txt"}

# Fator de emissão do Brasil apresentado em aula (gCO2/kWh).
FATOR_EMISSAO_BR = 45.8
# Potência usada na estimativa de energia: TDP do AMD Ryzen 5 5600X (W).
POTENCIA_CPU_W = 65.0


def carregar_split(problema):
    """Retorna X_train, X_test, y_train, y_test para 'regressao' ou 'classificacao'.

    A divisão é única (mesmos índices nos dois problemas), 80/20, estratificada
    pela classe de `has_heart_disease` e com semente fixa.
    """
    df = pd.read_csv(DATA / ARQUIVOS[problema])
    # Nomes sem espaço/hífen (ex.: chest_pain_Typical_Angina) para poderem ser citados nas regras.
    df.columns = df.columns.str.replace(r"[ -]", "_", regex=True)
    idx_train, idx_test = train_test_split(
        df.index, test_size=TEST_SIZE, random_state=SEED,
        stratify=df[TARGETS["classificacao"]],
    )
    alvo = TARGETS[problema]
    X = df.drop(columns=["patient_id", alvo])
    y = df[alvo]
    return X.loc[idx_train], X.loc[idx_test], y.loc[idx_train], y.loc[idx_test]


def salvar_fig(nome):
    """Salva a figura atual em reports/figures/<nome>.png."""
    plt.savefig(FIGURES / f"{nome}.png", dpi=150, bbox_inches="tight")


def salvar_tabela(df, nome, index=True):
    """Salva um DataFrame em reports/tables/<nome>.csv."""
    df.to_csv(TABLES / f"{nome}.csv", index=index)
