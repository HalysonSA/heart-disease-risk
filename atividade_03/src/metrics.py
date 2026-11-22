"""Métricas da Aula 08 para regressão e classificação, gap de generalização e diagnóstico."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, median_absolute_error, precision_score,
                             r2_score, recall_score, roc_auc_score)

# Métrica principal de cada problema (mesma da Atividade 2).
METRICA_PRINCIPAL = {"regressao": "RMSE", "classificacao": "F1"}
# True = quanto maior, melhor.
MAIOR_MELHOR = {"regressao": False, "classificacao": True}


def metricas_regressao(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "MAPE (%)": 100 * mean_absolute_percentage_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MedAE": median_absolute_error(y_true, y_pred),
    }


def metricas_classificacao(y_true, y_pred, y_score=None):
    """Classe positiva = 1 (com doença). ROC-AUC e PR-AUC usam a probabilidade da rede."""
    m = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": np.nan,
        "PR-AUC": np.nan,
    }
    if y_score is not None and len(np.unique(y_score)) > 1:
        m["ROC-AUC"] = roc_auc_score(y_true, y_score)
        m["PR-AUC"] = average_precision_score(y_true, y_score)
    return m


def matriz_confusao(y_true, y_pred):
    """Linhas = classe real, colunas = classe prevista (0, 1)."""
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def gap(treino, outro):
    """Gap = desempenho no treino - desempenho fora do treino (validação ou teste).
    F1: gap > 0 indica queda fora do treino. RMSE (erro): o sinal se inverte, gap < 0 indica
    erro maior fora do treino."""
    return treino - outro


def gap_relativo(treino, outro, problema):
    """Perda relativa de desempenho fora do treino (>0 = pior fora do treino)."""
    if MAIOR_MELHOR[problema]:
        return (treino - outro) / treino
    return (outro - treino) / treino


def diagnostico(treino, outro, melhor_outro, problema, lim_gap=0.10, lim_vies=0.05):
    """Critério (adaptado do usado na Atividade 2), avaliado nesta ordem:
    - underfitting: o desempenho no PRÓPRIO TREINO é > 5% pior que o da melhor configuração fora
      do treino (alto viés: o modelo não consegue nem ajustar os dados que viu);
    - overfitting: gap relativo treino -> fora do treino > 10% (alta variância);
    - bem ajustado: caso contrário.
    Verificar o viés primeiro evita classificar como overfitting um modelo que colapsou (erro de
    treino alto e erro fora do treino ainda maior)."""
    g = gap_relativo(treino, outro, problema)
    if MAIOR_MELHOR[problema]:
        vies = (melhor_outro - treino) / melhor_outro
    else:
        vies = (treino - melhor_outro) / melhor_outro
    if vies > lim_vies:
        return "underfitting"
    if g > lim_gap:
        return "overfitting"
    return "bem ajustado"
