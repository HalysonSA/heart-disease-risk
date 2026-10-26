"""Métricas da Aula 08 para regressão e classificação."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, median_absolute_error, precision_score,
                             r2_score, recall_score, roc_auc_score)

# Métrica principal de cada problema (usada no ajuste, gap e validação cruzada).
METRICA_PRINCIPAL = {"regressao": "RMSE", "classificacao": "F1"}
SCORING_CV = {"regressao": "neg_root_mean_squared_error", "classificacao": "f1"}


def metricas_regressao(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "MAPE (%)": 100 * mean_absolute_percentage_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MedAE": median_absolute_error(y_true, y_pred),
    }


def metricas_classificacao(y_true, y_pred, y_score=None):
    """Classe positiva = 1 (com doença). ROC-AUC/PR-AUC exigem uma pontuação contínua;
    sem ela ficam NaN."""
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


def gap(treino, teste):
    """Gap = desempenho no treino - desempenho no teste.
    Para métricas de acerto (F1, R2) gap > 0 indica queda fora do treino;
    para métricas de erro (RMSE, MAE) o sinal se inverte: gap < 0 indica overfitting."""
    return treino - teste
