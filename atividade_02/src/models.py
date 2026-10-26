"""Carrega os modelos treinados (notebooks 04 e 05) e o sistema especialista com a mesma
interface `predict(X)`, para que avaliação, generalização e eficiência tratem todos iguais."""
import joblib

from .config import ARQUIVO_REGRAS, MODELS, TARGETS
from .expert_system import BaseConhecimento, MotorInferencia

NOMES = ["Sistema Especialista", "Árvore (sem restrição)", "Árvore (ajustada)",
         "Random Forest", "Gradient Boosting"]
ARQUIVOS_ML = {"Árvore (sem restrição)": "dt_sem_restricao", "Árvore (ajustada)": "dt_ajustada",
               "Random Forest": "rf", "Gradient Boosting": "gb"}


class SistemaEspecialista:
    """Adaptador do motor de inferência com interface no estilo scikit-learn."""

    def __init__(self, problema):
        self.alvo = TARGETS[problema]
        self.arquivo = ARQUIVO_REGRAS[problema]
        self.motor = MotorInferencia(BaseConhecimento.de_arquivo(self.arquivo))

    def predict(self, X):
        return self.motor.prever(X, self.alvo)

    def pontuacao(self, X):
        return self.motor.pontuacao(X)


def carregar_modelos(problema):
    """dict nome -> modelo, na ordem de NOMES."""
    modelos = {"Sistema Especialista": SistemaEspecialista(problema)}
    for nome, arq in ARQUIVOS_ML.items():
        modelos[nome] = joblib.load(MODELS / f"{arq}_{problema}.joblib")
    return modelos


def pontuacao(modelo, X):
    """Pontuação contínua da classe positiva (probabilidade, ou score das regras; None se não houver)."""
    if isinstance(modelo, SistemaEspecialista):
        return modelo.pontuacao(X)
    return modelo.predict_proba(X)[:, 1]
