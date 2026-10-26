"""Mostra as métricas das regras atuais no TREINO. Uso (na pasta atividade_02):
    python -m src.testar_regras
"""
from src.config import ARQUIVO_REGRAS, TARGETS, carregar_split
from src.expert_system import BaseConhecimento, MotorInferencia
from src.metrics import metricas_classificacao, metricas_regressao

for problema, metricas in [("classificacao", metricas_classificacao), ("regressao", metricas_regressao)]:
    X, _, y, _ = carregar_split(problema)
    motor = MotorInferencia(BaseConhecimento.de_arquivo(ARQUIVO_REGRAS[problema]))
    pred = motor.prever(X, TARGETS[problema])
    extra = [motor.pontuacao(X)] if problema == "classificacao" else []
    m = metricas(y, pred, *extra)
    print(f"{problema:14s} ({len(motor.base)} regras):", {k: round(float(v), 3) for k, v in m.items()})
