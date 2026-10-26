"""Medição de tempo, memória, tamanho em disco e energia/CO2."""
import os
import tempfile
import time
import tracemalloc

import joblib
import numpy as np
from sklearn.base import clone

from .config import FATOR_EMISSAO_BR, POTENCIA_CPU_W


def medir_inferencia(funcao_predicao, X, n_rep=30):
    """Mede tempo e pico de memória apenas da etapa de inferência (adaptado do enunciado).
    O tempo é medido sem o tracemalloc ligado (ele adiciona overhead); o pico de memória
    vem de uma execução extra com tracemalloc."""
    tempos = []
    for _ in range(n_rep):
        inicio = time.perf_counter()
        funcao_predicao(X)
        tempos.append(time.perf_counter() - inicio)
    tracemalloc.start()
    funcao_predicao(X)
    _, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    media, desvio = np.mean(tempos), np.std(tempos)
    return {
        "inf_tempo_medio_s": media,
        "inf_tempo_desvio_s": desvio,
        "inf_tempo_por_amostra_ms": 1000 * media / len(X),
        "inf_vazao_amostras_s": len(X) / media,
        "inf_pico_memoria_kb": pico / 1024,
    }


def medir_treino(modelo, X, y, n_rep=5):
    """Treina `n_rep` cópias do modelo (mesmos hiperparâmetros) medindo o tempo de parede;
    depois uma cópia extra com tracemalloc para o pico de memória alocada.
    Retorna (métricas, modelo treinado)."""
    tempos = []
    for _ in range(n_rep):
        m = clone(modelo)
        inicio = time.perf_counter()
        m.fit(X, y)
        tempos.append(time.perf_counter() - inicio)
    tracemalloc.start()
    clone(modelo).fit(X, y)
    _, pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "treino_tempo_medio_s": np.mean(tempos),
        "treino_tempo_desvio_s": np.std(tempos),
        "treino_pico_memoria_kb": pico / 1024,
    }, m


def tamanho_disco_kb(modelo=None, arquivo=None):
    """Tamanho do modelo serializado com joblib (ou de um arquivo existente, ex.: regras)."""
    if arquivo is not None:
        return os.path.getsize(arquivo) / 1024
    with tempfile.TemporaryDirectory() as d:
        caminho = os.path.join(d, "modelo.joblib")
        joblib.dump(modelo, caminho)
        return os.path.getsize(caminho) / 1024


def energia_co2(tempo_s, potencia_w=POTENCIA_CPU_W, fator=FATOR_EMISSAO_BR):
    """Estimativa: energia (kWh) = potência média (W) x tempo (s) / 3.6e6;
    CO2 (g) = energia x fator de emissão do Brasil (45,8 gCO2/kWh, apresentado em aula).
    Usar o TDP como potência média dá um limite superior (o processo raramente ocupa a
    CPU inteira)."""
    kwh = potencia_w * tempo_s / 3.6e6
    return kwh, kwh * fator


def complexidade(modelo):
    """Profundidade / nós / folhas de uma árvore, ou nº de árvores e nós totais de um ensemble."""
    if hasattr(modelo, "tree_"):
        return {"n_arvores": 1, "profundidade": modelo.get_depth(),
                "n_nos": modelo.tree_.node_count, "n_folhas": modelo.get_n_leaves()}
    arvores = np.ravel(modelo.estimators_)
    return {"n_arvores": len(arvores),
            "profundidade": max(a.get_depth() for a in arvores),
            "n_nos": sum(a.tree_.node_count for a in arvores),
            "n_folhas": sum(a.get_n_leaves() for a in arvores)}
