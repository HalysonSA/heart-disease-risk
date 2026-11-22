"""Medição de eficiência computacional: tempo, memória (psutil), disco, energia e CO2."""
import os
import threading
import time

import numpy as np
import psutil
import torch

from .config import FATOR_EMISSAO_BR, POTENCIA_CPU_W


def medir_inferencia(modelo, X, n_rep=30, n_aquecimento=5):
    """Mede tempo e memória apenas da etapa de inferência (adaptado do Listing 4 do enunciado).
    Executa inferências de aquecimento (warm-up) antes de cronometrar. Em CPU a execução é
    síncrona; em GPU seria preciso torch.cuda.synchronize() antes de parar o cronômetro."""
    modelo.eval()
    proc = psutil.Process()
    with torch.inference_mode():
        for _ in range(n_aquecimento):
            modelo(X)
        mem_antes = proc.memory_info().rss
        tempos = []
        for _ in range(n_rep):
            inicio = time.perf_counter()
            modelo(X)
            tempos.append(time.perf_counter() - inicio)
        mem_depois = proc.memory_info().rss
    media, desvio = np.mean(tempos), np.std(tempos)
    return {
        "tempo_medio_s": media,
        "tempo_desvio_s": desvio,
        "tempo_por_amostra_ms": 1000 * media / len(X),
        "vazao_amostras_s": len(X) / media,
        "rss_mb": mem_depois / 2**20,
        "delta_memoria_kb": (mem_depois - mem_antes) / 1024,
    }


def medir_latencia_individual(modelo, X, n_aquecimento=5):
    """Uma amostra por chamada (X[i:i+1]) para todas as amostras de X."""
    modelo.eval()
    with torch.inference_mode():
        for i in range(n_aquecimento):
            modelo(X[i:i + 1])
        tempos = []
        for i in range(len(X)):
            inicio = time.perf_counter()
            modelo(X[i:i + 1])
            tempos.append(time.perf_counter() - inicio)
    tempos = np.array(tempos)
    return {"latencia_media_ms": 1000 * tempos.mean(), "latencia_desvio_ms": 1000 * tempos.std(),
            "latencia_p95_ms": 1000 * np.percentile(tempos, 95),
            "vazao_amostras_s": 1 / tempos.mean(), "tempo_total_s": tempos.sum()}


class MonitorRSS:
    """Amostra a memória residente (RSS) do processo numa thread em segundo plano e guarda o
    pico. O tracemalloc não serve aqui: ele não enxerga a memória alocada pelo código nativo
    (C++) do PyTorch."""

    def __init__(self, intervalo_s=0.005):
        self.intervalo_s = intervalo_s
        self.proc = psutil.Process()
        self.pico = 0
        self.inicial = 0
        self._parar = threading.Event()

    def _loop(self):
        while not self._parar.is_set():
            self.pico = max(self.pico, self.proc.memory_info().rss)
            time.sleep(self.intervalo_s)

    def __enter__(self):
        self.inicial = self.proc.memory_info().rss
        self.pico = self.inicial
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._parar.set()
        self._thread.join()
        self.pico = max(self.pico, self.proc.memory_info().rss)

    @property
    def pico_mb(self):
        return self.pico / 2**20

    @property
    def acrescimo_mb(self):
        return (self.pico - self.inicial) / 2**20


def tamanho_kb(caminho):
    return os.path.getsize(caminho) / 1024


def energia_co2(tempo_s, potencia_w=POTENCIA_CPU_W, fator=FATOR_EMISSAO_BR):
    """Energia (kWh) = potência média (W) x tempo (s) / 3,6e6; CO2 (g) = energia x 45,8 gCO2/kWh
    (fator de emissão do Brasil apresentado em aula). Usar o TDP como potência média dá um
    limite superior, pois o processo raramente ocupa a CPU inteira."""
    kwh = potencia_w * tempo_s / 3.6e6
    return kwh, kwh * fator
