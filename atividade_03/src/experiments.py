"""Estudo de hiperparâmetros: definição dos experimentos, execução paralela e resumo.

Cada execução (problema, configuração, semente) é registrada uma única vez em
logs/hiperparametros.csv (métricas) e logs/curvas.jsonl (perda por época). Uma execução já
registrada não é refeita, de modo que o estudo pode ser interrompido e retomado, e a
configuração de referência, que aparece em todos os estudos, é treinada só uma vez.

Os treinos rodam em paralelo (1 processo por núcleo físico, 1 thread cada). Os tempos de
treino desta etapa servem para comparar configurações entre si; a medição de eficiência do
modelo final é feita isoladamente no notebook 08.
"""
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from .config import LOGS, NUCLEOS_FISICOS, SEMENTES, preparar
from .metrics import MAIOR_MELHOR, METRICA_PRINCIPAL
from .train import BASELINE, executar

ARQ_LOG = LOGS / "hiperparametros.csv"
ARQ_CURVAS = LOGS / "curvas.jsonl"
N_TREINO = 5760  # tamanho do subconjunto de treino (para o batch = tamanho total)

# ---------------------------------------------------------------------------------------
# Estudos de um fator por vez: (nome, rótulo do valor, alterações em relação ao baseline)
# ---------------------------------------------------------------------------------------
def _variacoes(chave, valores):
    return [(str(v), {chave: v}) for v in valores]


def estudos(problema):
    """Estudos isolados (um fator por vez) para um problema."""
    e = {
        # obrigatórios
        "neuronios": _variacoes("neuronios", [2, 4, 8, 16, 32, 64, 128]),
        "camadas": _variacoes("camadas", [1, 2, 3, 5]),
        "ativacao": _variacoes("ativacao", ["ReLU", "LeakyReLU", "Tanh", "Sigmoid", "ELU", "GELU"]),
        # AdamW é avaliado com o weight decay desacoplado padrão do PyTorch (0,01), que é o
        # que o diferencia do Adam; com weight_decay=0 os dois seriam idênticos.
        "otimizador": [("SGD", {"otimizador": "SGD"}),
                       ("SGD_momentum", {"otimizador": "SGD_momentum"}),
                       ("RMSprop", {"otimizador": "RMSprop"}),
                       ("Adam", {"otimizador": "Adam"}),
                       ("AdamW", {"otimizador": "AdamW", "weight_decay": 0.01})],
        "lr": _variacoes("lr", [1e-4, 1e-3, 1e-2, 1e-1, 1.0]),
        "batch": [(str(b), {"batch": b}) for b in [8, 32, 128, 512]] + [("total", {"batch": N_TREINO})],
        # opcionais
        # sem early stopping o modelo é o da última época; com ES, o melhor estado é restaurado
        "epocas_es": [("20 ép., sem ES", {"epocas": 20, "paciencia": None}),
                      ("50 ép., sem ES", {"epocas": 50, "paciencia": None}),
                      ("200 ép., sem ES", {"epocas": 200, "paciencia": None}),
                      ("500 ép., sem ES", {"epocas": 500, "paciencia": None}),
                      ("200 ép., pac. 5", {"paciencia": 5}),
                      ("200 ép., pac. 20", {})],
        "weight_decay": _variacoes("weight_decay", [0.0, 1e-5, 1e-4, 1e-3, 1e-2]),
        "dropout": _variacoes("dropout", [0.0, 0.1, 0.3, 0.5]),
        "padronizar": [("com StandardScaler", {"padronizar": True}),
                       ("sem StandardScaler", {"padronizar": False})],
    }
    if problema == "regressao":
        e["perda"] = _variacoes("perda", ["MSE", "MAE", "Huber"])
    else:
        # razão negativos/positivos no treino = 0,697/0,303 = 2,30
        e["pos_weight"] = _variacoes("pos_weight", [1.0, 1.5, 2.3, 3.5])
    return e


# Varreduras em duas dimensões (mapas de calor), 3 sementes.
GRADES_2D = {
    "lr_x_batch": ("lr", [1e-4, 1e-3, 1e-2, 1e-1], "batch", [8, 32, 128, 512, N_TREINO]),
    "neuronios_x_wd": ("neuronios", [8, 32, 128, 256], "weight_decay", [0.0, 1e-4, 1e-3, 1e-2]),
}


def tarefas_grade(nome):
    p1, v1, p2, v2 = GRADES_2D[nome]
    return [(f"{a}|{b}", {p1: a, p2: b}) for a in v1 for b in v2]


# ---------------------------------------------------------------------------------------
# Registro e execução
# ---------------------------------------------------------------------------------------
def montar_cfg(alteracoes):
    cfg = dict(BASELINE)
    cfg.update(alteracoes)
    return cfg


def chave_cfg(cfg):
    """Identificador curto e estável de uma configuração."""
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def carregar_log():
    if not ARQ_LOG.exists():
        return pd.DataFrame()
    return pd.read_csv(ARQ_LOG)


_CACHE_DADOS = {}


def _iniciar_worker():
    import torch
    torch.set_num_threads(1)


def _rodar_uma(problema, cfg, seed):
    chave_dados = (problema, cfg["padronizar"])
    if chave_dados not in _CACHE_DADOS:
        _CACHE_DADOS[chave_dados] = preparar(problema, padronizar=cfg["padronizar"])
    res, hist = executar(cfg, _CACHE_DADOS[chave_dados], seed)
    return problema, cfg, res, hist


# Esquema fixo do log: as duas tarefas compartilham o arquivo, cada linha preenche as métricas
# do seu problema e deixa as do outro vazias.
_METRICAS_REG = ["MAE", "MSE", "RMSE", "MAPE (%)", "R2", "MedAE"]
_METRICAS_CLF = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
COLUNAS_LOG = (["problema", "chave"] + [f"cfg_{k}" for k in BASELINE]
               + ["seed", "n_parametros", "macs", "melhor_epoca", "epocas_executadas", "divergiu",
                  "tempo_treino_s", "tempo_por_epoca_s", "perda_val_min"]
               + [f"{c}_{m}" for c in ("treino", "val") for m in _METRICAS_REG + _METRICAS_CLF])


def _registrar(problema, cfg, res, hist):
    linha = {"problema": problema, "chave": chave_cfg(cfg), **{f"cfg_{k}": v for k, v in cfg.items()}, **res}
    df = pd.DataFrame([linha]).reindex(columns=COLUNAS_LOG)
    df.to_csv(ARQ_LOG, mode="a", header=not ARQ_LOG.exists(), index=False)
    with open(ARQ_CURVAS, "a", encoding="utf-8") as f:
        f.write(json.dumps({"problema": problema, "chave": linha["chave"], "seed": res["seed"],
                            "treino": hist["treino"], "val": hist["val"]}) + "\n")


def rodar(tarefas, n_jobs=NUCLEOS_FISICOS, verbose=True):
    """Executa as tarefas (problema, cfg, seed) que ainda não estão no log."""
    log = carregar_log()
    feitas = set(zip(log["problema"], log["chave"], log["seed"])) if len(log) else set()
    unicas, vistas = [], set()
    for problema, cfg, seed in tarefas:
        k = (problema, chave_cfg(cfg), seed)
        if k not in feitas and k not in vistas:
            vistas.add(k)
            unicas.append((problema, cfg, seed))
    if verbose:
        print(f"{len(tarefas)} tarefas, {len(tarefas) - len(unicas)} já registradas, "
              f"{len(unicas)} a executar")
    if not unicas:
        return
    with ProcessPoolExecutor(n_jobs, initializer=_iniciar_worker) as ex:
        futs = [ex.submit(_rodar_uma, *t) for t in unicas]
        for i, fut in enumerate(as_completed(futs), 1):
            _registrar(*fut.result())
            if verbose and (i % 25 == 0 or i == len(futs)):
                print(f"  {i}/{len(futs)} concluídas")


def tarefas_estudo(problema, variacoes, sementes=SEMENTES):
    return [(problema, montar_cfg(alt), s) for _, alt in variacoes for s in sementes]


# ---------------------------------------------------------------------------------------
# Resumo
# ---------------------------------------------------------------------------------------
def resumir(problema, variacoes, log=None, sementes=SEMENTES):
    """Tabela do estudo: para cada valor, média ± desvio (entre sementes) da métrica principal
    no treino e na validação, gap, nº de parâmetros, época de parada e tempo de treino."""
    log = carregar_log() if log is None else log
    met = METRICA_PRINCIPAL[problema]
    linhas = []
    for rotulo, alt in variacoes:
        cfg = montar_cfg(alt)
        r = log[(log["problema"] == problema) & (log["chave"] == chave_cfg(cfg))
                & (log["seed"].isin(sementes))]
        if r.empty:
            continue
        tr, va = r[f"treino_{met}"], r[f"val_{met}"]
        linhas.append({
            "valor": rotulo,
            "n_sementes": len(r),
            "n_parametros": int(r["n_parametros"].iloc[0]),
            f"treino_{met}_media": tr.mean(), f"treino_{met}_desvio": tr.std(),
            f"val_{met}_media": va.mean(), f"val_{met}_desvio": va.std(),
            "gap_media": (tr - va).mean(),
            "melhor_epoca_media": r["melhor_epoca"].mean(),
            "epocas_executadas_media": r["epocas_executadas"].mean(),
            "tempo_treino_s_media": r["tempo_treino_s"].mean(),
            "tempo_por_epoca_s_media": r["tempo_por_epoca_s"].mean(),
            "divergencias": int(r["divergiu"].sum()),
            "chave": chave_cfg(cfg),
        })
    return pd.DataFrame(linhas)


def carregar_curvas(problema, chave):
    """Curvas de perda (treino, val) de todas as sementes de uma configuração."""
    curvas = []
    with open(ARQ_CURVAS, encoding="utf-8") as f:
        for linha in f:
            d = json.loads(linha)
            if d["problema"] == problema and d["chave"] == chave:
                curvas.append(d)
    return sorted(curvas, key=lambda d: d["seed"])


def sensibilidade(problema, tabelas, log=None, sementes=SEMENTES):
    """Ordena os hiperparâmetros pelo impacto na métrica de validação: diferença entre o
    melhor e o pior valor testado (médias entre sementes).

    Para separar efeito real de variação entre sementes, as 5 sementes do melhor valor são
    comparadas às 5 do pior com o teste de Mann-Whitney (bilateral, não paramétrico, robusto a
    sementes que colapsam). O efeito é 'relevante' se p < 0,05, o que com 5 x 5 sementes exige
    separação quase completa entre os dois grupos; caso contrário é 'desprezível' (dentro da
    variação entre sementes)."""
    from scipy.stats import mannwhitneyu
    log = carregar_log() if log is None else log
    met = METRICA_PRINCIPAL[problema]
    maior = MAIOR_MELHOR[problema]
    linhas = []
    for nome, t in tabelas.items():
        t = t.dropna(subset=[f"val_{met}_media"])
        if t.empty:
            continue
        ordem = t.sort_values(f"val_{met}_media", ascending=not maior)
        melhor, pior = ordem.iloc[0], ordem.iloc[-1]
        amostra = lambda ch: log[(log["problema"] == problema) & (log["chave"] == ch)
                                 & log["seed"].isin(sementes)][f"val_{met}"].to_numpy()
        a, b = amostra(melhor["chave"]), amostra(pior["chave"])
        p = mannwhitneyu(a, b, alternative="two-sided").pvalue if len(a) > 1 and len(b) > 1 else np.nan
        linhas.append({"hiperparametro": nome, "melhor_valor": melhor["valor"],
                       f"melhor_val_{met}": melhor[f"val_{met}_media"],
                       "desvio_melhor": melhor[f"val_{met}_desvio"],
                       "pior_valor": pior["valor"], f"pior_val_{met}": pior[f"val_{met}_media"],
                       "desvio_pior": pior[f"val_{met}_desvio"],
                       "amplitude": abs(melhor[f"val_{met}_media"] - pior[f"val_{met}_media"]),
                       "p_mann_whitney": p,
                       "efeito": "relevante" if p < 0.05 else "desprezível"})
    return pd.DataFrame(linhas).sort_values("amplitude", ascending=False).reset_index(drop=True)


def tarefas_todas_isoladas_e_grades():
    """Todos os estudos de um fator por vez (5 sementes) e as grades 2D (3 sementes)."""
    tarefas = []
    for problema in ("regressao", "classificacao"):
        for variacoes in estudos(problema).values():
            tarefas += tarefas_estudo(problema, variacoes)
        for nome in GRADES_2D:
            tarefas += tarefas_estudo(problema, tarefas_grade(nome), SEMENTES[:3])
    return tarefas


if __name__ == "__main__":
    # Execução completa fora do notebook: python -m src.experiments
    rodar(tarefas_todas_isoladas_e_grades())
