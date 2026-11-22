"""Laço de treinamento com early stopping, sementes e avaliação de uma configuração.

Uma configuração é um dicionário com todos os hiperparâmetros (ver BASELINE). Assim o
estudo de hiperparâmetros só troca valores no dicionário, sem reescrever código.
"""
import copy
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, TensorDataset

from .config import CLASSES, TARGETS
from .metrics import metricas_classificacao, metricas_regressao
from .models import criar_modelo, macs_por_amostra, n_parametros

# Configuração de referência (baseline), igual nos dois problemas.
BASELINE = {
    "neuronios": 32,        # neurônios por camada oculta
    "camadas": 2,           # nº de camadas ocultas
    "ativacao": "ReLU",
    "otimizador": "Adam",
    "lr": 1e-3,             # taxa de aprendizado
    "batch": 32,
    "epocas": 200,          # máximo de épocas
    "paciencia": 20,        # early stopping (épocas sem melhora na perda de validação)
    "weight_decay": 0.0,    # regularização L2
    "dropout": 0.0,
    "perda": "MSE",         # regressão: MSE, MAE ou Huber
    "pos_weight": 1.0,      # classificação: peso da classe positiva na BCE (1 = sem peso)
    "padronizar": True,     # StandardScaler nas features
}


def fixar_sementes(seed):
    """Fixa as sementes de random, numpy e torch (inicialização dos pesos, dropout e
    embaralhamento dos batches)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def criar_dataloader(X, y, batch, seed):
    """DataLoader sobre um TensorDataset. O BatchSampler entrega ao dataset um lote de
    índices de uma vez (TensorDataset aceita indexação por vetor), o que evita montar o
    lote amostra por amostra; o resultado é o mesmo de DataLoader(..., shuffle=True)."""
    ds = TensorDataset(torch.as_tensor(X, dtype=torch.float32),
                       torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1))
    g = torch.Generator().manual_seed(seed)
    amostrador = BatchSampler(RandomSampler(ds, generator=g), batch_size=min(batch, len(ds)),
                              drop_last=False)
    return DataLoader(ds, sampler=amostrador, batch_size=None)


def criar_perda(cfg, problema):
    """Regressão: MSE (padrão), MAE (L1) ou Huber, sobre o target padronizado.
    Classificação binária: BCEWithLogitsLoss (sigmoide embutida, numericamente estável),
    com pos_weight opcional para compensar o desbalanceamento."""
    if problema == "regressao":
        return {"MSE": nn.MSELoss(), "MAE": nn.L1Loss(), "Huber": nn.HuberLoss(delta=1.0)}[cfg["perda"]]
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg["pos_weight"]], dtype=torch.float32))


def criar_otimizador(cfg, parametros):
    nome, lr, wd = cfg["otimizador"], cfg["lr"], cfg["weight_decay"]
    if nome == "SGD":
        return torch.optim.SGD(parametros, lr=lr, weight_decay=wd)
    if nome == "SGD_momentum":
        return torch.optim.SGD(parametros, lr=lr, momentum=0.9, weight_decay=wd)
    if nome == "RMSprop":
        return torch.optim.RMSprop(parametros, lr=lr, weight_decay=wd)
    if nome == "Adam":
        return torch.optim.Adam(parametros, lr=lr, weight_decay=wd)
    if nome == "AdamW":
        return torch.optim.AdamW(parametros, lr=lr, weight_decay=wd)
    raise ValueError(nome)


def treinar(modelo, dl_tr, X_val, y_val, loss_fn, otimizador, n_epocas=200, paciencia=20):
    """Treina com early stopping na perda de validação e restaura o melhor estado
    (adaptado do Listing 2 do enunciado). `paciencia=None` desliga o early stopping: o treino
    vai até `n_epocas` e o modelo devolvido é o da última época (sem restaurar o melhor ponto).

    Retorna o histórico de perdas, a melhor época (1-based), o nº de épocas executadas e se
    o treino divergiu (perda não finita)."""
    X_val = torch.as_tensor(X_val, dtype=torch.float32)
    y_val = torch.as_tensor(y_val, dtype=torch.float32).reshape(-1, 1)
    hist = {"treino": [], "val": []}
    melhor, espera, melhor_epoca = float("inf"), 0, 0
    melhor_estado = copy.deepcopy(modelo.state_dict())  # garante um estado válido
    divergiu = False
    for ep in range(n_epocas):
        modelo.train()
        soma = 0.0
        for xb, yb in dl_tr:
            otimizador.zero_grad()
            perda = loss_fn(modelo(xb), yb)
            perda.backward()
            otimizador.step()
            soma += perda.item() * len(xb)
        perda_tr = soma / len(dl_tr.dataset)

        modelo.eval()
        with torch.no_grad():
            perda_val = loss_fn(modelo(X_val), y_val).item()
        hist["treino"].append(perda_tr)
        hist["val"].append(perda_val)

        if not (np.isfinite(perda_tr) and np.isfinite(perda_val)):
            divergiu = True
            break
        if perda_val < melhor - 1e-6:
            melhor, espera, melhor_epoca = perda_val, 0, ep + 1
            melhor_estado = copy.deepcopy(modelo.state_dict())
        else:
            espera += 1
            if paciencia is not None and espera >= paciencia:
                break
    if paciencia is not None or divergiu:
        modelo.load_state_dict(melhor_estado)  # restaura o melhor ponto
    return hist, melhor_epoca, len(hist["treino"]), divergiu


@torch.no_grad()
def prever(modelo, X):
    """Saída bruta da rede em modo de avaliação (valor padronizado ou logit)."""
    modelo.eval()
    return modelo(torch.as_tensor(X, dtype=torch.float32)).squeeze(1).numpy()


def avaliar(modelo, dados, X, y_orig, limiar=0.5):
    """Métricas na escala original. Regressão: desfaz a padronização do target.
    Classificação: probabilidade = sigmoide(logit), classe = prob >= limiar."""
    saida = prever(modelo, X)
    if dados.problema == "regressao":
        y_pred = dados.para_original(saida)
        if not np.all(np.isfinite(y_pred)):
            return {k: np.nan for k in ("MAE", "MSE", "RMSE", "MAPE (%)", "R2", "MedAE")}
        return metricas_regressao(y_orig, y_pred)
    prob = 1 / (1 + np.exp(-np.clip(saida, -50, 50)))
    prob = np.nan_to_num(prob, nan=0.0)
    return metricas_classificacao(y_orig, (prob >= limiar).astype(int), prob)


def executar(cfg, dados, seed, retornar_modelo=False):
    """Treina uma configuração com uma semente e devolve as métricas de treino e validação
    (o teste NÃO é avaliado aqui), o histórico de perdas e, opcionalmente, o modelo."""
    fixar_sementes(seed)
    modelo = criar_modelo(cfg, len(dados.features))
    loss_fn = criar_perda(cfg, dados.problema)
    otimizador = criar_otimizador(cfg, modelo.parameters())
    dl_tr = criar_dataloader(dados.X_tr, dados.y_tr, cfg["batch"], seed)

    inicio = time.perf_counter()
    hist, melhor_ep, n_ep, divergiu = treinar(modelo, dl_tr, dados.X_val, dados.y_val, loss_fn,
                                             otimizador, cfg["epocas"], cfg["paciencia"])
    tempo = time.perf_counter() - inicio

    m_tr = avaliar(modelo, dados, dados.X_tr, dados.y_tr_orig)
    m_val = avaliar(modelo, dados, dados.X_val, dados.y_val_orig)
    res = {
        "seed": seed,
        "n_parametros": n_parametros(modelo),
        "macs": macs_por_amostra(len(dados.features), [cfg["neuronios"]] * cfg["camadas"]),
        "melhor_epoca": melhor_ep,
        "epocas_executadas": n_ep,
        "divergiu": divergiu,
        "tempo_treino_s": tempo,
        "tempo_por_epoca_s": tempo / max(n_ep, 1),
        "perda_val_min": float(np.nanmin(hist["val"])) if np.isfinite(np.nanmin(hist["val"])) else np.nan,
        **{f"treino_{k}": v for k, v in m_tr.items()},
        **{f"val_{k}": v for k, v in m_val.items()},
    }
    if retornar_modelo:
        return res, hist, modelo
    return res, hist


def salvar_checkpoint(modelo, cfg, dados, caminho, limiar=0.5, extras=None):
    """Salva um checkpoint completo (dicionário só com tensores e tipos básicos, para poder
    ser lido com torch.load(weights_only=True)):
    state_dict + configuração da arquitetura + média/desvio do StandardScaler (+ do target na
    regressão) + features na ordem usada (+ classes e limiar na classificação)."""
    ckpt = {
        "state_dict": modelo.state_dict(),
        "config": {"in_dim": len(dados.features), "out_dim": 1,
                   "hidden": [cfg["neuronios"]] * cfg["camadas"],
                   "activation": cfg["ativacao"], "dropout": cfg["dropout"]},
        "hiperparametros": {k: v for k, v in cfg.items()},
        "problema": dados.problema,
        "target": TARGETS[dados.problema],
        "features": list(dados.features),
        "x_media": torch.tensor(dados.x_media, dtype=torch.float64),  # float64, como no StandardScaler
        "x_desvio": torch.tensor(dados.x_desvio, dtype=torch.float64),
        "versao_torch": str(torch.__version__),  # str: TorchVersion não é aceito por weights_only=True
    }
    if dados.problema == "regressao":
        ckpt["y_media"] = float(dados.y_media)
        ckpt["y_desvio"] = float(dados.y_desvio)
    else:
        ckpt["classes"] = list(CLASSES)
        ckpt["limiar"] = float(limiar)
    if extras:
        ckpt.update(extras)
    torch.save(ckpt, caminho)
    return ckpt
