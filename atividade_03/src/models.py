"""Rede neural perceptron multicamadas (MLP) parametrizável em PyTorch.

A mesma classe serve aos dois problemas; só mudam `out_dim` e a função de perda:
- regressão: 1 neurônio de saída linear + MSELoss (sobre o target padronizado);
- classificação binária: 1 neurônio de saída sem sigmoide (logit) + BCEWithLogitsLoss.
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """MLP com `hidden` camadas ocultas (tupla com o nº de neurônios de cada uma).

    activation: nome de uma ativação de torch.nn (ReLU, LeakyReLU, Tanh, Sigmoid, ELU, GELU).
    dropout: probabilidade de dropout após cada ativação oculta (0 = sem dropout).
    A saída é linear (sem ativação): valor previsto na regressão, logit na classificação.
    """

    def __init__(self, in_dim, out_dim=1, hidden=(32, 32), activation="ReLU", dropout=0.0):
        super().__init__()
        act = getattr(nn, activation)
        camadas, d = [], in_dim
        for h in hidden:
            camadas += [nn.Linear(d, h), act()]
            if dropout > 0:
                camadas.append(nn.Dropout(dropout))
            d = h
        camadas.append(nn.Linear(d, out_dim))  # saída em logits / valor contínuo
        self.net = nn.Sequential(*camadas)

    def forward(self, x):
        return self.net(x)


def n_parametros(modelo):
    """Número de parâmetros treináveis (pesos + vieses)."""
    return sum(p.numel() for p in modelo.parameters() if p.requires_grad)


def macs_por_amostra(in_dim, hidden, out_dim=1):
    """Multiplicações-acumulações por amostra numa passada para frente: em cada camada
    Linear(d_in, d_out) são d_in x d_out MACs (as ativações e somas de viés são desprezíveis)."""
    dims = [in_dim, *hidden, out_dim]
    return sum(a * b for a, b in zip(dims[:-1], dims[1:]))


def criar_modelo(cfg, in_dim):
    """Cria a MLP a partir de um dicionário de configuração (ver train.BASELINE)."""
    return MLP(in_dim, 1, tuple([cfg["neuronios"]] * cfg["camadas"]),
               cfg["ativacao"], cfg["dropout"])
