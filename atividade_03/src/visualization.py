"""Gráficos do estudo de hiperparâmetros e da avaliação."""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .experiments import carregar_curvas
from .metrics import METRICA_PRINCIPAL

NOME_METRICA = {"RMSE": "RMSE (bpm)", "F1": "F1 (classe positiva)"}
NOME_PROBLEMA = {"regressao": "Regressão", "classificacao": "Classificação"}


def curva_media(curvas, chave):
    """Média e desvio por época entre sementes; sementes que pararam antes contribuem só até
    a época em que pararam."""
    n = max(len(c[chave]) for c in curvas)
    m = np.full((len(curvas), n), np.nan)
    for i, c in enumerate(curvas):
        m[i, :len(c[chave])] = c[chave]
    with np.errstate(all="ignore"):
        return np.nanmean(m, axis=0), np.nanstd(m, axis=0)


def curvas_estudo(problema, tabela, titulo, ylim=None, ncols=4, log_y=False, max_epoca=None):
    """Um painel por valor do hiperparâmetro: perda de treino e de validação por época
    (média entre sementes; faixa = ±1 desvio). Linha tracejada = época média do melhor
    estado (early stopping)."""
    n = len(tabela)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.6 * nrows), sharey=True, squeeze=False)
    for ax, (_, linha) in zip(axes.flat, tabela.iterrows()):
        curvas = carregar_curvas(problema, linha["chave"])
        for chave, cor in (("treino", "#1f77b4"), ("val", "#d62728")):
            mu, sd = curva_media(curvas, chave)
            ep = np.arange(1, len(mu) + 1)
            ax.plot(ep, mu, color=cor, lw=1.3, label="treino" if chave == "treino" else "validação")
            ax.fill_between(ep, mu - sd, mu + sd, color=cor, alpha=0.18, lw=0)
        ax.axvline(linha["melhor_epoca_media"], color="gray", ls="--", lw=0.8)
        ax.set_title(str(linha["valor"]), fontsize=9)
        ax.grid(alpha=0.3)
        if log_y:
            ax.set_yscale("log")
        if ylim:
            ax.set_ylim(*ylim)
        if max_epoca:
            ax.set_xlim(0, max_epoca)
    for ax in axes.flat[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("época")
    for ax in axes[:, 0]:
        ax.set_ylabel("perda")
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(titulo, fontsize=11)
    fig.tight_layout()
    return fig


def metrica_por_valor(tabelas_por_problema, titulo, log=None, ylim=None):
    """Métrica principal no treino e na validação em função do valor (barras = ±1 desvio
    entre sementes), um painel por problema. Com `log`, mostra também o valor de validação de
    cada semente (pontos pequenos). `ylim` = {problema: (min, max)} limita o eixo; sementes
    fora da faixa aparecem como setas na borda, com o valor anotado."""
    fig, axes = plt.subplots(1, len(tabelas_por_problema), figsize=(5.2 * len(tabelas_por_problema), 3.3))
    axes = np.atleast_1d(axes)
    for ax, (problema, t) in zip(axes, tabelas_por_problema.items()):
        met = METRICA_PRINCIPAL[problema]
        x = np.arange(len(t))
        ax.errorbar(x - 0.06, t[f"treino_{met}_media"], t[f"treino_{met}_desvio"], fmt="o-",
                    capsize=3, label="treino", color="#1f77b4")
        ax.errorbar(x + 0.06, t[f"val_{met}_media"], t[f"val_{met}_desvio"], fmt="s-",
                    capsize=3, label="validação", color="#d62728")
        lim = (ylim or {}).get(problema)
        if log is not None:
            for xi, ch in zip(x, t["chave"]):
                vals = log[(log["problema"] == problema) & (log["chave"] == ch)][f"val_{met}"].to_numpy()
                dentro = vals if lim is None else vals[(vals >= lim[0]) & (vals <= lim[1])]
                ax.scatter(np.full(len(dentro), xi + 0.2), dentro, s=9, color="#d62728", alpha=0.45,
                           zorder=3, label="validação por semente" if xi == 0 else None)
                if lim is not None:
                    fora = vals[(vals < lim[0]) | (vals > lim[1])]
                    if len(fora):
                        acima = fora.mean() > lim[1]
                        borda = lim[1] if acima else lim[0]
                        cs = 2 if met == "RMSE" else 3
                        faixa = f"{fora.min():.{cs}f}" if len(fora) == 1 or np.ptp(fora) < 10 ** -cs \
                            else f"{fora.min():.{cs}f}–{fora.max():.{cs}f}"
                        rot = f"{len(fora)} semente{'s' if len(fora) > 1 else ''}: {faixa}"
                        ultimo = xi == len(x) - 1
                        ax.annotate(rot, (xi + 0.2, borda), xytext=(-4 if ultimo else 6, -12 if acima else 6),
                                    textcoords="offset points", fontsize=7, color="#7f0000",
                                    ha="right" if ultimo else "left")
                        ax.scatter([xi + 0.2], [borda], marker="^" if acima else "v", s=30,
                                   color="#7f0000", zorder=4, clip_on=False)
        if lim is not None:
            ax.set_ylim(*lim)
            for xi, mv, mt in zip(x, t[f"val_{met}_media"], t[f"treino_{met}_media"]):
                fora = [(n, v) for n, v in (("val", mv), ("treino", mt)) if not lim[0] <= v <= lim[1]]
                if fora:
                    txt = "média fora da escala\n" + ", ".join(f"{n} {v:.2f}" if met == "RMSE" else f"{n} {v:.3f}" for n, v in fora)
                    ultimo = xi == len(x) - 1
                    ax.annotate(txt, (xi + (-0.35 if ultimo else 0.35), lim[0] + 0.62 * (lim[1] - lim[0])),
                                fontsize=6.5, ha="right" if ultimo else "left", color="#444444",
                                bbox=dict(boxstyle="round", fc="white", ec="#bbbbbb", lw=0.6))
        rot = 30 if t["valor"].astype(str).str.len().max() > 6 else 0
        ax.set_xticks(x, t["valor"], rotation=rot, fontsize=8, ha="right" if rot else "center")
        ax.set_ylabel(NOME_METRICA[met])
        ax.set_title(NOME_PROBLEMA[problema], fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(titulo, fontsize=11)
    fig.tight_layout()
    return fig


def mapa_calor(media, desvio, rot_linhas, rot_colunas, xlabel, ylabel, titulo, cmap, ax=None, casas=3):
    """Mapa de calor da média (anotada com ± desvio entre sementes) de uma varredura 2D."""
    ax = ax or plt.gca()
    anot = np.array([[f"{m:.{casas}f}\n±{s:.{casas}f}" for m, s in zip(lm, ls)]
                     for lm, ls in zip(media, desvio)])
    sns.heatmap(media, annot=anot, fmt="", cmap=cmap, xticklabels=rot_colunas,
                yticklabels=rot_linhas, ax=ax, annot_kws={"fontsize": 7.5}, cbar_kws={"shrink": 0.8})
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(titulo, fontsize=10)
    return ax
