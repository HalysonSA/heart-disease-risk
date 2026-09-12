import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_correlation_matrix(df: pd.DataFrame, save_path: str = None) -> None:
    numericas = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(16, 12))
    sns.heatmap(
        numericas.corr(),
        annot=False,
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Correlação"},
        ax=ax,
    )
    ax.set_title("Matriz de Correlação - Variáveis Numéricas", fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()


def plot_missing_values(df: pd.DataFrame, save_path: str = None) -> None:
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) == 0:
        print("Não há valores ausentes no dataset.")
        return
    fig, ax = plt.subplots(figsize=(12, 6))
    missing.sort_values().plot(kind="barh", color="coral", ax=ax)
    ax.set_xlabel("Quantidade de Valores Ausentes")
    ax.set_title("Valores Ausentes por Coluna")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()


def plot_outliers_boxplot(df: pd.DataFrame, save_path: str = None) -> None:
    numericas = df.select_dtypes(include=[np.number]).columns[:25]
    n = len(numericas)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 3))
    axes = axes.flatten()
    for i, col in enumerate(numericas):
        axes[i].boxplot(df[col].dropna())
        axes[i].set_title(col, fontsize=8)
        axes[i].tick_params(labelsize=6)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()


def plot_class_distribution(df: pd.DataFrame, target: str, labels: list = None, save_path: str = None) -> None:
    counts = df[target].value_counts().sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    counts.plot(kind="bar", ax=axes[0], color=["green", "red"], alpha=0.7, edgecolor="black")
    axes[0].set_title(f"Distribuição de {target}")
    axes[0].set_ylabel("Quantidade")
    pie_labels = labels or [str(k) for k in counts.index]
    axes[1].pie(counts.values, labels=pie_labels, autopct="%1.1f%%", colors=["green", "red"])
    axes[1].set_title(f"Proporção de {target}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.show()
