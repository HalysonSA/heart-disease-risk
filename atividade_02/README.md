# Atividade 2: Avaliação Comparativa de Modelos (Sistema Especialista, Árvore de Decisão e Ensemble Learning)

**Aluno:** Halyson Santos
**Matrícula:** 20220062812
**Disciplina:** DCT1401 - Inteligência Artificial (UFRN / DCT, 2026.2)
**Professor:** Thommas K. S. Flores
**Dataset:** [Heart Disease Risk 2026 - Kaggle](https://www.kaggle.com/datasets/uditjain13/heart-disease-risk-2026)
**Repositório:** [https://github.com/HalysonSA/heart-disease-risk](https://github.com/HalysonSA/heart-disease-risk)

## Descrição

Continuação da Atividade 1 (EDA e pré-processamento, na raiz do repositório). Aqui são
construídas e comparadas três famílias de modelos (Aulas 05, 06 e 07) nos dois problemas
definidos na atividade anterior, com as métricas da Aula 08 (desempenho, generalização e
eficiência computacional):

| Problema | Target | Métrica principal |
|---|---|---|
| Regressão | `max_heart_rate_achieved` (FC máxima, bpm) | RMSE |
| Classificação | `has_heart_disease` (0/1) | F1 (classe positiva = doença) |

Modelos: **Sistema Especialista** (regras SE-ENTÃO + motor de encadeamento para frente),
**Árvore de Decisão** (ajustada e sem restrição), **Random Forest** e **Gradient Boosting**.

Protocolo: mesma divisão treino/teste 80/20 para todos os modelos e os dois problemas
(estratificada por `has_heart_disease`, `random_state=42`); ajuste de hiperparâmetros com
`GridSearchCV` 5-fold apenas no treino; teste usado só na avaliação final.

## Estrutura

```
atividade_02/
├── README.md
├── requirements.txt
├── data/processed/                 # dataset_regression.csv, dataset_classification.csv (da Atividade 1)
├── notebooks/
│   ├── 03_expert_system.ipynb       # Apoio à elaboração, bases de regras, cronometragem
│   ├── 04_decision_tree.ipynb       # GridSearch, critérios, árvore livre x ajustada, visualização
│   ├── 05_ensemble.ipynb            # Random Forest, Gradient Boosting, curva n_árvores, importâncias
│   ├── 06_evaluation_metrics.ipynb  # Métricas, matriz de confusão, ROC, PR, resíduos
│   ├── 07_generalization.ipynb      # Gap, K-fold, diagnóstico, curvas de aprendizado
│   └── 08_computational_efficiency.ipynb  # Tempo, memória, disco, energia/CO2, tabela-resumo
├── src/
│   ├── config.py                    # Caminhos, semente, divisão treino/teste
│   ├── expert_system.py             # Parser da base de regras + motor de inferência
│   ├── testar_regras.py             # Métricas das bases de regras no treino
│   ├── models.py                    # Carrega todos os modelos com a mesma interface
│   ├── metrics.py                   # Métricas de regressão/classificação e gap
│   └── profiling.py                 # Tempo, memória, tamanho em disco, energia e CO2
├── rules/
│   ├── _sintaxe.txt                 # Sintaxe da base de conhecimento
│   ├── regras_regressao.txt         # Base de regras SE-ENTÃO (regressão)
│   └── regras_classificacao.txt     # Base de regras SE-ENTÃO (classificação)
├── logs/cronometro_regras.csv       # Sessões cronometradas de elaboração das regras
├── reports/
│   ├── tables/                      # Resultados em CSV (resumo_final.csv = tabela comparativa)
│   └── figures/                     # Gráficos
└── docs/
    ├── Atividade Avaliativa 2 - IA.pdf  # Relatório técnico
    └── gerar_relatorio.py           # Gera o relatório (.docx) a partir de reports/
```

## Como executar

```bash
cd atividade_02
pip install -r requirements.txt
python src/expert_system.py      # auto-teste do motor de inferência
python -m src.testar_regras      # métricas das regras no treino
jupyter notebook                 # execute 03 → 08, nesta ordem
```

Os notebooks 04 e 05 salvam os modelos em `models/` (fora do git), que são usados por 06-08.
Tempos dependem do hardware. As medições foram feitas em um AMD Ryzen 5 5600X (6 núcleos),
16 GB RAM, Windows 10 Pro e Python 3.13.

## Principais resultados

Conjunto de teste (1.800 pacientes). A tabela completa está em `reports/tables/resumo_final.csv`.

| Modelo | F1 (classif.) | ROC-AUC | RMSE (regr., bpm) | R² | Disco (KB, classif./regr.) |
|---|---|---|---|---|---|
| Sistema Especialista | 0,731 | 0,898 | 12,58 | 0,655 | 1,9 / 1,6 |
| Árvore (sem restrição) | 0,706 | 0,787 | 16,75 | 0,389 | 101 / 886 |
| Árvore (ajustada) | 0,756 | 0,885 | 12,30 | 0,671 | 29 / 6 |
| Random Forest | 0,793 | 0,939 | 12,13 | 0,680 | 18.494 / 7.193 |
| **Gradient Boosting** | **0,807** | **0,945** | **12,07** | **0,683** | 388 / 230 |

- **Classificação:** o Gradient Boosting é o melhor, com diferença relevante sobre a árvore e as regras
  (mais de 2 desvios padrão da CV). A recomendação é usá-lo junto das regras do especialista como explicação.
- **Regressão:** os quatro modelos ajustados ficam a até 0,5 bpm de RMSE uns dos outros. Árvore ajustada
  ou sistema especialista (fórmula de Tanaka + ajustes) entregam quase o mesmo resultado sendo
  interpretáveis e cabendo em poucos KB.
- **Sistema especialista:** as regras foram propostas com auxílio de IA generativa; a justificativa
  de cada uma está comentada nos arquivos de `rules/`. Os tempos em `logs/cronometro_regras.csv` são reais, mas medem só a inserção
  e a validação das regras, não o esforço de um especialista humano.
