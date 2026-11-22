# Atividade 3: Redes Neurais (MLP) com PyTorch — Hiperparâmetros, Persistência e Inferência

**Aluno:** Halyson Santos
**Matrícula:** 20220062812
**Disciplina:** DCT1401 - Inteligência Artificial (UFRN / DCT, 2026.2)
**Professor:** Thommas K. S. Flores
**Dataset:** [Heart Disease Risk 2026 - Kaggle](https://www.kaggle.com/datasets/uditjain13/heart-disease-risk-2026)
**Repositório:** [https://github.com/HalysonSA/heart-disease-risk](https://github.com/HalysonSA/heart-disease-risk) (pasta `atividade_03/`)
**Relatório:** [`docs/relatorio_final.pdf`](docs/relatorio_final.pdf) (fonte LaTeX em `docs/relatorio_final.tex`)

## Descrição

Continuação das Atividades 1 (EDA e pré-processamento, raiz do repositório) e 2 (sistema especialista, árvore
e ensembles, pasta `atividade_02/`). Aqui são construídas redes neurais perceptron multicamadas (MLP) em PyTorch
para os mesmos dois problemas, com os mesmos targets, features e divisão treino/teste:

| Problema              | Target                                     | Saída / perda                                      | Métrica principal    |
| --------------------- | ------------------------------------------ | -------------------------------------------------- | -------------------- |
| Regressão             | `max_heart_rate_achieved` (FC máxima, bpm) | 1 neurônio linear / `MSELoss` (target padronizado) | RMSE                 |
| Classificação binária | `has_heart_disease` (0/1, 30,3% positivos) | 1 logit / `BCEWithLogitsLoss`                      | F1 (classe positiva) |

O trabalho tem três partes:

1. **Estudo de hiperparâmetros** (um fator por vez, 5 sementes): neurônios, camadas, ativação, otimizador, taxa de
   aprendizado, batch (obrigatórios) + épocas/early stopping, weight decay, dropout, função de perda / `pos_weight` e
   padronização (opcionais); 2 mapas de calor de interação; busca aleatória com 30 configurações.
2. **Persistência:** checkpoint completo em `.pt` (state_dict + arquitetura + padronização + features + limiar),
   carregado com `weights_only=True` num ambiente limpo, com verificação de fidelidade.
3. **Avaliação** com as métricas da Aula 08: desempenho, generalização (gap, curvas, K-fold, viés × variância) e
   eficiência computacional (tempo, memória, disco, MACs, energia e CO₂).

## Protocolo

- Divisão treino/teste 80/20 **idêntica** à das atividades anteriores (`random_state=42`, estratificada por
  `has_heart_disease`); validação = 20% do treino (estratificada). Resultado: 5.760 / 1.440 / 1.800 pacientes.
- `StandardScaler` ajustado só no treino (features e, na regressão, também o target).
- Sementes fixas (`random`, `numpy`, `torch`); cada configuração treinada com 5 sementes; 1 thread por treino,
  o que torna os resultados reprodutíveis bit a bit.
- Early stopping na perda de validação (paciência 20, restaura o melhor estado). O teste só é usado na avaliação
  final, depois de congelar configuração e limiar.

## Estrutura

```
atividade_03/
├── README.md
├── requirements.txt
├── data/processed/                    # dataset_regression.csv, dataset_classification.csv (da Atividade 1)
├── notebooks/
│   ├── 03_mlp_regressao.ipynb          # Protocolo, saída/perda, baseline de regressão (5 sementes)
│   ├── 04_mlp_classificacao.ipynb      # Desbalanceamento, saída/perda, baseline de classificação
│   ├── 05_hiperparametros.ipynb        # Estudos, mapas de calor, sensibilidade, busca aleatória, treino final
│   ├── 06_salvar_carregar_inferencia.ipynb  # Kernel limpo: state_dict x modelo inteiro, fidelidade, eval(), TorchScript/int8
│   ├── 07_avaliacao_generalizacao.ipynb     # Métricas, ROC, PR, limiar, resíduos, gap, K-fold, diagnóstico
│   └── 08_eficiencia_computacional.ipynb    # Tempo, memória, disco, MACs, energia/CO2, tabela-resumo
├── src/
│   ├── config.py           # Caminhos, sementes, divisão treino/validação/teste, padronização
│   ├── models.py           # Classe MLP parametrizável, nº de parâmetros, MACs
│   ├── train.py            # Baseline, laço de treino com early stopping, avaliação, checkpoint
│   ├── experiments.py      # Definição dos estudos, execução paralela com log, resumo e sensibilidade
│   ├── metrics.py          # Métricas da Aula 08, gap e diagnóstico
│   ├── inference.py        # Script independente: carrega o .pt e faz inferência
│   ├── profiling.py        # Tempo, memória (psutil), energia e CO2
│   └── visualization.py    # Curvas de perda, métrica x valor, mapas de calor
├── models/
│   ├── modelo_regressao.pt         # Checkpoint final de regressão
│   ├── modelo_classificacao.pt     # Checkpoint final de classificação
│   ├── baseline_regressao.pt       # Baseline (referência na avaliação)
│   └── baseline_classificacao.pt
├── exemplos/novas_amostras.csv     # 5 pacientes para demonstrar a inferência
├── logs/
│   ├── hiperparametros.csv         # Todas as execuções (configuração, semente, métricas, tempo)
│   ├── curvas.jsonl                # Perda de treino/validação por época de cada execução
│   └── saidas_em_memoria_*.pt      # Saídas do modelo antes de salvar (verificação de fidelidade)
├── reports/
│   ├── tables/                     # Resultados em CSV (resumo_final.csv = tabela-resumo)
│   └── figures/                    # Curvas, mapas de calor, ROC, PR, resíduos
└── docs/
    ├── relatorio_final.tex
    └── relatorio_final.pdf
```

## Como executar

```bash
cd atividade_03
pip install -r requirements.txt     # testado com Python 3.13 e PyTorch 2.14 (CPU)
jupyter notebook                    # execute os notebooks 03 a 08, nesta ordem
```

O notebook 05 reaproveita as execuções registradas em `logs/hiperparametros.csv` e só treina o que falta. Para
refazer o estudo do zero, apague `logs/hiperparametros.csv` e `logs/curvas.jsonl` (cerca de 40 minutos em 6
processos paralelos). O estudo também pode ser executado fora do notebook com `python -m src.experiments`.

## Inferência com o modelo salvo

Não é preciso retreinar: o script lê o checkpoint, aplica o mesmo pré-processamento e prevê.

```bash
cd atividade_03
python src/inference.py --modelo models/modelo_classificacao.pt --entrada exemplos/novas_amostras.csv
python src/inference.py --modelo models/modelo_regressao.pt --entrada exemplos/novas_amostras.csv --saida previsoes.csv
```

Em Python:

```python
from src import inference
modelo, ckpt = inference.carregar("models/modelo_classificacao.pt")   # map_location="cpu", weights_only=True
inference.prever(modelo, ckpt, {"age": 61, "sex": 1, ...})            # dicionário, lista de dicionários ou DataFrame
```

## Principais resultados

**Configurações finais** (escolhidas na validação: estudo de um fator por vez + busca aleatória com 30 configurações):

|                        | Regressão                                      | Classificação                                        |
| ---------------------- | ---------------------------------------------- | ---------------------------------------------------- |
| Arquitetura            | 2 × 32, Sigmoid (2.081 parâmetros)             | 1 × 8, Tanh (257 parâmetros)                         |
| Otimizador / η / batch | AdamW / 1,6e-4 / 16                            | RMSprop / 1,6e-3 / 16                                |
| Regularização          | weight decay 1e-4, dropout 0,1, early stopping | weight decay 1e-4, dropout 0,1, early stopping       |
| Perda                  | Huber                                          | BCEWithLogits, `pos_weight` = 1,25                   |
| Limiar                 | –                                              | 0,50 (máx. F1 na validação); 0,23 para recall ≥ 0,90 |

**Desempenho no teste** (1.800 pacientes, modelo carregado do `.pt`):

|                        | Baseline | Final                                   | Média de 5 sementes (final) | K-fold 5 (final) |
| ---------------------- | -------- | --------------------------------------- | --------------------------- | ---------------- |
| Regressão — RMSE (bpm) | 12,33    | **12,16** (R² 0,678)                    | 12,16 ± 0,01                | 11,92 ± 0,14     |
| Classificação — F1     | 0,809    | **0,822** (recall 0,811, ROC-AUC 0,950) | 0,820 ± 0,004               | 0,836 ± 0,016    |

- **Hiperparâmetros mais críticos:** a taxa de aprendizado (com η = 1 a rede colapsa num preditor constante, por
  _dying ReLU_) e o early stopping (sem ele, com 500 épocas: F1 de treino 1,000 e de validação 0,757). Batch e
  otimizador importam na classificação; neurônios, camadas, weight decay, dropout e função de perda tiveram efeito
  dentro da variação entre sementes (teste de Mann-Whitney).
- **Interação:** o melhor η depende do batch (batch 8: η = 1e-4 é o melhor e 1e-1 colapsa; batch total: 1e-4 não
  converge e 1e-1 é o melhor).
- **Generalização:** gap treino–teste de 3,4% (regressão) e 2,3% (classificação), menor que o do baseline; os dois
  modelos foram diagnosticados como bem ajustados.
- **Persistência:** o modelo recarregado reproduz as saídas **bit a bit** (`torch.equal` verdadeiro, diferença máxima 0)
  e a mesma métrica no teste.
- **Custo:** treino de 37 s (regressão) e 4 s (classificação) em 1 núcleo; ~0,03 ms por paciente; 16 e 6 KB em disco.
- **Comparação com a Atividade 2:** na classificação, a rede tem o melhor F1 de teste (0,822, contra 0,807 do Gradient
  Boosting). Na regressão, empata com a Random Forest (12,13) e perde por pouco para o Gradient Boosting (12,07). Uma
  regressão linear é melhor que todas as redes na validação (11,61 contra 11,67): o problema é quase linear, e a rede
  não se justifica para a regressão.

Tabela completa: [`reports/tables/resumo_final.csv`](reports/tables/resumo_final.csv).

## Relatório

O relatório está em LaTeX (`docs/relatorio_final.tex`, com as seções em `docs/secoes/` e as tabelas geradas a partir
dos CSVs por `docs/gerar_tabelas_latex.py`). Para compilar: `cd docs && pdflatex relatorio_final.tex` (duas vezes).
