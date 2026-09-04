# Análise Exploratória e Preparação de Dados para Modelos de Aprendizado de Máquina

**Aluno:** Halyson Santos  
**Matrícula:** 20220062812
**Disciplina:** DCT1401 - Inteligência Artificial  
**Período:** 2026.2  
**Professor:** Thommas K. S. Flores  
**Instituição:** Universidade Federal do Rio Grande do Norte - UFRN  
**Departamento:** Departamento de Computação e Tecnologia (DCT)

---

## 📋 Descrição do Projeto

Este repositório contém a análise exploratória e o pré-processamento de dados para a atividade prática avaliativa da disciplina **DCT1401 - Inteligência Artificial**, realizada no período 2026.2.

O projeto utiliza um dataset de **risco de doenças cardíacas** contendo informações clínicas e de estilo de vida de aproximadamente 1.100 pacientes. O objetivo principal é preparar esses dados para construção de dois modelos de aprendizado de máquina com paradigmas diferentes:

1. **Modelo de Regressão**: Prever `max_heart_rate_achieved` (Frequência Cardíaca Máxima Alcançada)
2. **Modelo de Classificação**: Prever `has_heart_disease` (Presença de Doença Cardíaca: Sim/Não)

---

## 🎯 Objetivos da Atividade

Conduzir uma análise exploratória de dados completa, realizar limpeza e pré-processamento necessários e interpretar os dados para propor soluções adequadas para dois paradigmas de aprendizado:

- **Regressão** (supervisionado): Previsão de variável numérica contínua
- **Classificação** (supervisionado): Previsão de variável categórica binária

Todos os resultados devem ser documentados em um relatório técnico que trace o processo desde a compreensão inicial dos dados até as percepções finais que guiarão a modelagem.

---

## 📊 Dataset

- **Nome:** Heart Disease Risk 2026
- **Tamanho:** ~1.100 registros de pacientes
- **Variáveis:** 27 colunas (26 features + 1 target)
- **Localização:** `data/raw/dataset.csv`
- **Tema:** Risco cardiovascular e saúde cardíaca

### Variáveis Principais

| Categoria                  | Exemplos                                                        |
| -------------------------- | --------------------------------------------------------------- |
| **Dados Cardiovasculares** | Pressão arterial, frequência cardíaca, tipo de dor no peito     |
| **Colesterol e Açúcar**    | Colesterol total, HDL, LDL, triglicerídeos, hemoglobina glicada |
| **Dados Corporais**        | Idade, IMC (peso/altura)                                        |
| **Fatores de Risco**       | Histórico de fumante, histórico familiar                        |
| **Estilo de Vida**         | Exercício semanal, passos diários, horas de sono, estresse      |

**📖 Para tradução completa: ver arquivo `DICIONARIO_DADOS.md`**

---

## 📁 Estrutura do Repositório

```
│
├── README.md                                  # Este arquivo
├── DICIONARIO_DADOS.md                       # Tradução de variáveis em português
├── requirements.txt                           # Dependências Python
│
├── data/
│   ├── raw/
│   │   └── dataset.csv                       # Dataset original (não modificar)
│   │
│   └── processed/
│       ├── dataset_clean.csv                 # Dataset limpo
│       ├── dataset_scaled.csv                # Dataset padronizado
│       ├── dataset_regression.csv            # Dataset para regressão
│       └── dataset_classification.csv        # Dataset para classificação
│
├── notebooks/
│   ├── 01_eda.ipynb                          # Análise Exploratória de Dados
│   ├── 02_preprocessing.ipynb                # Limpeza e Pré-processamento
│   ├── 03_regression_analysis.ipynb          # Percepções para Regressão
│   └── 04_classification_analysis.ipynb      # Percepções para Classificação
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                        # Funções para carregar dados
│   ├── eda.py                                # Funções para análise exploratória
│   ├── preprocessing.py                      # Funções para limpeza e pré-processamento
│   └── visualization.py                      # Funções para criação de gráficos
│
├── reports/
│   ├── figures/
│   │   ├── correlation_matrix.png            # Matriz de correlação
│   │   ├── missing_values.png                # Valores ausentes
│   │   └── outliers_boxplot.png              # Análise de outliers
│   │
│   └── relatorio_final.pdf                   # Relatório técnico da atividade
│
└── docs/
    └── [Documentação adicional]
```

---

## ⚙️ Como Executar

### Pré-requisitos

- Python 3.13 ou superior
- pip (gerenciador de pacotes Python)

### Passo 1: Instalar Dependências

Abra o terminal/PowerShell na pasta do projeto e execute:

```bash
pip install -r requirements.txt
```

Isto instalará as seguintes bibliotecas:

- **pandas**: Manipulação de dados
- **numpy**: Computação numérica
- **matplotlib**: Visualização de gráficos
- **seaborn**: Gráficos estatísticos avançados
- **scikit-learn**: Algoritmos de machine learning
- **jupyter**: Ambiente interativo de notebooks

### Passo 2: Abrir Jupyter Notebook

No mesmo terminal, execute:

```bash
jupyter notebook
```

Isto abrirá uma janela no seu navegador padrão mostrando os arquivos do projeto.

### Passo 3: Executar os Notebooks na Ordem Recomendada

**Clique em cada notebook na ordem abaixo e execute as células usando `Shift + Enter`:**

#### 1️⃣ **01_eda.ipynb** - Análise Exploratória de Dados (25-30 min)

- Carregamento e inspeção inicial dos dados
- Estatísticas descritivas (média, mediana, desvio padrão)
- Identificação de valores ausentes
- Detecção de outliers (método IQR)
- Análise de correlação entre variáveis
- Visualizações: histogramas, boxplots, scatter plots

#### 2️⃣ **02_preprocessing.ipynb** - Limpeza e Pré-processamento (15-20 min)

- Tratamento de valores ausentes (neste dataset: nenhum)
- Decisão sobre outliers (mantém-se dados reais)
- Codificação de variáveis categóricas:
  - `sex`: Label Encoding (Male=1, Female=0)
  - `chest_pain_type`: One-Hot Encoding
  - `smoker_status`: One-Hot Encoding
  - Booleanas: Conversão para 0/1
- Padronização (Z-score) para modelos que necessitam
- Geração de datasets processados

#### 3️⃣ **03_regression_analysis.ipynb** - Análise para Regressão (20-25 min)

- **Variável Alvo:** `max_heart_rate_achieved` (Frequência Cardíaca Máxima)
- Análise estatística do target
- Correlação com features (independentes)
- Visualizações de relacionamento
- Verificação de pressupostos de regressão
- **Recomendações:** Algoritmos mais adequados
- Features mais relevantes

#### 4️⃣ **04_classification_analysis.ipynb** - Análise para Classificação (20-25 min)

- **Variável Alvo:** `has_heart_disease` (Tem Doença: 0 ou 1)
- Análise de distribuição das classes
- Diferenciação entre grupos (com vs sem doença)
- Testes ANOVA para poder preditivo
- Visualizações comparativas
- **Recomendações:** Algoritmos mais adequados
- Features mais relevantes

---

## 📊 Principais Resultados

### Regressão (max_heart_rate_achieved)

**Variável Alvo Escolhida:** Frequência Cardíaca Máxima Alcançada

**Justificativa:**

- Variável numérica contínua com variabilidade suficiente (range: ~90-220 bpm)
- Forte relevância clínica: indica capacidade cardiovascular
- Correlaciona com múltiplas features (idade, exercício, nível de estresse)
- Utilidade prática: essencial para prescrição de atividades físicas

**Top Features Correlacionadas:**

1. Idade (age)
2. Frequência cardíaca em repouso (resting_heart_rate)
3. IMC (bmi)
4. Minutos de exercício semanal (exercise_minutes_per_week)
5. Escore de estresse (stress_score)

**Algoritmos Recomendados:**

- Linear Regression, Ridge, Lasso (simples)
- Decision Trees, Random Forest (intermediário)
- SVR, Gradient Boosting (avançado)

---

### Classificação (has_heart_disease)

**Variável Alvo Escolhida:** Presença de Doença Cardíaca (Binária)

**Justificativa:**

- Variável categórica binária (0 = sem doença, 1 = com doença)
- Altamente relevante clinicamente: diagnóstico direto
- Balanceamento de classes: aceitável para classificação
- Múltiplas features diferenciam as classes significativamente

**Distribuição das Classes:**

- Sem doença (0): ~52% dos pacientes
- Com doença (1): ~48% dos pacientes
- Balanceamento: 92% (muito bom para classificação)

**Top Features Discriminativas:**

1. Tipo de dor no peito (chest_pain_type)
2. Depressão do segmento ST (st_depression)
3. Exercício induzido angina (exercise_induced_angina)
4. Histórico familiar (family_history)
5. Escore de estresse (stress_score)

**Algoritmos Recomendados:**

- Logistic Regression, Naive Bayes (simples)
- Decision Trees, Random Forest, SVM (intermediário)
- XGBoost, LightGBM (avançado)

---

## 📈 Visualizações Geradas

As seguintes visualizações são geradas automaticamente durante a execução:

- **correlation_matrix.png**: Matriz de correlação de todas as variáveis numéricas
- **missing_values.png**: Gráfico de valores ausentes por coluna
- **outliers_boxplot.png**: Boxplots para detecção de outliers

Salvas em: `reports/figures/`

---

## 📝 Relatório Técnico

O relatório final em PDF (`relatorio_final.pdf`) contém:

1. **Capa**: Informações da instituição, disciplina e aluno
2. **Introdução**: Descrição do objetivo e dataset
3. **Análise Exploratória**: Estatísticas, visualizações e interpretação
4. **Limpeza e Pré-processamento**: Técnicas aplicadas e justificativas
5. **Percepções para Regressão**: Target, features e recomendações
6. **Percepções para Classificação**: Target, features e recomendações
7. **Considerações Finais**: Dificuldades e aprendizados
8. **Referência ao Repositório**: Link e estrutura do código

---

## 📚 Recursos Adicionais

- **DICIONARIO_DADOS.md**: Tradução completa de todas as variáveis e suas explicações
- **PDF original da atividade**: `main_1.pdf`
- **Dataset original**: `data/raw/dataset.csv`

---

## 🔧 Dicas Importantes

### Ordem Executável

Os notebooks devem ser executados **nessa ordem específica**:

```
01_eda.ipynb → 02_preprocessing.ipynb → 03_regression_analysis.ipynb → 04_classification_analysis.ipynb
```

### Sem Erros

Cada notebook foi testado e é executável sem erros.

### Gráficos Salvos

Os gráficos gerados são **automaticamente salvos** em `reports/figures/` e referenciados no relatório.

### Comentários no Código

Todos os notebooks contêm comentários em português explicando cada etapa.

---

## 📞 Contato

- **Email:** halysonsa@gmail.com
- **Disciplina:** DCT1401 - Inteligência Artificial
- **Professor:** Thommas K. S. Flores

---

## ✅ Checklist de Execução

- [ ] Instalar dependências com `pip install -r requirements.txt`
- [ ] Abrir Jupyter com `jupyter notebook`
- [ ] Executar `01_eda.ipynb`
- [ ] Executar `02_preprocessing.ipynb`
- [ ] Executar `03_regression_analysis.ipynb`
- [ ] Executar `04_classification_analysis.ipynb`
- [ ] Revisar gráficos em `reports/figures/`
- [ ] Gerar relatório final em PDF

---

**Última atualização:** 2026-09-04  
**Versão do Projeto:** 1.0
