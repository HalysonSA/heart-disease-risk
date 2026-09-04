# Dicionário de Dados - Dataset de Risco de Doença Cardíaca

## Explicação das Variáveis em Português

### 📋 Informações Básicas
| Variável | Tradução | Explicação |
|----------|----------|-----------|
| `patient_id` | ID do Paciente | Número único de identificação do paciente |
| `age` | Idade | Idade do paciente em anos |
| `sex` | Sexo | Sexo biológico (Male = Masculino, Female = Feminino) |

### ❤️ Dados Cardiovasculares (Medidas do Coração)
| Variável | Tradução | Explicação |
|----------|----------|-----------|
| `resting_bp_systolic` | Pressão Sistólica em Repouso | Pressão máxima nas artérias (número maior da pressão) - Normal: < 120 mmHg |
| `resting_bp_diastolic` | Pressão Diastólica em Repouso | Pressão mínima nas artérias (número menor da pressão) - Normal: < 80 mmHg |
| `resting_heart_rate` | Frequência Cardíaca em Repouso | Quantas vezes o coração bate por minuto em repouso - Normal: 60-100 bpm |
| `max_heart_rate_achieved` | Frequência Cardíaca Máxima Alcançada | Máxima frequência cardíaca durante teste de esforço |
| `chest_pain_type` | Tipo de Dor no Peito | Tipo de desconforto no peito (Typical Angina, Atypical Angina, Non-Anginal Pain, Asymptomatic) |
| `exercise_induced_angina` | Angina Induzida por Exercício | Se aparece dor no peito durante exercício (True/False) |
| `st_depression` | Depressão do Segmento ST | Medida de um traço do eletrocardiograma (pode indicar problemas cardíacos) |

### 🩸 Dados de Colesterol e Açúcar no Sangue
| Variável | Tradução | Explicação |
|----------|----------|-----------|
| `cholesterol_total` | Colesterol Total | Quantidade total de colesterol no sangue - Ideal: < 200 mg/dL |
| `hdl` | HDL (Colesterol Bom) | Colesterol bom que protege o coração - Ideal: > 40 mg/dL (homens), > 50 (mulheres) |
| `ldl` | LDL (Colesterol Ruim) | Colesterol ruim que prejudica o coração - Ideal: < 100 mg/dL |
| `triglycerides` | Triglicerídeos | Tipo de gordura no sangue - Ideal: < 150 mg/dL |
| `fasting_blood_sugar` | Açúcar em Jejum | Nível de açúcar no sangue em jejum > 120 mg/dL (True = alto) |
| `hba1c` | HbA1c (Hemoglobina Glicada) | Média de açúcar no sangue nos últimos 3 meses - Normal: < 5.7% |

### ⚖️ Medidas Corporais
| Variável | Tradução | Explicação |
|----------|----------|-----------|
| `bmi` | IMC (Índice de Massa Corporal) | Peso relativo à altura - Normal: 18.5-24.9, Sobrepeso: 25-29.9, Obeso: >30 |

### 🚬 Fatores de Risco
| Variável | Tradução | Explicação |
|----------|----------|-----------|
| `smoker_status` | Status de Fumante | Never (Nunca), Current (Atual), Former (Antigo) |
| `family_history` | Histórico Familiar | Se há histórico de doença cardíaca na família (True/False) |
| `alcohol_units_per_week` | Unidades de Álcool por Semana | Quantidade de álcool consumida por semana |

### 🏃 Estilo de Vida
| Variável | Tradução | Explicação |
|----------|----------|-----------|
| `exercise_minutes_per_week` | Minutos de Exercício por Semana | Tempo dedicado a atividades físicas |
| `daily_steps` | Passos Diários | Quantidade média de passos por dia |
| `sleep_hours` | Horas de Sono | Quanto dorme por noite em média |
| `wearable_owner` | Usa Relógio Inteligente | Se usa pulseira/relógio para monitorar atividades (True/False) |

### 🧠 Saúde Mental
| Variável | Tradução | Explicação |
|----------|----------|-----------|
| `stress_score` | Escore de Estresse | Nível de estresse de 0-100 (quanto maior, mais estressado) |
| `diet_quality_score` | Escore de Qualidade da Dieta | Qualidade da alimentação de 0-100 (quanto maior, melhor) |

### 🎯 Variável Alvo (O que queremos prever)
| Variável | Tradução | Explicação |
|----------|----------|-----------|
| `has_heart_disease` | Tem Doença Cardíaca | 0 = Não tem doença, 1 = Tem doença (Classificação) |

---

## Dicas para Entender Melhor

### Pressão Arterial
- **Sistólica/Diastólica**: É escrito como "120/80" (sistólica sobre diastólica)
- **Sistólica**: Pressão quando o coração bate (número maior)
- **Diastólica**: Pressão quando o coração relaxa (número menor)

### Colesterol
- **Total**: Som de todos os tipos
- **HDL**: "High-Density Lipoprotein" = Colesterol PROTETOR (quanto mais, melhor)
- **LDL**: "Low-Density Lipoprotein" = Colesterol PREJUDICIAL (quanto menos, melhor)
- **Triglicer​ídeos**: Gordura no sangue (quanto menos, melhor)

### Frequência Cardíaca
- Medida em **BPM** (batidas por minuto)
- Em repouso é mais baixa
- Durante exercício é mais alta

### IMC (BMI)
- Fórmula: Peso (kg) ÷ Altura (m)²
- Exemplo: Uma pessoa com 70kg e 1.70m = 70 ÷ (1.70)² = 24.2 (Normal)

### HbA1c
- Também chamado de hemoglobina glicada
- Mostra a média de glicose nos últimos 3 meses
- Se está alto, significa que tem diabetes ou pré-diabetes

---

## Relação com Doença Cardíaca ❤️

**Aumentam o Risco:**
- ⚠️ Pressão alta (hipertensão)
- ⚠️ Colesterol alto (especialmente LDL)
- ⚠️ IMC alto (sobrepeso/obesidade)
- ⚠️ Fumar
- ⚠️ Estresse alto
- ⚠️ Pouca atividade física
- ⚠️ Histórico familiar

**Diminuem o Risco:**
- ✅ Pressão normal
- ✅ Colesterol baixo
- ✅ IMC saudável
- ✅ Não fumar
- ✅ Estresse baixo
- ✅ Exercício regular
- ✅ Boa qualidade de sono
- ✅ Boa qualidade de dieta
