"""Sistema especialista baseado em regras SE-ENTÃO com encadeamento para frente.

A BASE DE CONHECIMENTO fica em arquivos texto (rules/*.txt) e o MOTOR DE INFERÊNCIA
fica aqui, separados como na Aula 05. Sintaxe de cada linha do arquivo de regras:

    ID | TIPO | SE <condição> [E <condição> ...] ENTAO <fato> = <valor>
    ID | TIPO | SE <condição> [OU <condição> ...] ENTAO <fato> += <valor>
    PADRAO <fato> = <valor>

- TIPO: relacao, recomendacao, diretriz, estrategia ou heuristica (Durkin, 1994).
- <condição>: <fato> <op> <valor>, com op em ==, !=, >, >=, <, <=.
- Uma regra usa só E ou só OU (misturar é proibido e gera erro).
- `=` atribui o fato; `+=` acumula partindo do valor atual, ou do PADRAO do fato, ou 0
  (ex.: PADRAO = média do alvo e regras somam ajustes; ou pontuação de risco).
- Os fatos iniciais são as colunas da amostra; regras podem criar fatos intermediários
  (ex.: `risco = alto`) usados por outras regras.
- O valor da conclusão pode ser uma constante ou uma expressão aritmética sobre os fatos
  (ex.: `max_heart_rate_achieved = 208 - 0.7 * age`, fórmula de Tanaka).
- PADRAO define o valor de um fato caso nenhuma regra o tenha concluído.
- Resolução de conflito: ordem no arquivo (prioridade) e cada regra dispara no máximo
  uma vez por amostra (refração). O ciclo repete até nenhuma regra nova disparar.
"""
import operator
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TIPOS = {"relacao", "recomendacao", "diretriz", "estrategia", "heuristica"}
OPS = {"==": operator.eq, "!=": operator.ne, ">=": operator.ge,
       "<=": operator.le, ">": operator.gt, "<": operator.lt}
_COND = re.compile(r"^\s*(\w+)\s*(==|!=|>=|<=|>|<)\s*(.+?)\s*$")
_ACAO = re.compile(r"^\s*(\w+)\s*(\+=|=)\s*(.+?)\s*$")


class Expressao:
    """Expressão aritmética (+ - * / parênteses) sobre fatos, avaliada no disparo da regra."""

    def __init__(self, texto):
        self.texto = texto
        self.codigo = compile(texto, "<regra>", "eval")

    def avaliar(self, fatos):
        return float(eval(self.codigo, {"__builtins__": {}}, fatos))


def _valor(txt):
    txt = txt.strip().strip("'\"")
    try:
        return float(txt)
    except ValueError:
        return txt


@dataclass
class Regra:
    id: str
    tipo: str
    condicoes: list      # [(fato, op, valor)]
    conectivo: str       # "E" ou "OU"
    fato: str
    acao: str            # "=" ou "+="
    valor: object
    texto: str

    def satisfeita(self, fatos):
        resultados = (f in fatos and _comparar(fatos[f], op, v) for f, op, v in self.condicoes)
        return all(resultados) if self.conectivo == "E" else any(resultados)

    def executar(self, fatos, padroes):
        valor = self.valor.avaliar(fatos) if isinstance(self.valor, Expressao) else self.valor
        if self.acao == "+=":
            fatos[self.fato] = fatos.get(self.fato, padroes.get(self.fato, 0)) + valor
        else:
            fatos[self.fato] = valor


def _comparar(a, op, b):
    try:
        return OPS[op](a, b)
    except TypeError:  # ex.: comparar texto com número
        return False


class BaseConhecimento:
    """Conjunto de regras e valores padrão lidos de um arquivo texto."""

    def __init__(self, regras, padroes, caminho=None):
        self.regras = regras
        self.padroes = padroes
        self.caminho = Path(caminho) if caminho else None

    @classmethod
    def de_arquivo(cls, caminho):
        texto = Path(caminho).read_text(encoding="utf-8")
        return cls(*_parse(texto), caminho=caminho)

    @classmethod
    def de_texto(cls, texto):
        return cls(*_parse(texto))

    def __len__(self):
        return len(self.regras)

    def variaveis_usadas(self):
        """Fatos citados nas condições (para comparar com a importância das features)."""
        return sorted({f for r in self.regras for f, _, _ in r.condicoes})


def _parse(texto):
    regras, padroes, ids = [], {}, set()
    for n, linha in enumerate(texto.splitlines(), 1):
        linha = linha.split("#", 1)[0].strip()
        if not linha:
            continue
        if linha.upper().startswith("PADRAO"):
            fato, acao, valor = _ACAO.match(linha[6:]).groups()
            padroes[fato] = _valor(valor)
            continue
        partes = [p.strip() for p in linha.split("|")]
        if len(partes) != 3:
            raise ValueError(f"Linha {n}: esperado 'ID | TIPO | SE ... ENTAO ...'")
        rid, tipo, corpo = partes
        tipo = tipo.lower().replace("ç", "c").replace("ã", "a").replace("é", "e").replace("í", "i")
        if tipo not in TIPOS:
            raise ValueError(f"Linha {n}: tipo '{tipo}' inválido, use um de {sorted(TIPOS)}")
        if rid in ids:
            raise ValueError(f"Linha {n}: id '{rid}' repetido")
        m = re.match(r"^SE\s+(.+?)\s+ENT(?:A|Ã)O\s+(.+)$", corpo, re.IGNORECASE)
        if not m:
            raise ValueError(f"Linha {n}: esperado 'SE ... ENTAO ...'")
        antecedente, consequente = m.groups()
        tem_e = re.search(r"\sE\s", f" {antecedente} ") is not None
        tem_ou = re.search(r"\sOU\s", f" {antecedente} ") is not None
        if tem_e and tem_ou:
            raise ValueError(f"Linha {n}: não misture E e OU na mesma regra")
        conectivo = "OU" if tem_ou else "E"
        condicoes = []
        for c in re.split(rf"\s{conectivo}\s", antecedente):
            mc = _COND.match(c)
            if not mc:
                raise ValueError(f"Linha {n}: condição inválida '{c}'")
            f, op, v = mc.groups()
            condicoes.append((f, op, _valor(v)))
        ma = _ACAO.match(consequente)
        if not ma:
            raise ValueError(f"Linha {n}: conclusão inválida '{consequente}'")
        fato, acao, valor = ma.groups()
        valor = _valor(valor)
        if isinstance(valor, str) and re.search(r"[-+*/()]", valor):
            valor = Expressao(valor)
        if acao == "+=" and isinstance(valor, str):
            raise ValueError(f"Linha {n}: '+=' exige valor numérico")
        ids.add(rid)
        regras.append(Regra(rid, tipo, condicoes, conectivo, fato, acao, valor, linha))
    return regras, padroes


class MotorInferencia:
    """Motor de encadeamento para frente (forward chaining)."""

    def __init__(self, base):
        self.base = base

    def inferir(self, fatos):
        """Recebe um dict de fatos iniciais; retorna (fatos finais, ids das regras disparadas)."""
        fatos = dict(fatos)
        disparadas = []
        mudou = True
        while mudou:
            mudou = False
            for r in self.base.regras:
                if r.id not in disparadas and r.satisfeita(fatos):
                    r.executar(fatos, self.base.padroes)
                    disparadas.append(r.id)
                    mudou = True
        for fato, valor in self.base.padroes.items():
            fatos.setdefault(fato, valor)
        return fatos, disparadas

    def prever(self, X, alvo):
        """Aplica o motor a cada linha de um DataFrame e devolve o fato `alvo`."""
        return np.array([self.inferir(f)[0].get(alvo, np.nan) for f in X.to_dict("records")],
                        dtype=float)

    def pontuacao(self, X, fato="score"):
        """Valor do fato acumulado `fato` (ex.: score de risco) por amostra, ou None se as
        regras nunca o produzem. Serve como pontuação para curvas ROC/PR."""
        valores = [self.inferir(f)[0].get(fato) for f in X.to_dict("records")]
        if all(v is None for v in valores):
            return None
        return np.array([0.0 if v is None else v for v in valores], dtype=float)

    def cobertura(self, X):
        """Fração das amostras em que cada regra disparou."""
        contagem = {r.id: 0 for r in self.base.regras}
        registros = X.to_dict("records")
        for f in registros:
            for rid in self.inferir(f)[1]:
                contagem[rid] += 1
        return {rid: c / len(registros) for rid, c in contagem.items()}


if __name__ == "__main__":
    # Auto-teste do motor (rodar: python src/expert_system.py)
    base = BaseConhecimento.de_texto("""
        PADRAO doenca = 0
        R1 | relacao     | SE angina == 1 E st >= 2 ENTAO risco = alto
        R2 | heuristica  | SE risco == alto ENTAO doenca = 1
        R3 | heuristica  | SE idade > 60 OU st >= 3 ENTAO score += 0.5
        R4 | diretriz    | SE angina == 1 ENTAO score += 0.25
    """)
    motor = MotorInferencia(base)
    ajuste = MotorInferencia(BaseConhecimento.de_texto(
        "PADRAO fc = 160\nR1 | heuristica | SE ex >= 200 ENTAO fc += 8"))
    assert ajuste.inferir({"ex": 250})[0]["fc"] == 168 and ajuste.inferir({"ex": 0})[0]["fc"] == 160
    tanaka = MotorInferencia(BaseConhecimento.de_texto(
        "R1 | relacao | SE age > 0 ENTAO fc = 208 - 0.7 * age"))
    assert abs(tanaka.inferir({"age": 40})[0]["fc"] - 180) < 1e-9
    f, d = motor.inferir({"angina": 1, "st": 2.5, "idade": 70})
    assert f["risco"] == "alto" and f["doenca"] == 1 and f["score"] == 0.75, f
    assert d == ["R1", "R2", "R3", "R4"], d
    f, d = motor.inferir({"angina": 0, "st": 0.0, "idade": 30})
    assert f["doenca"] == 0 and "score" not in f and d == [], (f, d)
    assert base.variaveis_usadas() == ["angina", "idade", "risco", "st"]
    for ruim in ["R1 | relacao | SE a == 1 E b == 1 OU c == 1 ENTAO x = 1",
                 "R1 | palpite | SE a == 1 ENTAO x = 1",
                 "R1 | relacao | SE a == 1 ENTAO x += alto"]:
        try:
            BaseConhecimento.de_texto(ruim)
            raise AssertionError(f"deveria falhar: {ruim}")
        except ValueError:
            pass
    print("OK: motor de inferência passou no auto-teste")
