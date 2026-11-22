"""Inferência com um modelo salvo (.pt), sem reutilizar nada do treinamento.

Uso pela linha de comando (a partir da pasta atividade_03/):

    python src/inference.py --modelo models/modelo_classificacao.pt --entrada exemplos/novas_amostras.csv
    python src/inference.py --modelo models/modelo_regressao.pt --entrada exemplos/novas_amostras.csv --saida previsoes.csv
    python src/inference.py --modelo models/modelo_regressao.pt --json "{\"age\": 60, ...}"

O checkpoint contém tudo o que é preciso para reproduzir as previsões: state_dict, a
configuração da arquitetura, média e desvio do StandardScaler (e do target, na regressão),
a lista de features na ordem usada e, na classificação, as classes e o limiar de decisão.
Ele é lido com torch.load(weights_only=True), que só aceita tensores e tipos básicos e,
por isso, não executa código arbitrário embutido no arquivo.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:  # importado como pacote (src.inference) ou executado como script (python src/inference.py)
    from .models import MLP
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from models import MLP


def carregar(caminho, map_location="cpu"):
    """Recria a arquitetura a partir do checkpoint, carrega os pesos e coloca em modo eval.

    map_location="cpu" remapeia para a CPU tensores que tenham sido salvos em GPU (sem ele,
    um checkpoint salvo em 'cuda:0' falharia numa máquina sem GPU)."""
    ckpt = torch.load(caminho, map_location=map_location, weights_only=True)
    cfg = ckpt["config"]
    modelo = MLP(cfg["in_dim"], cfg["out_dim"], tuple(cfg["hidden"]), cfg["activation"], cfg["dropout"])
    modelo.load_state_dict(ckpt["state_dict"])
    modelo.eval()  # desliga o dropout (e fixaria a batch normalization, se houvesse)
    return modelo, ckpt


def preparar_entrada(ckpt, amostras):
    """Seleciona as features na ordem do treino e aplica a MESMA padronização do treino.
    `amostras` pode ser um DataFrame, um dicionário (uma amostra) ou uma lista de dicionários.
    Colunas extras (ex.: patient_id ou o próprio target) são ignoradas."""
    if isinstance(amostras, dict):
        amostras = [amostras]
    df = pd.DataFrame(amostras).copy()
    df.columns = df.columns.str.replace(r"[ -]", "_", regex=True)
    faltando = [f for f in ckpt["features"] if f not in df.columns]
    if faltando:
        raise ValueError(f"Features ausentes na entrada: {faltando}")
    # Padroniza em float64 (como o StandardScaler) e só depois converte para float32, a precisão
    # da rede. Assim a entrada fica idêntica, bit a bit, à usada no treino.
    x = torch.tensor(df[ckpt["features"]].to_numpy(np.float64), dtype=torch.float64)
    x = (x - ckpt["x_media"].double()) / ckpt["x_desvio"].double()
    return x.float()


def prever(modelo, ckpt, amostras):
    """Retorna um DataFrame com as previsões.
    Regressão: valor previsto na escala original (bpm).
    Classificação: probabilidade da classe positiva (sigmoide do logit) e classe prevista."""
    x = preparar_entrada(ckpt, amostras)
    with torch.inference_mode():
        saida = modelo(x).squeeze(1)
    if ckpt["problema"] == "regressao":
        # mesma operação usada no treino (Dados.para_original), para reproduzir os valores exatamente
        y = saida.numpy() * ckpt["y_desvio"] + ckpt["y_media"]
        return pd.DataFrame({f"{ckpt['target']}_previsto": y})
    prob = torch.sigmoid(saida)
    classe = (prob >= ckpt["limiar"]).long()
    return pd.DataFrame({"prob_" + ckpt["classes"][1]: prob.numpy(),
                         f"{ckpt['target']}_previsto": classe.numpy(),
                         "classe": [ckpt["classes"][c] for c in classe.tolist()]})


def main():
    ap = argparse.ArgumentParser(description="Inferência com o modelo PyTorch salvo.")
    ap.add_argument("--modelo", required=True, help="caminho do checkpoint .pt")
    grupo = ap.add_mutually_exclusive_group(required=True)
    grupo.add_argument("--entrada", help="CSV com as novas amostras")
    grupo.add_argument("--json", help="amostra(s) em JSON (objeto ou lista de objetos)")
    ap.add_argument("--saida", help="CSV para gravar as previsões (opcional)")
    args = ap.parse_args()

    modelo, ckpt = carregar(args.modelo)
    amostras = pd.read_csv(args.entrada) if args.entrada else json.loads(args.json)
    prev = prever(modelo, ckpt, amostras)

    if args.entrada:
        ids = amostras["patient_id"] if "patient_id" in amostras.columns else None
        if ids is not None:
            prev.insert(0, "patient_id", ids.to_numpy())
    print(f"Modelo: {args.modelo} ({ckpt['problema']}, target = {ckpt['target']})")
    print(prev.to_string(index=False))
    if args.saida:
        prev.to_csv(args.saida, index=False)
        print(f"Previsões gravadas em {args.saida}")


if __name__ == "__main__":
    main()
