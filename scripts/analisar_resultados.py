from __future__ import annotations

import argparse
import csv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Any


METRICAS = (
    "iou",
    "f1",
    "dice",
    "precisao",
    "revocacao",
    "especificidade",
    "acuracia",
)

ROTULOS_METRICAS = {
    "iou": "IoU",
    "f1": "F1/Dice",
    "dice": "Dice",
    "precisao": "Precisao",
    "revocacao": "Revocacao",
    "especificidade": "Especificidade",
    "acuracia": "Acuracia",
}

MODELOS = ("original", "aumentado")
RAIZ_PROJETO = Path(__file__).resolve().parent.parent


def ler_csv(caminho: Path) -> list[dict[str, str]]:
    with caminho.open("r", encoding="utf-8-sig", newline="") as arquivo:
        return list(csv.DictReader(arquivo))


def numero(linha: dict[str, str], chave: str) -> float:
    valor = linha.get(chave)
    if valor is None or valor == "":
        raise KeyError(f"Coluna ausente ou vazia: {chave}")
    return float(valor)


def inteiro(linha: dict[str, str], chave: str) -> int:
    return int(round(numero(linha, chave)))


def metricas_de_contagens(tp: int, tn: int, fp: int, fn: int) -> dict[str, float]:
    eps = 1e-12

    # As convencoes abaixo sao compativeis com a avaliacao do pipeline:
    # - IoU/F1 = 1 quando referencia e predicao sao ambas vazias.
    # - Precisao = 1 quando nao ha positivos previstos.
    # - Revocacao = 1 quando nao ha positivos reais.
    if tp + fp + fn == 0:
        iou = 1.0
        f1 = 1.0
    else:
        iou = tp / (tp + fp + fn + eps)
        f1 = (2 * tp) / (2 * tp + fp + fn + eps)

    precisao = 1.0 if tp + fp == 0 else tp / (tp + fp + eps)
    revocacao = 1.0 if tp + fn == 0 else tp / (tp + fn + eps)
    especificidade = 1.0 if tn + fp == 0 else tn / (tn + fp + eps)
    acuracia = (tp + tn) / (tp + tn + fp + fn + eps)

    return {
        "iou": iou,
        "f1": f1,
        "dice": f1,
        "precisao": precisao,
        "revocacao": revocacao,
        "especificidade": especificidade,
        "acuracia": acuracia,
    }


def agregar_por_classe(
    linhas: list[dict[str, str]],
    destino: Path,
) -> list[dict[str, Any]]:
    classes = sorted({linha["classe"] for linha in linhas})
    saida: list[dict[str, Any]] = []

    for classe in classes:
        subconjunto = [linha for linha in linhas if linha["classe"] == classe]

        for modelo in MODELOS:
            prefixo = f"{modelo}_"

            # Macro: calcula a media das metricas ja obtidas para cada imagem.
            macro = {
                metrica: float(
                    np.mean([numero(linha, prefixo + metrica) for linha in subconjunto])
                )
                for metrica in METRICAS
            }
            saida.append(
                {
                    "classe": classe,
                    "modelo": modelo,
                    "agregacao": "macro_imagens",
                    "n": len(subconjunto),
                    "tp": sum(inteiro(linha, prefixo + "tp") for linha in subconjunto),
                    "tn": sum(inteiro(linha, prefixo + "tn") for linha in subconjunto),
                    "fp": sum(inteiro(linha, prefixo + "fp") for linha in subconjunto),
                    "fn": sum(inteiro(linha, prefixo + "fn") for linha in subconjunto),
                    **macro,
                }
            )

            # Micro: soma TP/TN/FP/FN de toda a classe e calcula as metricas depois.
            tp = sum(inteiro(linha, prefixo + "tp") for linha in subconjunto)
            tn = sum(inteiro(linha, prefixo + "tn") for linha in subconjunto)
            fp = sum(inteiro(linha, prefixo + "fp") for linha in subconjunto)
            fn = sum(inteiro(linha, prefixo + "fn") for linha in subconjunto)
            micro = metricas_de_contagens(tp, tn, fp, fn)

            saida.append(
                {
                    "classe": classe,
                    "modelo": modelo,
                    "agregacao": "micro_pixels",
                    "n": len(subconjunto),
                    "tp": tp,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    **micro,
                }
            )

    destino.parent.mkdir(parents=True, exist_ok=True)
    campos = [
        "classe",
        "modelo",
        "agregacao",
        "n",
        "tp",
        "tn",
        "fp",
        "fn",
        *METRICAS,
    ]
    with destino.open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(saida)

    return saida



def plotar_metricas_por_classe(
    agregados: list[dict[str, Any]],
    destino: Path,
    agregacao: str = "macro_imagens",
) -> None:
    """
    Plota, por classe, as metricas dos modelos original e aumentado.

    Por padrao usa macro_imagens, isto e, calcula-se cada metrica em cada
    imagem e depois tira-se a media dentro da classe.
    """
    classes = sorted({linha["classe"] for linha in agregados})
    metricas = ("iou", "f1", "precisao", "revocacao", "especificidade", "acuracia")
    rotulos = [ROTULOS_METRICAS[m] for m in metricas]

    figura, eixos = plt.subplots(
        1,
        len(classes),
        figsize=(6 * len(classes), 5.5),
        sharey=True,
    )
    if len(classes) == 1:
        eixos = [eixos]

    x = np.arange(len(metricas))
    largura = 0.36

    for eixo, classe in zip(eixos, classes):
        por_modelo = {
            linha["modelo"]: linha
            for linha in agregados
            if linha["classe"] == classe
            and linha["agregacao"] == agregacao
            and linha["modelo"] in MODELOS
        }

        for indice, modelo in enumerate(MODELOS):
            if modelo not in por_modelo:
                continue

            valores = [float(por_modelo[modelo][m]) for m in metricas]
            deslocamento = (indice - 0.5) * largura
            barras = eixo.bar(
                x + deslocamento,
                valores,
                largura,
                label=modelo.capitalize(),
            )
            eixo.bar_label(barras, fmt="%.3f", padding=2, fontsize=7)

        eixo.set_title(classe.capitalize())
        eixo.set_xticks(x, rotulos, rotation=35, ha="right")
        eixo.set_ylim(0, 1.08)
        eixo.grid(axis="y", alpha=0.2)

    eixos[0].set_ylabel("Media por imagem")
    eixos[-1].legend()
    figura.suptitle(
        "Metricas de segmentacao por classe - modelo original vs aumentado"
    )
    figura.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, dpi=170, bbox_inches="tight")
    plt.close(figura)


def plotar_metricas_todas(resumo: list[dict[str, str]], destino: Path) -> None:
    """
    Compara os modelos usando macro_todas:
    media por imagem considerando todas as classes configuradas.
    """
    linhas = {
        linha["modelo"]: linha
        for linha in resumo
        if linha["escopo"] == "macro_todas" and linha["modelo"] in MODELOS
    }
    ausentes = [m for m in MODELOS if m not in linhas]
    if ausentes:
        raise RuntimeError(
            "Nao encontrei macro_todas para: " + ", ".join(ausentes)
        )

    # Dice e F1 sao iguais nesta implementacao. Mostramos apenas F1/Dice.
    metricas = ("iou", "f1", "precisao", "revocacao", "especificidade", "acuracia")
    rotulos = [ROTULOS_METRICAS[m] for m in metricas]
    x = np.arange(len(metricas))
    largura = 0.36

    figura, eixo = plt.subplots(figsize=(11, 6))
    for indice, modelo in enumerate(MODELOS):
        valores = [float(linhas[modelo][m]) for m in metricas]
        deslocamento = (indice - 0.5) * largura
        barras = eixo.bar(
            x + deslocamento,
            valores,
            largura,
            label=modelo.capitalize(),
        )
        eixo.bar_label(barras, fmt="%.3f", padding=3, fontsize=8)

    eixo.set_xticks(x, rotulos)
    eixo.set_ylim(0, 1.08)
    eixo.set_ylabel("Media por imagem")
    eixo.set_title("Comparacao no conjunto de teste real - todas as imagens")
    eixo.legend()
    eixo.grid(axis="y", alpha=0.2)
    figura.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, dpi=170, bbox_inches="tight")
    plt.close(figura)


def plotar_curvas_conjuntas(
    historico_original: list[dict[str, str]],
    historico_aumentado: list[dict[str, str]],
    destino: Path,
) -> None:
    figura, eixos = plt.subplots(1, 2, figsize=(15, 5.5))

    historicos = {
        "Original": historico_original,
        "Aumentado": historico_aumentado,
    }

    # Painel 1: perdas de treino e validacao de ambos os modelos.
    for nome, historico in historicos.items():
        epocas = [int(float(linha["epoca"])) for linha in historico]
        eixos[0].plot(
            epocas,
            [numero(linha, "treino_loss") for linha in historico],
            label=f"{nome} - treino",
        )
        eixos[0].plot(
            epocas,
            [numero(linha, "validacao_loss") for linha in historico],
            linestyle="--",
            label=f"{nome} - validacao",
        )

    eixos[0].set_title("Perda")
    eixos[0].set_xlabel("Epoca")
    eixos[0].set_ylabel("BCE + Dice")
    eixos[0].grid(alpha=0.2)
    eixos[0].legend(fontsize=8)

    # Painel 2: metricas de validacao. Linhas continuas = IoU; tracejadas = F1/Dice.
    for nome, historico in historicos.items():
        epocas = [int(float(linha["epoca"])) for linha in historico]
        eixos[1].plot(
            epocas,
            [numero(linha, "validacao_iou") for linha in historico],
            label=f"{nome} - IoU",
        )
        eixos[1].plot(
            epocas,
            [numero(linha, "validacao_f1") for linha in historico],
            linestyle="--",
            label=f"{nome} - F1/Dice",
        )

    eixos[1].set_title("Validacao")
    eixos[1].set_xlabel("Epoca")
    eixos[1].set_ylabel("Metrica")
    eixos[1].set_ylim(0, 1)
    eixos[1].grid(alpha=0.2)
    eixos[1].legend(fontsize=8)

    figura.suptitle("Curvas de aprendizado - U-Net original vs aumentado")
    figura.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, dpi=170, bbox_inches="tight")
    plt.close(figura)


def encontrar_arquivo(explicito: str | None, candidatos: list[Path], descricao: str) -> Path:
    if explicito:
        caminho = Path(explicito)
        if not caminho.exists():
            raise FileNotFoundError(f"{descricao} nao encontrado: {caminho}")
        return caminho

    for caminho in candidatos:
        if caminho.exists():
            return caminho

    raise FileNotFoundError(
        f"Nao foi encontrado {descricao}. Tente informar o caminho explicitamente."
    )


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analise complementar dos resultados do pipeline: agrega metricas por classe, "
            "plota macro de todas as imagens e une as curvas de aprendizado dos dois cenarios."
        )
    )
    parser.add_argument(
        "--resultados-dir",
        default=None,
        help=(
            "Pasta de resultados do pipeline. Padrao: <raiz-do-projeto>/resultados"
        ),
    )
    parser.add_argument(
        "--metricas-imagem",
        help="Caminho opcional para metricas_por_imagem.csv.",
    )
    parser.add_argument(
        "--resumo",
        help="Caminho opcional para resumo_metricas.csv.",
    )
    parser.add_argument(
        "--historico-original",
        help="Caminho opcional para historico_original.csv.",
    )
    parser.add_argument(
        "--historico-aumentado",
        help="Caminho opcional para historico_aumentado.csv.",
    )
    parser.add_argument(
        "--saida",
        help="Pasta de saida. Padrao: <resultados-dir>/analise",
    )
    return parser


def main() -> None:
    args = construir_parser().parse_args()

    resultados = (
        Path(args.resultados_dir) if args.resultados_dir else RAIZ_PROJETO / "resultados"
    )
    saida = Path(args.saida) if args.saida else resultados / "analise"
    treinamento = resultados / "treinamento"

    caminho_metricas = encontrar_arquivo(
        args.metricas_imagem,
        [resultados / "metricas_por_imagem.csv"],
        "metricas_por_imagem.csv",
    )
    caminho_resumo = encontrar_arquivo(
        args.resumo,
        [resultados / "resumo_metricas.csv"],
        "resumo_metricas.csv",
    )
    caminho_hist_original = encontrar_arquivo(
        args.historico_original,
        [
            treinamento / "historico_original.csv",
            resultados / "historico_original.csv",
            RAIZ_PROJETO / "historico_original.csv",
            Path("historico_original.csv"),
        ],
        "historico_original.csv",
    )
    caminho_hist_aumentado = encontrar_arquivo(
        args.historico_aumentado,
        [
            treinamento / "historico_aumentado.csv",
            resultados / "historico_aumentado.csv",
            RAIZ_PROJETO / "historico_aumentado.csv",
            Path("historico_aumentado.csv"),
        ],
        "historico_aumentado.csv",
    )

    linhas_metricas = ler_csv(caminho_metricas)
    resumo = ler_csv(caminho_resumo)
    hist_original = ler_csv(caminho_hist_original)
    hist_aumentado = ler_csv(caminho_hist_aumentado)

    saida.mkdir(parents=True, exist_ok=True)

    agregados = agregar_por_classe(
        linhas_metricas,
        saida / "metricas_por_classe.csv",
    )
    plotar_metricas_por_classe(
        agregados,
        saida / "comparacao_metricas_por_classe.png",
        agregacao="macro_imagens",
    )
    plotar_metricas_todas(
        resumo,
        saida / "comparacao_metricas_todas_imagens.png",
    )
    plotar_curvas_conjuntas(
        hist_original,
        hist_aumentado,
        saida / "curvas_aprendizado_comparadas.png",
    )

    print(f"Analise concluida. Saida: {saida.resolve()}")
    print(f"Imagens avaliadas: {len(linhas_metricas)}")
    print("Arquivos gerados:")
    for caminho in sorted(saida.iterdir()):
        print(f"  - {caminho.name}")


if __name__ == "__main__":
    main()
