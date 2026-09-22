from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image

from .configuracao import caminho_configurado, caminho_portatil
from .dados import (
    Amostra,
    carregar_mascara_unificada,
    classes_configuradas,
    classes_sem_mascara_configuradas,
    coletar_amostras,
)
from .metricas import metricas_binarias, metricas_de_contagens, somar_contagens
from .modelo import criar_modelo


METRICAS = (
    "iou",
    "f1",
    "dice",
    "precisao",
    "revocacao",
    "especificidade",
    "acuracia",
)
MODELOS = ("original", "aumentado")
ROTULOS_METRICAS = {
    "iou": "IoU",
    "f1": "F1/Dice",
    "dice": "Dice",
    "precisao": "Precisao",
    "revocacao": "Revocacao",
    "especificidade": "Especificidade",
    "acuracia": "Acuracia",
}


def _torch():
    try:
        import torch
    except ImportError as erro:
        raise RuntimeError("Instale PyTorch antes de executar a avaliacao.") from erro
    return torch


def _dispositivo(valor: str):
    torch = _torch()
    if valor == "auto":
        valor = "cuda" if torch.cuda.is_available() else "cpu"
    if valor == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA foi solicitada, mas nao esta disponivel.")
    return torch.device(valor)


def _carregar_checkpoint(caminho: Path, dispositivo):
    torch = _torch()
    checkpoint = torch.load(caminho, map_location=dispositivo)
    modelo = criar_modelo(checkpoint["modelo_segmentacao"], carregar_pesos_encoder=False)
    modelo.load_state_dict(checkpoint["estado_modelo"])
    modelo.to(dispositivo).eval()
    return modelo, checkpoint


def _preparar_tensor(imagem: Image.Image, cfg: dict[str, Any], dispositivo):
    torch = _torch()
    altura, largura = map(int, cfg["tamanho_imagem"])
    redimensionada = imagem.convert("RGB").resize((largura, altura), Image.Resampling.BILINEAR)
    array = np.asarray(redimensionada, dtype=np.float32) / 255.0
    media = np.asarray(cfg["normalizacao_media"], dtype=np.float32).reshape(1, 1, 3)
    desvio = np.asarray(cfg["normalizacao_desvio"], dtype=np.float32).reshape(1, 1, 3)
    array = ((array - media) / desvio).transpose(2, 0, 1)[None]
    return torch.from_numpy(array).float().to(dispositivo)


def _predizer(modelo, imagem: Image.Image, cfg: dict[str, Any], dispositivo, limiar: float) -> np.ndarray:
    torch = _torch()
    tensor = _preparar_tensor(imagem, cfg, dispositivo)
    with torch.inference_mode():
        prob = torch.sigmoid(modelo(tensor))[0, 0].cpu().numpy()
    prob_imagem = Image.fromarray((prob * 255).astype(np.uint8), mode="L")
    prob_original = np.asarray(
        prob_imagem.resize(imagem.size, Image.Resampling.BILINEAR), dtype=np.float32
    ) / 255.0
    return prob_original >= limiar


def _sobrepor(eixo, imagem: np.ndarray, mascara: np.ndarray, titulo: str, cor: str) -> None:
    eixo.imshow(imagem, cmap="gray")
    mapa = np.ma.masked_where(~mascara, mascara)
    eixo.imshow(mapa, cmap="autumn" if cor == "vermelho" else "winter", alpha=0.42)
    if mascara.any():
        eixo.contour(
            mascara,
            levels=[0.5],
            colors=["red" if cor == "vermelho" else "cyan"],
            linewidths=1,
        )
    eixo.set_title(titulo, fontsize=9)
    eixo.axis("off")


def _salvar_comparacao(
    amostra: Amostra,
    imagem: np.ndarray,
    alvo: np.ndarray,
    predicoes: dict[str, np.ndarray],
    metricas: dict[str, dict[str, float]],
    destino: Path,
) -> None:
    import matplotlib.pyplot as plt

    destino.parent.mkdir(parents=True, exist_ok=True)
    figura, eixos = plt.subplots(1, 4, figsize=(15, 4))
    eixos[0].imshow(imagem, cmap="gray")
    eixos[0].set_title(f"Imagem real\n{amostra.classe}")
    eixos[0].axis("off")
    _sobrepor(eixos[1], imagem, alvo, "Referencia", "vermelho")
    for eixo, nome in zip(eixos[2:], MODELOS):
        m = metricas[nome]
        _sobrepor(
            eixo,
            imagem,
            predicoes[nome],
            f"Modelo {nome}\nIoU {m['iou']:.3f} | F1 {m['f1']:.3f}",
            "azul",
        )
    figura.tight_layout()
    figura.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(figura)


def _resumir_modelo(nome: str, linhas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    prefixo = f"{nome}_"
    todas_contagens = [
        {chave: int(linha[f"{prefixo}{chave}"]) for chave in ("tp", "tn", "fp", "fn")}
        for linha in linhas
    ]
    micro = metricas_de_contagens(somar_contagens(todas_contagens))
    com_regiao = [linha for linha in linhas if bool(linha.get("tem_regiao", False))]
    saida: list[dict[str, Any]] = []
    for escopo, subconjunto in (("macro_todas", linhas), ("macro_regiao_presente", com_regiao)):
        if not subconjunto:
            continue
        resumo = {
            metrica: float(np.mean([float(l[f"{prefixo}{metrica}"]) for l in subconjunto]))
            for metrica in METRICAS
        }
        saida.append({"modelo": nome, "escopo": escopo, "n": len(subconjunto), **resumo})
    saida.append({"modelo": nome, "escopo": "micro_pixels", "n": len(linhas), **micro})
    return saida


def _plotar_resumo(resumo: list[dict[str, Any]], destino: Path, titulo_contexto: str) -> None:
    import matplotlib.pyplot as plt

    linhas = [r for r in resumo if r["escopo"] == "macro_regiao_presente"]
    if not linhas:
        linhas = [r for r in resumo if r["escopo"] == "macro_todas"]
    metricas = ("iou", "f1")
    x = np.arange(len(metricas))
    largura = 0.34
    figura, eixo = plt.subplots(figsize=(7, 4.5))
    for indice, linha in enumerate(linhas):
        valores = [linha[m] for m in metricas]
        barras = eixo.bar(
            x + (indice - 0.5) * largura,
            valores,
            largura,
            label=linha["modelo"].capitalize(),
        )
        eixo.bar_label(barras, fmt="%.3f", padding=3)
    eixo.set_xticks(x, ["IoU", "F1-Score / Dice"])
    eixo.set_ylim(0, 1.08)
    eixo.set_ylabel("Media por imagem")
    eixo.set_title(f"Comparacao - {titulo_contexto}")
    eixo.legend()
    eixo.grid(axis="y", alpha=0.2)
    figura.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, dpi=160, bbox_inches="tight")
    plt.close(figura)


def _salvar_csv(linhas: Sequence[dict[str, Any]], destino: Path, campos: Sequence[str] | None = None) -> None:
    if not linhas:
        return
    destino.parent.mkdir(parents=True, exist_ok=True)
    campos = list(campos) if campos is not None else list(linhas[0])
    with destino.open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos, extrasaction="ignore")
        escritor.writeheader()
        escritor.writerows(linhas)


def avaliar_checkpoints_em_amostras(
    config: dict[str, Any],
    amostras: Sequence[Amostra],
    caminhos: Mapping[str, Path],
    resultados: Path,
    *,
    limite: int | None = None,
    salvar_sobreposicoes: bool = False,
    gerar_analise: bool = False,
    titulo_contexto: str = "conjunto de teste externo",
) -> dict[str, Any]:
    """Avalia um par de checkpoints em uma lista explicita de imagens reais."""
    cfg = config["avaliacao"]
    dispositivo = _dispositivo(cfg.get("dispositivo", config["treinamento"].get("dispositivo", "auto")))
    ausentes = [str(c) for c in caminhos.values() if not c.exists()]
    if ausentes:
        raise FileNotFoundError(f"Checkpoints ausentes: {ausentes}")
    if set(caminhos) != set(MODELOS):
        raise ValueError(f"Esperados checkpoints para {MODELOS}; recebidos: {tuple(caminhos)}")
    carregados = {nome: _carregar_checkpoint(Path(caminho), dispositivo) for nome, caminho in caminhos.items()}
    amostras = list(amostras)
    if limite is not None:
        amostras = amostras[:limite]
    if not amostras:
        raise RuntimeError("O conjunto informado para avaliacao esta vazio.")

    resultados.mkdir(parents=True, exist_ok=True)
    # A avaliacao final mantem todos os artefatos numericos/graficos em uma pasta
    # propria, deixando a raiz de resultados reservada para as demais etapas do pipeline.
    if gerar_analise:
        pasta_metricas = resultados / "analise_modelo_final"
        sobreposicoes = resultados / "sobreposicoes_modelo_final"
    else:
        # Avaliacoes internas (por exemplo, a validacao de cada fold) preservam
        # sua estrutura local para nao alterar os artefatos da validacao cruzada.
        pasta_metricas = resultados
        sobreposicoes = resultados / "sobreposicoes"
    pasta_metricas.mkdir(parents=True, exist_ok=True)
    limiar = float(cfg.get("limiar", config["treinamento"].get("limiar", 0.5)))
    max_plots = cfg.get("max_sobreposicoes")
    linhas: list[dict[str, Any]] = []

    for indice, amostra in enumerate(amostras):
        with Image.open(amostra.imagem) as arquivo:
            imagem_pil = arquivo.convert("L")
        imagem = np.asarray(imagem_pil)
        alvo = np.asarray(carregar_mascara_unificada(amostra, imagem_pil.size)) > 0
        predicoes: dict[str, np.ndarray] = {}
        metricas: dict[str, dict[str, float]] = {}
        linha: dict[str, Any] = {
            "id": amostra.id,
            "classe": amostra.classe,
            "tem_regiao": bool(alvo.any()),
            "imagem": caminho_portatil(config, amostra.imagem),
        }
        for nome, (modelo, checkpoint) in carregados.items():
            predicao = _predizer(modelo, imagem_pil, checkpoint["treinamento"], dispositivo, limiar)
            predicoes[nome] = predicao
            metricas[nome] = metricas_binarias(predicao, alvo)
            linha.update({f"{nome}_{chave}": valor for chave, valor in metricas[nome].items()})
        linhas.append(linha)
        if salvar_sobreposicoes and (max_plots is None or indice < int(max_plots)):
            _salvar_comparacao(
                amostra,
                imagem,
                alvo,
                predicoes,
                metricas,
                sobreposicoes / amostra.classe / f"{amostra.imagem.stem}_comparacao.png",
            )

    _salvar_csv(linhas, pasta_metricas / "metricas_por_imagem.csv")
    resumo = _resumir_modelo("original", linhas) + _resumir_modelo("aumentado", linhas)
    campos_resumo = ["modelo", "escopo", "n", *METRICAS]
    _salvar_csv(resumo, pasta_metricas / "resumo_metricas.csv", campos_resumo)

    analise = None
    if gerar_analise:
        analise = analisar_resultados_avaliacao(config, linhas, resumo, pasta_metricas)
    else:
        # Mantido apenas para avaliacoes internas dos folds. Na avaliacao final,
        # a comparacao e mais completa e contempla todas as metricas.
        _plotar_resumo(resumo, pasta_metricas / "comparacao_iou_f1.png", titulo_contexto)

    return {
        "imagens_avaliadas": len(linhas),
        "limiar": limiar,
        "resumo": resumo,
        "analise": analise,
    }


def _agregar_por_classe(linhas: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    classes = sorted({str(linha["classe"]) for linha in linhas})
    saida: list[dict[str, Any]] = []
    for classe in classes:
        subconjunto = [linha for linha in linhas if linha["classe"] == classe]
        for modelo in MODELOS:
            prefixo = f"{modelo}_"
            macro = {
                metrica: float(np.mean([float(linha[prefixo + metrica]) for linha in subconjunto]))
                for metrica in METRICAS
            }
            tp = sum(int(linha[prefixo + "tp"]) for linha in subconjunto)
            tn = sum(int(linha[prefixo + "tn"]) for linha in subconjunto)
            fp = sum(int(linha[prefixo + "fp"]) for linha in subconjunto)
            fn = sum(int(linha[prefixo + "fn"]) for linha in subconjunto)
            saida.append(
                {
                    "classe": classe,
                    "modelo": modelo,
                    "agregacao": "macro_imagens",
                    "n": len(subconjunto),
                    "tp": tp,
                    "tn": tn,
                    "fp": fp,
                    "fn": fn,
                    **macro,
                }
            )
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
                    **metricas_de_contagens({"tp": tp, "tn": tn, "fp": fp, "fn": fn}),
                }
            )
    return saida


def _plotar_metricas_por_classe(agregados: Sequence[dict[str, Any]], destino: Path) -> None:
    import matplotlib.pyplot as plt

    classes = sorted({str(linha["classe"]) for linha in agregados})
    if not classes:
        return
    metricas = ("iou", "f1", "precisao", "revocacao", "especificidade", "acuracia")
    rotulos = [ROTULOS_METRICAS[m] for m in metricas]
    figura, eixos = plt.subplots(1, len(classes), figsize=(6 * len(classes), 5.5), sharey=True)
    if len(classes) == 1:
        eixos = [eixos]
    x = np.arange(len(metricas))
    largura = 0.36
    for eixo, classe in zip(eixos, classes):
        por_modelo = {
            linha["modelo"]: linha
            for linha in agregados
            if linha["classe"] == classe and linha["agregacao"] == "macro_imagens"
        }
        for indice, modelo in enumerate(MODELOS):
            if modelo not in por_modelo:
                continue
            valores = [float(por_modelo[modelo][m]) for m in metricas]
            barras = eixo.bar(x + (indice - 0.5) * largura, valores, largura, label=modelo.capitalize())
            eixo.bar_label(barras, fmt="%.3f", padding=2, fontsize=7)
        eixo.set_title(classe.capitalize())
        eixo.set_xticks(x, rotulos, rotation=35, ha="right")
        eixo.set_ylim(0, 1.08)
        eixo.grid(axis="y", alpha=0.2)
    eixos[0].set_ylabel("Media por imagem")
    eixos[-1].legend()
    figura.suptitle("Metricas de segmentacao por classe - teste externo")
    figura.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, dpi=170, bbox_inches="tight")
    plt.close(figura)


def _plotar_metricas_resumo(
    resumo: Sequence[dict[str, Any]],
    destino: Path,
    *,
    escopo: str,
    titulo: str,
) -> None:
    """Plota todas as metricas macro de um escopo da avaliacao final."""
    import matplotlib.pyplot as plt

    linhas = {
        linha["modelo"]: linha
        for linha in resumo
        if linha["escopo"] == escopo and linha["modelo"] in MODELOS
    }
    if any(m not in linhas for m in MODELOS):
        return
    metricas = ("iou", "f1", "precisao", "revocacao", "especificidade", "acuracia")
    rotulos = [ROTULOS_METRICAS[m] for m in metricas]
    x = np.arange(len(metricas))
    largura = 0.36
    figura, eixo = plt.subplots(figsize=(11, 6))
    for indice, modelo in enumerate(MODELOS):
        valores = [float(linhas[modelo][m]) for m in metricas]
        barras = eixo.bar(x + (indice - 0.5) * largura, valores, largura, label=modelo.capitalize())
        eixo.bar_label(barras, fmt="%.3f", padding=3, fontsize=8)
    eixo.set_xticks(x, rotulos)
    eixo.set_ylim(0, 1.08)
    eixo.set_ylabel("Media por imagem")
    eixo.set_title(titulo)
    eixo.legend()
    eixo.grid(axis="y", alpha=0.2)
    figura.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, dpi=170, bbox_inches="tight")
    plt.close(figura)


def _ler_csv(caminho: Path) -> list[dict[str, str]]:
    with caminho.open("r", encoding="utf-8-sig", newline="") as arquivo:
        return list(csv.DictReader(arquivo))


def _plotar_curvas_comparadas(config: dict[str, Any], destino: Path) -> str | None:
    import matplotlib.pyplot as plt

    resultados = caminho_configurado(config, "resultados")
    candidatos = [resultados / "treinamento_final", resultados / "treinamento"]
    pasta = next(
        (
            p
            for p in candidatos
            if (p / "historico_original.csv").exists() and (p / "historico_aumentado.csv").exists()
        ),
        None,
    )
    if pasta is None:
        return None
    historicos = {
        "Original": _ler_csv(pasta / "historico_original.csv"),
        "Aumentado": _ler_csv(pasta / "historico_aumentado.csv"),
    }
    tem_validacao = all("validacao_loss" in h[0] for h in historicos.values() if h)
    figura, eixos = plt.subplots(1, 2, figsize=(15, 5.5))
    for nome, historico in historicos.items():
        epocas = [int(float(linha["epoca"])) for linha in historico]
        eixos[0].plot(epocas, [float(linha["treino_loss"]) for linha in historico], label=f"{nome} - treino")
        if tem_validacao:
            eixos[0].plot(
                epocas,
                [float(linha["validacao_loss"]) for linha in historico],
                linestyle="--",
                label=f"{nome} - validacao",
            )
    eixos[0].set_title("Perda")
    eixos[0].set_xlabel("Epoca")
    eixos[0].set_ylabel("BCE + Dice")
    eixos[0].grid(alpha=0.2)
    eixos[0].legend(fontsize=8)

    for nome, historico in historicos.items():
        epocas = [int(float(linha["epoca"])) for linha in historico]
        prefixo = "validacao" if tem_validacao else "treino"
        eixos[1].plot(epocas, [float(linha[f"{prefixo}_iou"]) for linha in historico], label=f"{nome} - IoU")
        eixos[1].plot(
            epocas,
            [float(linha[f"{prefixo}_f1"]) for linha in historico],
            linestyle="--",
            label=f"{nome} - F1/Dice",
        )
    eixos[1].set_title("Validacao" if tem_validacao else "Treinamento final")
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
    return str(destino)


def analisar_resultados_avaliacao(
    config: dict[str, Any],
    linhas: Sequence[dict[str, Any]],
    resumo: Sequence[dict[str, Any]],
    saida: Path | None = None,
) -> dict[str, Any]:
    """Gera as agregacoes e graficos da avaliacao final em uma unica pasta."""
    saida = saida or (caminho_configurado(config, "resultados") / "analise_modelo_final")
    saida.mkdir(parents=True, exist_ok=True)
    agregados = _agregar_por_classe(linhas)
    campos_classe = [
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
    _salvar_csv(agregados, saida / "metricas_por_classe.csv", campos_classe)
    _plotar_metricas_por_classe(agregados, saida / "comparacao_metricas_por_classe.png")
    _plotar_metricas_resumo(
        resumo,
        saida / "comparacao_metricas_todas_imagens.png",
        escopo="macro_todas",
        titulo="Comparacao no conjunto de teste externo - todas as imagens",
    )
    _plotar_metricas_resumo(
        resumo,
        saida / "comparacao_metricas_imagens_com_regiao.png",
        escopo="macro_regiao_presente",
        titulo="Comparacao no conjunto de teste externo - imagens com regiao de interesse",
    )
    curva = _plotar_curvas_comparadas(config, saida / "curvas_aprendizado_comparadas.png")
    arquivos = sorted(p.name for p in saida.iterdir() if p.is_file())
    return {
        "pasta": str(saida),
        "arquivos": arquivos,
        "curvas_aprendizado": curva,
    }


def avaliar_modelos(config: dict[str, Any], limite: int | None = None) -> dict[str, Any]:
    """Avalia os modelos finais uma unica vez no conjunto de teste externo fixo."""
    modelos_dir = caminho_configurado(config, "modelos")
    caminhos = {
        "original": modelos_dir / "unet_original.pt",
        "aumentado": modelos_dir / "unet_aumentado.pt",
    }
    classes = classes_configuradas(config)
    classes_sem_mascara = classes_sem_mascara_configuradas(config)
    amostras = coletar_amostras(
        caminho_configurado(config, "teste"),
        classes=classes,
        incluir_sinteticas=False,
        classes_sem_mascara=classes_sem_mascara,
    )
    return avaliar_checkpoints_em_amostras(
        config,
        amostras,
        caminhos,
        caminho_configurado(config, "resultados"),
        limite=limite,
        salvar_sobreposicoes=bool(config["avaliacao"].get("salvar_sobreposicoes", True)),
        gerar_analise=True,
        titulo_contexto="conjunto de teste externo",
    )
