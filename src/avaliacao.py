from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

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
from .metricas import contagens_binarias, metricas_binarias, metricas_de_contagens, somar_contagens
from .modelo import criar_modelo


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
    prob_original = np.asarray(prob_imagem.resize(imagem.size, Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    return prob_original >= limiar


def _sobrepor(eixo, imagem: np.ndarray, mascara: np.ndarray, titulo: str, cor: str) -> None:
    eixo.imshow(imagem, cmap="gray")
    mapa = np.ma.masked_where(~mascara, mascara)
    eixo.imshow(mapa, cmap="autumn" if cor == "vermelho" else "winter", alpha=0.42)
    if mascara.any():
        eixo.contour(mascara, levels=[0.5], colors=["red" if cor == "vermelho" else "cyan"], linewidths=1)
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
    for eixo, nome in zip(eixos[2:], ("original", "aumentado")):
        m = metricas[nome]
        _sobrepor(
            eixo, imagem, predicoes[nome],
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
        resumo = {metrica: float(np.mean([l[f"{prefixo}{metrica}"] for l in subconjunto])) for metrica in (
            "iou", "f1", "dice", "precisao", "revocacao", "especificidade", "acuracia"
        )}
        saida.append({"modelo": nome, "escopo": escopo, "n": len(subconjunto), **resumo})
    saida.append({"modelo": nome, "escopo": "micro_pixels", "n": len(linhas), **micro})
    return saida


def _plotar_resumo(resumo: list[dict[str, Any]], destino: Path) -> None:
    import matplotlib.pyplot as plt

    linhas = [r for r in resumo if r["escopo"] == "macro_regiao_presente"]
    metricas = ("iou", "f1")
    x = np.arange(len(metricas))
    largura = 0.34
    figura, eixo = plt.subplots(figsize=(7, 4.5))
    for indice, linha in enumerate(linhas):
        valores = [linha[m] for m in metricas]
        barras = eixo.bar(x + (indice - 0.5) * largura, valores, largura, label=linha["modelo"].capitalize())
        eixo.bar_label(barras, fmt="%.3f", padding=3)
    eixo.set_xticks(x, ["IoU", "F1-Score / Dice"])
    eixo.set_ylim(0, 1.08)
    eixo.set_ylabel("Media por imagem (referencia com regiao)")
    eixo.set_title("Comparacao no conjunto de teste real")
    eixo.legend()
    eixo.grid(axis="y", alpha=0.2)
    figura.tight_layout()
    figura.savefig(destino, dpi=160, bbox_inches="tight")
    plt.close(figura)


def avaliar_modelos(config: dict[str, Any], limite: int | None = None) -> dict[str, Any]:
    cfg = config["avaliacao"]
    dispositivo = _dispositivo(cfg.get("dispositivo", config["treinamento"].get("dispositivo", "auto")))
    modelos_dir = caminho_configurado(config, "modelos")
    caminhos = {
        "original": modelos_dir / "unet_original.pt",
        "aumentado": modelos_dir / "unet_aumentado.pt",
    }
    ausentes = [str(c) for c in caminhos.values() if not c.exists()]
    if ausentes:
        raise FileNotFoundError(f"Checkpoints ausentes: {ausentes}")
    carregados = {nome: _carregar_checkpoint(caminho, dispositivo) for nome, caminho in caminhos.items()}
    classes = classes_configuradas(config)
    classes_sem_mascara = classes_sem_mascara_configuradas(config)
    amostras = coletar_amostras(
        caminho_configurado(config, "teste"),
        classes=classes,
        incluir_sinteticas=False,
        classes_sem_mascara=classes_sem_mascara,
    )
    if limite is not None:
        amostras = amostras[:limite]
    if not amostras:
        raise RuntimeError("O conjunto de teste esta vazio.")

    resultados = caminho_configurado(config, "resultados")
    sobreposicoes = resultados / "sobreposicoes"
    resultados.mkdir(parents=True, exist_ok=True)
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
        if cfg.get("salvar_sobreposicoes", True) and (max_plots is None or indice < int(max_plots)):
            _salvar_comparacao(
                amostra, imagem, alvo, predicoes, metricas,
                sobreposicoes / amostra.classe / f"{amostra.imagem.stem}_comparacao.png",
            )

    csv_imagens = resultados / "metricas_por_imagem.csv"
    with csv_imagens.open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)
    resumo = _resumir_modelo("original", linhas) + _resumir_modelo("aumentado", linhas)
    campos_resumo = ["modelo", "escopo", "n", "iou", "f1", "dice", "precisao", "revocacao", "especificidade", "acuracia"]
    with (resultados / "resumo_metricas.csv").open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos_resumo, extrasaction="ignore")
        escritor.writeheader()
        escritor.writerows(resumo)
    _plotar_resumo(resumo, resultados / "comparacao_iou_f1.png")
    return {"imagens_avaliadas": len(linhas), "limiar": limiar, "resumo": resumo}
