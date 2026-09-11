from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .configuracao import caminho_configurado
from .dados import (
    Amostra,
    carregar_mascara_unificada,
    classes_configuradas,
    classes_sem_mascara_configuradas,
    coletar_amostras,
)
from .metricas import metricas_binarias


def _cv2():
    try:
        import cv2
    except ImportError as erro:
        raise RuntimeError(
            "Instale opencv-python-headless para gerar as mascaras watershed."
        ) from erro
    return cv2


def _kernel_eliptico(tamanho: int, cv2):
    tamanho = max(1, int(tamanho))
    if tamanho % 2 == 0:
        tamanho += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamanho, tamanho))


def _normalizar_uint8(imagem: np.ndarray) -> np.ndarray:
    imagem = np.asarray(imagem, dtype=np.float32)
    minimo = float(np.min(imagem)) if imagem.size else 0.0
    maximo = float(np.max(imagem)) if imagem.size else 0.0
    if maximo <= minimo:
        return np.zeros(imagem.shape, dtype=np.uint8)
    normalizada = (imagem - minimo) * (255.0 / (maximo - minimo))
    return np.clip(normalizada, 0, 255).astype(np.uint8)


def _criar_marcadores(
    mascara_origem: np.ndarray,
    cfg: dict[str, Any],
    cv2,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Cria os dois rotulos sugeridos para o marker-controlled watershed.

    Rotulo 1 (interno): afinamento/esqueletizacao da mascara original.
    Rotulo 2 (externo): gradiente morfologico da mascara original apos dilatacao.
    Pixels restantes ficam sem rotulo (0), onde o watershed procura a fronteira.
    """
    mascara = (mascara_origem > 0).astype(np.uint8) * 255
    if not np.any(mascara):
        raise ValueError("A mascara de origem da regiao de interesse esta vazia.")

    kernel_dilatacao = _kernel_eliptico(
        int(cfg.get("dilatacao_externa_kernel", 7)), cv2
    )
    iteracoes = max(1, int(cfg.get("dilatacao_externa_iteracoes", 4)))
    dilatada = cv2.dilate(mascara, kernel_dilatacao, iterations=iteracoes)

    kernel_gradiente = _kernel_eliptico(
        int(cfg.get("gradiente_dilatacao_kernel", 3)), cv2
    )
    gradiente_dilatacao = cv2.morphologyEx(
        dilatada, cv2.MORPH_GRADIENT, kernel_gradiente
    )
    # O marcador externo precisa ficar fora da anotacao original. A dilatacao
    # cria a folga na qual o watershed podera ajustar o contorno na sintetica.
    marcador_externo = (
        (gradiente_dilatacao > 0) & (mascara == 0)
    ).astype(np.uint8) * 255

    try:
        from skimage.morphology import thin
    except ImportError as erro:
        raise RuntimeError(
            "Instale scikit-image para criar o marcador interno por afinamento."
        ) from erro

    marcador_interno = thin(mascara > 0).astype(np.uint8) * 255
    if not np.any(marcador_interno):
        raise RuntimeError("O afinamento produziu um marcador interno vazio.")
    if not np.any(marcador_externo):
        raise RuntimeError(
            "A dilatacao/gradiente produziu um marcador externo vazio; "
            "revise os parametros de watershed."
        )

    marcadores = np.zeros(mascara.shape, dtype=np.int32)
    marcadores[marcador_interno > 0] = 1
    marcadores[marcador_externo > 0] = 2

    return marcadores, {
        "mascara_origem": mascara,
        "mascara_dilatada": dilatada,
        "gradiente_dilatacao": gradiente_dilatacao,
        "marcador_interno": marcador_interno,
        "marcador_externo": marcador_externo,
    }


def _calcular_gradiente_imagem(
    cinza: np.ndarray,
    cfg: dict[str, Any],
    cv2,
) -> tuple[np.ndarray, str]:
    metodo = str(cfg.get("gradiente_imagem", "sobel")).strip().lower()
    if metodo == "sobel":
        tamanho = max(1, int(cfg.get("sobel_kernel", 3)))
        if tamanho % 2 == 0:
            tamanho += 1
        if tamanho not in (1, 3, 5, 7):
            raise ValueError("watershed.sobel_kernel deve ser 1, 3, 5 ou 7.")
        sobel_x = cv2.Sobel(cinza, cv2.CV_32F, 1, 0, ksize=tamanho)
        sobel_y = cv2.Sobel(cinza, cv2.CV_32F, 0, 1, ksize=tamanho)
        return cv2.magnitude(sobel_x, sobel_y), "sobel"

    if metodo in {"morfologico", "morph", "morphological"}:
        kernel = _kernel_eliptico(
            int(cfg.get("gradiente_morfologico_kernel", 3)), cv2
        )
        gradiente = cv2.morphologyEx(cinza, cv2.MORPH_GRADIENT, kernel)
        return gradiente.astype(np.float32), "morfologico"

    raise ValueError(
        "watershed.gradiente_imagem deve ser 'sobel' ou 'morfologico'."
    )


def segmentar_watershed_guiado(
    imagem_sintetica: np.ndarray,
    mascara_origem: np.ndarray,
    cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Segmenta a sintetica usando a mascara original apenas como fonte de marcadores.

    A mascara original nao e copiada como saida. Ela define uma semente interna
    (rotulo 1) e uma semente externa (rotulo 2). A fronteira final e decidida pelo
    watershed sobre o gradiente da imagem sintetica em tons de cinza.
    """
    cv2 = _cv2()

    imagem = np.asarray(imagem_sintetica)
    if imagem.ndim == 3:
        cinza = cv2.cvtColor(imagem.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        cinza = imagem.astype(np.uint8)

    mascara = (np.asarray(mascara_origem) > 0).astype(np.uint8) * 255
    if mascara.shape != cinza.shape:
        mascara = cv2.resize(
            mascara,
            (cinza.shape[1], cinza.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        mascara = (mascara > 0).astype(np.uint8) * 255

    marcadores, passos = _criar_marcadores(mascara, cfg, cv2)
    gradiente, metodo_gradiente = _calcular_gradiente_imagem(cinza, cfg, cv2)

    try:
        from skimage.segmentation import watershed
    except ImportError as erro:
        raise RuntimeError(
            "Instale scikit-image para executar o watershed sobre o gradiente."
        ) from erro

    rotulos = watershed(
        gradiente,
        marcadores,
        watershed_line=bool(cfg.get("linha_watershed", True)),
    ).astype(np.int32)

    # O rotulo interno representa a regiao de interesse. Limitamos o resultado a regiao
    # dilatada, que e justamente a janela de busca criada a partir da mascara
    # original e fechada pelo marcador externo.
    mascara_resultado = (
        (rotulos == 1) & (passos["mascara_dilatada"] > 0)
    ).astype(np.uint8) * 255

    passos = {
        "imagem_cinza": cinza,
        **passos,
        "marcadores": marcadores,
        "gradiente_imagem": gradiente,
        "rotulos_watershed": rotulos,
        "mascara_resultado": mascara_resultado,
    }
    diagnostico = {
        "gradiente_imagem": metodo_gradiente,
        "pixels_mascara_origem": int(np.count_nonzero(mascara)),
        "pixels_marcador_interno": int(np.count_nonzero(passos["marcador_interno"])),
        "pixels_marcador_externo": int(np.count_nonzero(passos["marcador_externo"])),
        "pixels_mascara_resultado": int(np.count_nonzero(mascara_resultado)),
        "area_origem_relativa": float(np.mean(mascara > 0)),
        "area_resultado_relativa": float(np.mean(mascara_resultado > 0)),
    }
    return mascara_resultado, passos, diagnostico


def _imagem_marcadores_rgb(marcadores: np.ndarray) -> np.ndarray:
    visual = np.zeros((*marcadores.shape, 3), dtype=np.uint8)
    # Vermelho = marcador interno (regiao); ciano = marcador externo (fundo).
    visual[marcadores == 1] = (255, 0, 0)
    visual[marcadores == 2] = (0, 220, 255)
    return visual


def _sobrepor_mascara(
    imagem: np.ndarray,
    mascara: np.ndarray,
    cor: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.40,
) -> np.ndarray:
    if imagem.ndim == 2:
        rgb = np.repeat(imagem[..., None], 3, axis=2).astype(np.float32)
    else:
        rgb = imagem[..., :3].astype(np.float32)
    regiao = mascara > 0
    cor_array = np.asarray(cor, dtype=np.float32)
    rgb[regiao] = (1.0 - alpha) * rgb[regiao] + alpha * cor_array
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _salvar_debug(
    imagem_sintetica: np.ndarray,
    passos: dict[str, np.ndarray],
    destino: Path,
) -> None:
    """Salva cada etapa e um painel unico para inspecao visual do watershed."""
    import matplotlib.pyplot as plt

    destino.mkdir(parents=True, exist_ok=True)
    imagem = np.asarray(imagem_sintetica)
    if imagem.ndim == 3:
        imagem_base = imagem[..., :3].astype(np.uint8)
    else:
        imagem_base = imagem.astype(np.uint8)

    mascara_origem = passos["mascara_origem"]
    mascara_resultado = passos["mascara_resultado"]
    marcadores_rgb = _imagem_marcadores_rgb(passos["marcadores"])
    sobreposicao_origem = _sobrepor_mascara(
        imagem_base, mascara_origem, cor=(255, 190, 0), alpha=0.38
    )
    sobreposicao_resultado = _sobrepor_mascara(
        imagem_base, mascara_resultado, cor=(255, 0, 0), alpha=0.42
    )
    sobreposicao_marcadores = imagem_base.copy().astype(np.float32)
    internos = passos["marcador_interno"] > 0
    externos = passos["marcador_externo"] > 0
    sobreposicao_marcadores[internos] = (
        0.35 * sobreposicao_marcadores[internos]
        + 0.65 * np.asarray((255, 0, 0), dtype=np.float32)
    )
    sobreposicao_marcadores[externos] = (
        0.35 * sobreposicao_marcadores[externos]
        + 0.65 * np.asarray((0, 220, 255), dtype=np.float32)
    )
    sobreposicao_marcadores = np.clip(sobreposicao_marcadores, 0, 255).astype(np.uint8)

    arquivos = {
        "01_imagem_sintetica.png": imagem_base,
        "02_mascara_origem.png": mascara_origem,
        "03_mascara_origem_sobre_sintetica.png": sobreposicao_origem,
        "04_mascara_dilatada.png": passos["mascara_dilatada"],
        "05_marcador_interno_afinamento.png": passos["marcador_interno"],
        "06_marcador_externo_gradiente_dilatacao.png": passos["marcador_externo"],
        "07_marcadores_sobre_sintetica.png": sobreposicao_marcadores,
        "08_gradiente_imagem.png": _normalizar_uint8(passos["gradiente_imagem"]),
        "09_mascara_resultante.png": mascara_resultado,
        "10_resultado_sobre_sintetica.png": sobreposicao_resultado,
    }
    for nome, array in arquivos.items():
        Image.fromarray(array).save(destino / nome)

    figura, eixos = plt.subplots(2, 5, figsize=(19, 8))
    itens = [
        (imagem_base, "Imagem sintetica", None),
        (mascara_origem, "Mascara original", "gray"),
        (sobreposicao_origem, "Origem sobre sintetica", None),
        (passos["mascara_dilatada"], "Dilatacao externa", "gray"),
        (passos["marcador_interno"], "Rotulo 1: afinamento", "gray"),
        (passos["marcador_externo"], "Rotulo 2: borda dilatada", "gray"),
        (sobreposicao_marcadores, "Marcadores", None),
        (_normalizar_uint8(passos["gradiente_imagem"]), "Gradiente da sintetica", "gray"),
        (mascara_resultado, "Mascara resultante", "gray"),
        (sobreposicao_resultado, "Resultado sobre sintetica", None),
    ]
    for eixo, (conteudo, titulo, cmap) in zip(eixos.flat, itens):
        eixo.imshow(conteudo, cmap=cmap)
        eixo.set_title(titulo)
        eixo.axis("off")
    figura.tight_layout()
    figura.savefig(destino / "painel_debug.png", dpi=150, bbox_inches="tight")
    plt.close(figura)


def _referencia_original(
    amostra: Amostra,
    originais_referencia: dict[str, Amostra],
) -> Amostra:
    id_origem = amostra.id_origem
    referencia = originais_referencia.get(id_origem)
    if referencia is None:
        raise FileNotFoundError(
            f"Nao foi encontrada a imagem original correspondente a {amostra.imagem.name}: "
            f"esperado {id_origem}."
        )
    return referencia


def gerar_mascaras_watershed(
    config: dict[str, Any],
    limite: int | None = None,
    debug: bool = False,
) -> dict[str, Any]:
    cfg = config["watershed"]
    original = caminho_configurado(config, "original")
    aumentado = caminho_configurado(config, "aumentado")
    resultados = caminho_configurado(config, "resultados") / "watershed"

    classes = classes_configuradas(config)
    classes_sem_mascara = classes_sem_mascara_configuradas(config)
    sinteticas = coletar_amostras(
        aumentado,
        classes=classes,
        incluir_originais=False,
        incluir_sinteticas=True,
        exigir_mascara=False,
        classes_sem_mascara=classes_sem_mascara,
    )
    if limite is not None:
        sinteticas = sinteticas[:limite]

    originais_referencia = {
        a.id: a
        for a in coletar_amostras(
            original,
            classes=classes,
            incluir_originais=True,
            incluir_sinteticas=False,
            exigir_mascara=True,
            classes_sem_mascara=classes_sem_mascara,
        )
    }

    manifesto = resultados / "metricas_watershed.csv"
    manifesto.parent.mkdir(parents=True, exist_ok=True)
    linhas: list[dict[str, Any]] = []
    base_projeto = Path(config.get("_raiz_projeto", config["_diretorio_base"]))

    for amostra in sinteticas:
        referencia = _referencia_original(amostra, originais_referencia)
        with Image.open(amostra.imagem) as arquivo:
            imagem_rgb = np.asarray(arquivo.convert("RGB"))
        altura, largura = imagem_rgb.shape[:2]
        mascara_origem = np.asarray(
            carregar_mascara_unificada(referencia, (largura, altura))
        )

        if not np.any(mascara_origem):
            mascara_resultado = np.zeros((altura, largura), dtype=np.uint8)
            passos = {
                "imagem_cinza": np.asarray(Image.fromarray(imagem_rgb).convert("L")),
                "mascara_origem": (mascara_origem > 0).astype(np.uint8) * 255,
                "mascara_dilatada": np.zeros((altura, largura), dtype=np.uint8),
                "gradiente_dilatacao": np.zeros((altura, largura), dtype=np.uint8),
                "marcador_interno": np.zeros((altura, largura), dtype=np.uint8),
                "marcador_externo": np.zeros((altura, largura), dtype=np.uint8),
                "marcadores": np.zeros((altura, largura), dtype=np.int32),
                "gradiente_imagem": np.zeros((altura, largura), dtype=np.float32),
                "rotulos_watershed": np.zeros((altura, largura), dtype=np.int32),
                "mascara_resultado": mascara_resultado,
            }
            diagnostico = {
                "gradiente_imagem": "nao_aplicado_sem_regiao",
                "pixels_mascara_origem": int(np.count_nonzero(mascara_origem)),
                "pixels_marcador_interno": 0,
                "pixels_marcador_externo": 0,
                "pixels_mascara_resultado": 0,
                "area_origem_relativa": float(np.mean(mascara_origem > 0)),
                "area_resultado_relativa": 0.0,
            }
        else:
            mascara_resultado, passos, diagnostico = segmentar_watershed_guiado(
                imagem_rgb,
                mascara_origem,
                cfg,
            )

        caminho_mascara = amostra.imagem.with_name(
            f"{amostra.imagem.stem}_mask.png"
        )
        Image.fromarray(mascara_resultado, mode="L").save(caminho_mascara)

        comparacao = metricas_binarias(
            mascara_resultado > 0,
            mascara_origem > 0,
        )
        linhas.append(
            {
                "classe": amostra.classe,
                "imagem_sintetica": str(amostra.imagem.relative_to(base_projeto)),
                "imagem_origem": str(referencia.imagem.relative_to(base_projeto)),
                "mascara_origem": "|".join(
                    str(p.relative_to(base_projeto)) for p in referencia.mascaras
                ),
                "mascara_watershed": str(caminho_mascara.relative_to(base_projeto)),
                **diagnostico,
                # Estes valores medem quanto o watershed deslocou a anotacao de
                # origem; nao sao metricas independentes de qualidade, pois a
                # propria mascara original fornece os marcadores.
                "iou_com_origem": comparacao["iou"],
                "f1_com_origem": comparacao["f1"],
            }
        )

        if debug:
            _salvar_debug(
                imagem_rgb,
                passos,
                resultados / "debug" / amostra.classe / amostra.imagem.stem,
            )

    campos = [
        "classe",
        "imagem_sintetica",
        "imagem_origem",
        "mascara_origem",
        "mascara_watershed",
        "gradiente_imagem",
        "pixels_mascara_origem",
        "pixels_marcador_interno",
        "pixels_marcador_externo",
        "pixels_mascara_resultado",
        "area_origem_relativa",
        "area_resultado_relativa",
        "iou_com_origem",
        "f1_com_origem",
    ]
    with manifesto.open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(linhas)

    return {
        "sinteticas_processadas": len(linhas),
        "mascaras_criadas": len(linhas),
        "debug_ativo": debug,
        "pasta_debug": str((resultados / "debug").relative_to(base_projeto)) if debug else None,
        "manifesto": str(manifesto.relative_to(base_projeto)),
    }
