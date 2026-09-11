from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image

from .configuracao import caminho_configurado, caminho_portatil, config_publica
from .dados import (
    Amostra,
    carregar_mascara_unificada,
    classes_configuradas,
    classes_sem_mascara_configuradas,
    coletar_amostras,
    contar_por_classe,
    dividir_originais,
)
from .modelo import criar_modelo


def _dependencias_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as erro:
        raise RuntimeError("Instale PyTorch antes de executar o treinamento.") from erro
    return torch, nn, DataLoader, Dataset


def fixar_semente(semente: int, deterministico: bool = True) -> None:
    torch, _, _, _ = _dependencias_torch()
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semente)
    if deterministico:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def obter_dispositivo(valor: str):
    torch, _, _, _ = _dependencias_torch()
    if valor == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if valor == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA foi solicitada, mas nao esta disponivel.")
    return torch.device(valor)


class DatasetSegmentacao:
    """Dataset no nivel do modulo para funcionar com workers no Windows."""

    def __init__(self, amostras: Sequence[Amostra], cfg: dict[str, Any], treino: bool = False):
        self.amostras = list(amostras)
        self.altura, self.largura = map(int, cfg["tamanho_imagem"])
        self.media = np.asarray(cfg["normalizacao_media"], dtype=np.float32).reshape(1, 1, 3)
        self.desvio = np.asarray(cfg["normalizacao_desvio"], dtype=np.float32).reshape(1, 1, 3)
        self.treino = treino
        self.flip = bool(cfg.get("flip_horizontal", False))

    def __len__(self) -> int:
        return len(self.amostras)

    def __getitem__(self, indice: int):
        torch, _, _, _ = _dependencias_torch()
        amostra = self.amostras[indice]
        with Image.open(amostra.imagem) as arquivo:
            imagem = arquivo.convert("RGB").resize((self.largura, self.altura), Image.Resampling.BILINEAR)
        mascara = carregar_mascara_unificada(amostra).resize(
            (self.largura, self.altura), Image.Resampling.NEAREST
        )
        imagem_np = np.asarray(imagem, dtype=np.float32) / 255.0
        mascara_np = (np.asarray(mascara) > 0).astype(np.float32)
        if self.treino and self.flip and random.random() < 0.5:
            imagem_np = np.flip(imagem_np, axis=1).copy()
            mascara_np = np.flip(mascara_np, axis=1).copy()
        imagem_np = (imagem_np - self.media) / self.desvio
        imagem_tensor = torch.from_numpy(imagem_np.transpose(2, 0, 1)).float()
        mascara_tensor = torch.from_numpy(mascara_np[None]).float()
        return imagem_tensor, mascara_tensor, amostra.id


class PerdaBCEDice:
    """Wrapper simples para BCE com logits + Dice; criado sem importar torch no CLI."""

    def __init__(self, peso_bce: float, peso_dice: float, suavizacao: float = 1.0):
        torch, nn, _, _ = _dependencias_torch()
        self.torch = torch
        self.bce = nn.BCEWithLogitsLoss()
        self.peso_bce = peso_bce
        self.peso_dice = peso_dice
        self.suavizacao = suavizacao

    def __call__(self, logits, alvos):
        probabilidades = self.torch.sigmoid(logits)
        dimensoes = (1, 2, 3)
        intersecao = (probabilidades * alvos).sum(dimensoes)
        denominador = probabilidades.sum(dimensoes) + alvos.sum(dimensoes)
        dice = (2 * intersecao + self.suavizacao) / (denominador + self.suavizacao)
        return self.peso_bce * self.bce(logits, alvos) + self.peso_dice * (1 - dice.mean())


def _contagens_tensor(logits, alvos, limiar: float) -> dict[str, int]:
    torch, _, _, _ = _dependencias_torch()
    pred = torch.sigmoid(logits) >= limiar
    alvo = alvos >= 0.5
    return {
        "tp": int((pred & alvo).sum().item()),
        "tn": int((~pred & ~alvo).sum().item()),
        "fp": int((pred & ~alvo).sum().item()),
        "fn": int((~pred & alvo).sum().item()),
    }


def _metricas_contagens(c: dict[str, int]) -> dict[str, float]:
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    iou = tp / (tp + fp + fn) if tp + fp + fn else 1.0
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0
    acuracia = (tp + tn) / max(tp + tn + fp + fn, 1)
    return {"iou": iou, "f1": f1, "dice": f1, "acuracia": acuracia}


def _epoca(modelo, carregador, perda_fn, dispositivo, limiar: float, otimizador=None, scaler=None):
    torch, _, _, _ = _dependencias_torch()
    treinamento = otimizador is not None
    modelo.train(treinamento)
    perda_total = 0.0
    contagens = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
    for imagens, mascaras, _ in carregador:
        imagens, mascaras = imagens.to(dispositivo), mascaras.to(dispositivo)
        if treinamento:
            otimizador.zero_grad(set_to_none=True)
        contexto = torch.autocast(device_type=dispositivo.type, enabled=scaler is not None)
        with torch.set_grad_enabled(treinamento), contexto:
            logits = modelo(imagens)
            perda = perda_fn(logits, mascaras)
        if treinamento:
            if scaler is not None:
                scaler.scale(perda).backward()
                scaler.step(otimizador)
                scaler.update()
            else:
                perda.backward()
                otimizador.step()
        perda_total += float(perda.item()) * imagens.shape[0]
        atuais = _contagens_tensor(logits.detach(), mascaras, limiar)
        for chave in contagens:
            contagens[chave] += atuais[chave]
    return {"loss": perda_total / max(len(carregador.dataset), 1), **_metricas_contagens(contagens)}


def _salvar_historico(historico: list[dict[str, Any]], destino: Path) -> None:
    with destino.open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(historico[0]))
        escritor.writeheader()
        escritor.writerows(historico)


def _plotar_historico(historico: list[dict[str, Any]], destino: Path, titulo: str) -> None:
    import matplotlib.pyplot as plt

    epocas = [linha["epoca"] for linha in historico]
    figura, eixos = plt.subplots(1, 2, figsize=(11, 4))
    eixos[0].plot(epocas, [x["treino_loss"] for x in historico], label="Treino")
    eixos[0].plot(epocas, [x["validacao_loss"] for x in historico], label="Validacao")
    eixos[0].set(title="Perda", xlabel="Epoca", ylabel="BCE + Dice")
    eixos[0].legend()
    eixos[1].plot(epocas, [x["validacao_iou"] for x in historico], label="IoU")
    eixos[1].plot(epocas, [x["validacao_f1"] for x in historico], label="F1/Dice")
    eixos[1].set(title="Validacao", xlabel="Epoca", ylabel="Metrica", ylim=(0, 1))
    eixos[1].legend()
    figura.suptitle(titulo)
    figura.tight_layout()
    figura.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(figura)


def treinar_modelo(
    nome: str,
    treino: Sequence[Amostra],
    validacao: Sequence[Amostra],
    config: dict[str, Any],
) -> dict[str, Any]:
    torch, _, DataLoader, _ = _dependencias_torch()
    cfg = config["treinamento"]
    semente = int(config["experimento"]["semente"])
    fixar_semente(semente, bool(config["experimento"].get("deterministico", True)))
    dispositivo = obter_dispositivo(cfg.get("dispositivo", "auto"))
    dataset_treino = DatasetSegmentacao(treino, cfg, treino=True)
    dataset_validacao = DatasetSegmentacao(validacao, cfg, treino=False)
    gerador = torch.Generator().manual_seed(semente)
    carregador_treino = DataLoader(
        dataset_treino, batch_size=int(cfg["batch_size"]), shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)), generator=gerador,
        pin_memory=dispositivo.type == "cuda",
    )
    carregador_validacao = DataLoader(
        dataset_validacao, batch_size=int(cfg["batch_size"]), shuffle=False,
        num_workers=int(cfg.get("num_workers", 0)), pin_memory=dispositivo.type == "cuda",
    )
    modelo = criar_modelo(config["modelo_segmentacao"]).to(dispositivo)
    otimizador = torch.optim.AdamW(
        modelo.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
    )
    perda_fn = PerdaBCEDice(float(cfg["peso_bce"]), float(cfg["peso_dice"]))
    amp = bool(cfg.get("amp", True)) and dispositivo.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=True) if amp else None
    limiar = float(cfg.get("limiar", 0.5))
    melhor_iou = -1.0
    epocas_sem_melhora = 0
    historico: list[dict[str, Any]] = []
    pasta_modelos = caminho_configurado(config, "modelos")
    pasta_resultados = caminho_configurado(config, "resultados") / "treinamento"
    pasta_modelos.mkdir(parents=True, exist_ok=True)
    pasta_resultados.mkdir(parents=True, exist_ok=True)
    checkpoint = pasta_modelos / f"unet_{nome}.pt"

    for epoca in range(1, int(cfg["epocas"]) + 1):
        treino_m = _epoca(modelo, carregador_treino, perda_fn, dispositivo, limiar, otimizador, scaler)
        validacao_m = _epoca(modelo, carregador_validacao, perda_fn, dispositivo, limiar)
        linha = {"epoca": epoca}
        linha.update({f"treino_{k}": v for k, v in treino_m.items()})
        linha.update({f"validacao_{k}": v for k, v in validacao_m.items()})
        historico.append(linha)
        if validacao_m["iou"] > melhor_iou + float(cfg.get("min_delta", 0.0)):
            melhor_iou = validacao_m["iou"]
            epocas_sem_melhora = 0
            torch.save({
                "estado_modelo": modelo.state_dict(),
                "nome_experimento": nome,
                "epoca": epoca,
                "metricas_validacao": validacao_m,
                "modelo_segmentacao": config["modelo_segmentacao"],
                "treinamento": cfg,
                "configuracao": config_publica(config),
            }, checkpoint)
        else:
            epocas_sem_melhora += 1
        print(
            f"[{nome}] epoca {epoca:03d} | loss val={validacao_m['loss']:.4f} "
            f"| IoU={validacao_m['iou']:.4f} | F1={validacao_m['f1']:.4f}"
        )
        if epocas_sem_melhora >= int(cfg.get("paciencia", 10)):
            break

    _salvar_historico(historico, pasta_resultados / f"historico_{nome}.csv")
    _plotar_historico(historico, pasta_resultados / f"curvas_{nome}.png", f"U-Net - {nome}")
    return {
        "nome": nome,
        "checkpoint": str(checkpoint),
        "melhor_iou_validacao": melhor_iou,
        "epocas_executadas": len(historico),
        "treino": len(treino),
        "validacao": len(validacao),
        "contagem_treino": contar_por_classe(treino, classes_configuradas(config)),
    }


def construir_cenarios(config: dict[str, Any]) -> tuple[list[Amostra], list[Amostra], list[Amostra]]:
    classes = classes_configuradas(config)
    classes_sem_mascara = classes_sem_mascara_configuradas(config)
    originais = coletar_amostras(
        caminho_configurado(config, "original"),
        classes=classes,
        incluir_sinteticas=False,
        classes_sem_mascara=classes_sem_mascara,
    )
    treino_original, validacao = dividir_originais(
        originais,
        float(config["divisao"]["proporcao_treino"]),
        int(config["experimento"]["semente"]),
        classes=classes,
    )
    ids_treino = {a.id for a in treino_original}
    sinteticas = coletar_amostras(
        caminho_configurado(config, "aumentado"),
        classes=classes,
        incluir_originais=False,
        incluir_sinteticas=True,
        classes_sem_mascara=classes_sem_mascara,
    )
    sinteticas_treino = [a for a in sinteticas if a.id_origem in ids_treino]
    if not sinteticas_treino:
        raise RuntimeError("Nenhuma imagem sintetica com mascara foi encontrada para o subconjunto de treino.")
    treino_aumentado = [*treino_original, *sinteticas_treino]
    return treino_original, treino_aumentado, validacao


def _salvar_divisao_experimental(
    config: dict[str, Any],
    treino_original: Sequence[Amostra],
    treino_aumentado: Sequence[Amostra],
    validacao: Sequence[Amostra],
) -> None:
    destino = caminho_configurado(config, "resultados") / "divisao_experimental.csv"
    destino.parent.mkdir(parents=True, exist_ok=True)
    linhas: list[dict[str, Any]] = []
    for cenario, treino in (("original", treino_original), ("aumentado", treino_aumentado)):
        for particao, amostras in (("treino", treino), ("validacao", validacao)):
            for amostra in amostras:
                linhas.append({
                    "cenario": cenario,
                    "particao": particao,
                    "classe": amostra.classe,
                    "id": amostra.id,
                    "sintetica": amostra.sintetica,
                    "id_origem": amostra.id_origem,
                    "imagem": caminho_portatil(config, amostra.imagem),
                })
    with destino.open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)


def treinar_dois_modelos(config: dict[str, Any]) -> list[dict[str, Any]]:
    treino_original, treino_aumentado, validacao = construir_cenarios(config)
    _salvar_divisao_experimental(config, treino_original, treino_aumentado, validacao)
    return [
        treinar_modelo("original", treino_original, validacao, config),
        treinar_modelo("aumentado", treino_aumentado, validacao, config),
    ]
