from __future__ import annotations

import csv
import random
from pathlib import Path
from statistics import median
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


METRICAS_RELATORIO = (
    "iou",
    "f1",
    "dice",
    "precisao",
    "revocacao",
    "especificidade",
    "acuracia",
)


def _dependencias_torch():
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as erro:
        raise RuntimeError("Instale PyTorch antes de executar o treinamento.") from erro
    return torch, nn, DataLoader, Dataset


def _stratified_kfold():
    try:
        from sklearn.model_selection import StratifiedKFold
    except ImportError as erro:
        raise RuntimeError(
            "A validacao cruzada requer scikit-learn. Instale as dependencias do projeto."
        ) from erro
    return StratifiedKFold


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
    destino.parent.mkdir(parents=True, exist_ok=True)
    if not historico:
        return
    with destino.open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(historico[0]))
        escritor.writeheader()
        escritor.writerows(historico)


def _plotar_historico(historico: list[dict[str, Any]], destino: Path, titulo: str) -> None:
    import matplotlib.pyplot as plt

    if not historico:
        return
    epocas = [linha["epoca"] for linha in historico]
    figura, eixos = plt.subplots(1, 2, figsize=(11, 4))
    eixos[0].plot(epocas, [x["treino_loss"] for x in historico], label="Treino")
    if "validacao_loss" in historico[0]:
        eixos[0].plot(epocas, [x["validacao_loss"] for x in historico], label="Validacao")
    eixos[0].set(title="Perda", xlabel="Epoca", ylabel="BCE + Dice")
    eixos[0].legend()

    if "validacao_iou" in historico[0]:
        eixos[1].plot(epocas, [x["validacao_iou"] for x in historico], label="IoU validacao")
        eixos[1].plot(epocas, [x["validacao_f1"] for x in historico], label="F1/Dice validacao")
        titulo_eixo = "Validacao"
    else:
        eixos[1].plot(epocas, [x["treino_iou"] for x in historico], label="IoU treino")
        eixos[1].plot(epocas, [x["treino_f1"] for x in historico], label="F1/Dice treino")
        titulo_eixo = "Treino final"
    eixos[1].set(title=titulo_eixo, xlabel="Epoca", ylabel="Metrica", ylim=(0, 1))
    eixos[1].legend()
    figura.suptitle(titulo)
    figura.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, dpi=150, bbox_inches="tight")
    plt.close(figura)


def _criar_carregador(amostras: Sequence[Amostra], cfg: dict[str, Any], dispositivo, semente: int, treino: bool):
    torch, _, DataLoader, _ = _dependencias_torch()
    dataset = DatasetSegmentacao(amostras, cfg, treino=treino)
    gerador = torch.Generator().manual_seed(semente) if treino else None
    return DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=treino,
        num_workers=int(cfg.get("num_workers", 0)),
        generator=gerador,
        pin_memory=dispositivo.type == "cuda",
    )


def treinar_modelo(
    nome: str,
    treino: Sequence[Amostra],
    validacao: Sequence[Amostra],
    config: dict[str, Any],
    *,
    pasta_modelos: Path | None = None,
    pasta_resultados: Path | None = None,
    semente: int | None = None,
    checkpoint_nome: str | None = None,
    rotulo_log: str | None = None,
) -> dict[str, Any]:
    """Treina com validacao/early stopping; usado nos folds e no modo simples."""
    torch, _, _, _ = _dependencias_torch()
    cfg = config["treinamento"]
    semente = int(config["experimento"]["semente"] if semente is None else semente)
    fixar_semente(semente, bool(config["experimento"].get("deterministico", True)))
    dispositivo = obter_dispositivo(cfg.get("dispositivo", "auto"))
    carregador_treino = _criar_carregador(treino, cfg, dispositivo, semente, treino=True)
    carregador_validacao = _criar_carregador(validacao, cfg, dispositivo, semente, treino=False)
    modelo = criar_modelo(config["modelo_segmentacao"]).to(dispositivo)
    otimizador = torch.optim.AdamW(
        modelo.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
    )
    perda_fn = PerdaBCEDice(float(cfg["peso_bce"]), float(cfg["peso_dice"]))
    amp = bool(cfg.get("amp", True)) and dispositivo.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=True) if amp else None
    limiar = float(cfg.get("limiar", 0.5))
    melhor_iou = -1.0
    melhor_epoca = 0
    epocas_sem_melhora = 0
    historico: list[dict[str, Any]] = []
    pasta_modelos = pasta_modelos or caminho_configurado(config, "modelos")
    pasta_resultados = pasta_resultados or (caminho_configurado(config, "resultados") / "treinamento")
    pasta_modelos.mkdir(parents=True, exist_ok=True)
    pasta_resultados.mkdir(parents=True, exist_ok=True)
    checkpoint = pasta_modelos / (checkpoint_nome or f"unet_{nome}.pt")
    prefixo = rotulo_log or nome

    for epoca in range(1, int(cfg["epocas"]) + 1):
        treino_m = _epoca(modelo, carregador_treino, perda_fn, dispositivo, limiar, otimizador, scaler)
        validacao_m = _epoca(modelo, carregador_validacao, perda_fn, dispositivo, limiar)
        linha = {"epoca": epoca}
        linha.update({f"treino_{k}": v for k, v in treino_m.items()})
        linha.update({f"validacao_{k}": v for k, v in validacao_m.items()})
        historico.append(linha)
        if validacao_m["iou"] > melhor_iou + float(cfg.get("min_delta", 0.0)):
            melhor_iou = validacao_m["iou"]
            melhor_epoca = epoca
            epocas_sem_melhora = 0
            torch.save(
                {
                    "estado_modelo": modelo.state_dict(),
                    "nome_experimento": nome,
                    "epoca": epoca,
                    "semente_treinamento": semente,
                    "metricas_validacao": validacao_m,
                    "modelo_segmentacao": config["modelo_segmentacao"],
                    "treinamento": cfg,
                    "configuracao": config_publica(config),
                },
                checkpoint,
            )
        else:
            epocas_sem_melhora += 1
        print(
            f"[{prefixo}] epoca {epoca:03d} | loss val={validacao_m['loss']:.4f} "
            f"| IoU={validacao_m['iou']:.4f} | F1={validacao_m['f1']:.4f}"
        )
        if epocas_sem_melhora >= int(cfg.get("paciencia", 10)):
            break

    _salvar_historico(historico, pasta_resultados / f"historico_{nome}.csv")
    _plotar_historico(historico, pasta_resultados / f"curvas_{nome}.png", f"U-Net - {prefixo}")
    return {
        "nome": nome,
        "checkpoint": str(checkpoint),
        "melhor_iou_validacao": melhor_iou,
        "melhor_epoca": melhor_epoca,
        "epocas_executadas": len(historico),
        "treino": len(treino),
        "validacao": len(validacao),
        "contagem_treino": contar_por_classe(treino, classes_configuradas(config)),
    }


def treinar_modelo_final(
    nome: str,
    treino: Sequence[Amostra],
    config: dict[str, Any],
    epocas: int,
    *,
    semente: int | None = None,
) -> dict[str, Any]:
    """Treina o modelo final usando todo o desenvolvimento e epocas definidas pelo k-fold."""
    torch, _, _, _ = _dependencias_torch()
    cfg = config["treinamento"]
    semente = int(config["experimento"]["semente"] if semente is None else semente)
    fixar_semente(semente, bool(config["experimento"].get("deterministico", True)))
    dispositivo = obter_dispositivo(cfg.get("dispositivo", "auto"))
    carregador = _criar_carregador(treino, cfg, dispositivo, semente, treino=True)
    modelo = criar_modelo(config["modelo_segmentacao"]).to(dispositivo)
    otimizador = torch.optim.AdamW(
        modelo.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"])
    )
    perda_fn = PerdaBCEDice(float(cfg["peso_bce"]), float(cfg["peso_dice"]))
    amp = bool(cfg.get("amp", True)) and dispositivo.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=True) if amp else None
    limiar = float(cfg.get("limiar", 0.5))
    historico: list[dict[str, Any]] = []

    for epoca in range(1, max(1, int(epocas)) + 1):
        treino_m = _epoca(modelo, carregador, perda_fn, dispositivo, limiar, otimizador, scaler)
        linha = {"epoca": epoca, **{f"treino_{k}": v for k, v in treino_m.items()}}
        historico.append(linha)
        print(
            f"[final/{nome}] epoca {epoca:03d} | loss={treino_m['loss']:.4f} "
            f"| IoU={treino_m['iou']:.4f} | F1={treino_m['f1']:.4f}"
        )

    pasta_modelos = caminho_configurado(config, "modelos")
    pasta_resultados = caminho_configurado(config, "resultados") / "treinamento_final"
    pasta_modelos.mkdir(parents=True, exist_ok=True)
    pasta_resultados.mkdir(parents=True, exist_ok=True)
    checkpoint = pasta_modelos / f"unet_{nome}.pt"
    torch.save(
        {
            "estado_modelo": modelo.state_dict(),
            "nome_experimento": nome,
            "tipo_treinamento": "final_todo_desenvolvimento",
            "epoca": int(epocas),
            "semente_treinamento": semente,
            "metricas_treino": {k.removeprefix("treino_"): v for k, v in historico[-1].items() if k.startswith("treino_")},
            "modelo_segmentacao": config["modelo_segmentacao"],
            "treinamento": cfg,
            "configuracao": config_publica(config),
        },
        checkpoint,
    )
    _salvar_historico(historico, pasta_resultados / f"historico_{nome}.csv")
    _plotar_historico(historico, pasta_resultados / f"curvas_{nome}.png", f"U-Net final - {nome}")
    return {
        "nome": nome,
        "checkpoint": str(checkpoint),
        "epocas": int(epocas),
        "treino": len(treino),
        "contagem_treino": contar_por_classe(treino, classes_configuradas(config)),
    }


def gerar_folds_estratificados(
    originais: Sequence[Amostra],
    n_splits: int,
    semente: int,
    embaralhar: bool = True,
) -> list[tuple[list[Amostra], list[Amostra]]]:
    """Gera folds estratificados pelas categorias do dataset usando sklearn.StratifiedKFold."""
    if n_splits < 2:
        raise ValueError("validacao_cruzada.folds deve ser >= 2")
    originais = sorted((a for a in originais if not a.sintetica), key=lambda a: a.id)
    if not originais:
        raise ValueError("Nao ha imagens originais no conjunto de desenvolvimento.")
    contagens = contar_por_classe(originais)
    insuficientes = {classe: n for classe, n in contagens.items() if n < n_splits}
    if insuficientes:
        detalhe = ", ".join(f"{classe}={n}" for classe, n in sorted(insuficientes.items()))
        raise ValueError(
            f"StratifiedKFold com {n_splits} folds exige ao menos {n_splits} amostras por classe; "
            f"classes insuficientes: {detalhe}."
        )

    StratifiedKFold = _stratified_kfold()
    classes = np.asarray([a.classe for a in originais])
    indices = np.arange(len(originais))
    kfold = StratifiedKFold(
        n_splits=n_splits,
        shuffle=bool(embaralhar),
        random_state=semente if embaralhar else None,
    )
    folds: list[tuple[list[Amostra], list[Amostra]]] = []
    for idx_treino, idx_validacao in kfold.split(indices, classes):
        treino = [originais[int(i)] for i in idx_treino]
        validacao = [originais[int(i)] for i in idx_validacao]
        folds.append((treino, validacao))
    return folds


def construir_cenarios_para_divisao(
    treino_original: Sequence[Amostra],
    validacao: Sequence[Amostra],
    sinteticas: Sequence[Amostra],
) -> tuple[list[Amostra], list[Amostra], list[Amostra]]:
    """Monta os dois cenarios garantindo que sinteticas da validacao nao vazem para treino."""
    treino_original = sorted(treino_original, key=lambda a: a.id)
    validacao = sorted(validacao, key=lambda a: a.id)
    ids_treino = {a.id for a in treino_original}
    ids_validacao = {a.id for a in validacao}
    if ids_treino & ids_validacao:
        raise RuntimeError("A divisao contem amostras simultaneamente em treino e validacao.")
    sinteticas_treino = sorted(
        (a for a in sinteticas if a.id_origem in ids_treino),
        key=lambda a: a.id,
    )
    vazamento = [a.id for a in sinteticas_treino if a.id_origem in ids_validacao]
    if vazamento:
        raise RuntimeError(f"Vazamento de sinteticas da validacao detectado: {vazamento[:5]}")
    if not sinteticas_treino:
        raise RuntimeError("Nenhuma imagem sintetica com mascara foi encontrada para o subconjunto de treino.")
    return treino_original, [*treino_original, *sinteticas_treino], validacao


def _carregar_desenvolvimento_e_sinteticas(config: dict[str, Any]) -> tuple[list[Amostra], list[Amostra]]:
    classes = classes_configuradas(config)
    classes_sem_mascara = classes_sem_mascara_configuradas(config)
    originais = coletar_amostras(
        caminho_configurado(config, "original"),
        classes=classes,
        incluir_sinteticas=False,
        classes_sem_mascara=classes_sem_mascara,
    )
    sinteticas = coletar_amostras(
        caminho_configurado(config, "aumentado"),
        classes=classes,
        incluir_originais=False,
        incluir_sinteticas=True,
        classes_sem_mascara=classes_sem_mascara,
    )
    return originais, sinteticas


def construir_cenarios(config: dict[str, Any]) -> tuple[list[Amostra], list[Amostra], list[Amostra]]:
    """Mantem o holdout simples para depuracao rapida."""
    classes = classes_configuradas(config)
    originais, sinteticas = _carregar_desenvolvimento_e_sinteticas(config)
    treino_original, validacao = dividir_originais(
        originais,
        float(config["divisao"].get("proporcao_treino", 0.8)),
        int(config["experimento"]["semente"]),
        classes=classes,
    )
    return construir_cenarios_para_divisao(treino_original, validacao, sinteticas)


def _salvar_divisao_experimental(
    config: dict[str, Any],
    treino_original: Sequence[Amostra],
    treino_aumentado: Sequence[Amostra],
    validacao: Sequence[Amostra],
    destino: Path | None = None,
    fold: int | None = None,
) -> Path:
    destino = destino or (caminho_configurado(config, "resultados") / "divisao_experimental.csv")
    destino.parent.mkdir(parents=True, exist_ok=True)
    linhas: list[dict[str, Any]] = []
    for cenario, treino in (("original", treino_original), ("aumentado", treino_aumentado)):
        for particao, amostras in (("treino", treino), ("validacao", validacao)):
            for amostra in amostras:
                linhas.append(
                    {
                        "fold": fold if fold is not None else "",
                        "cenario": cenario,
                        "particao": particao,
                        "classe": amostra.classe,
                        "id": amostra.id,
                        "sintetica": amostra.sintetica,
                        "id_origem": amostra.id_origem,
                        "imagem": caminho_portatil(config, amostra.imagem),
                    }
                )
    with destino.open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)
    return destino


def treinar_dois_modelos(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Modo simples de depuracao: uma unica divisao treino/validacao."""
    treino_original, treino_aumentado, validacao = construir_cenarios(config)
    _salvar_divisao_experimental(config, treino_original, treino_aumentado, validacao)
    return [
        treinar_modelo("original", treino_original, validacao, config),
        treinar_modelo("aumentado", treino_aumentado, validacao, config),
    ]


def _salvar_manifesto_folds(config: dict[str, Any], folds: Sequence[tuple[Sequence[Amostra], Sequence[Amostra]]], destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8", newline="") as arquivo:
        campos = ["fold", "particao", "classe", "id", "imagem"]
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        for numero_fold, (treino, validacao) in enumerate(folds, start=1):
            for particao, amostras in (("treino", treino), ("validacao", validacao)):
                for amostra in amostras:
                    escritor.writerow(
                        {
                            "fold": numero_fold,
                            "particao": particao,
                            "classe": amostra.classe,
                            "id": amostra.id,
                            "imagem": caminho_portatil(config, amostra.imagem),
                        }
                    )


def _agregar_resumos_folds(linhas: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    agregados: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    modelos = sorted({str(l["modelo"]) for l in linhas})
    escopos = sorted({str(l["escopo"]) for l in linhas})
    for modelo in modelos:
        for escopo in escopos:
            subconjunto = [l for l in linhas if l["modelo"] == modelo and l["escopo"] == escopo]
            if not subconjunto:
                continue
            item: dict[str, Any] = {
                "modelo": modelo,
                "escopo": escopo,
                "folds": len(subconjunto),
                "n_medio": float(np.mean([float(l["n"]) for l in subconjunto])),
            }
            for metrica in METRICAS_RELATORIO:
                valores = np.asarray([float(l[metrica]) for l in subconjunto], dtype=float)
                item[f"{metrica}_media"] = float(np.mean(valores))
                item[f"{metrica}_desvio"] = float(np.std(valores, ddof=1)) if len(valores) > 1 else 0.0
            agregados.append(item)

    por_chave = {(int(l["fold"]), str(l["escopo"]), str(l["modelo"])): l for l in linhas}
    folds = sorted({int(l["fold"]) for l in linhas})
    for fold in folds:
        for escopo in escopos:
            original = por_chave.get((fold, escopo, "original"))
            aumentado = por_chave.get((fold, escopo, "aumentado"))
            if original is None or aumentado is None:
                continue
            for metrica in METRICAS_RELATORIO:
                valor_original = float(original[metrica])
                valor_aumentado = float(aumentado[metrica])
                deltas.append(
                    {
                        "fold": fold,
                        "escopo": escopo,
                        "metrica": metrica,
                        "original": valor_original,
                        "aumentado": valor_aumentado,
                        "delta_aumentado_menos_original": valor_aumentado - valor_original,
                    }
                )
    return agregados, deltas


def _salvar_csv(linhas: Sequence[dict[str, Any]], destino: Path) -> None:
    if not linhas:
        return
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(linhas[0]))
        escritor.writeheader()
        escritor.writerows(linhas)


def _plotar_resumo_kfold(agregados: Sequence[dict[str, Any]], destino: Path, escopo: str) -> None:
    import matplotlib.pyplot as plt

    linhas = {str(l["modelo"]): l for l in agregados if l["escopo"] == escopo}
    if "original" not in linhas or "aumentado" not in linhas:
        return
    metricas = ("iou", "f1", "precisao", "revocacao", "especificidade", "acuracia")
    rotulos = ("IoU", "F1/Dice", "Precisao", "Revocacao", "Especificidade", "Acuracia")
    x = np.arange(len(metricas))
    largura = 0.36
    figura, eixo = plt.subplots(figsize=(11, 5.5))
    for indice, modelo in enumerate(("original", "aumentado")):
        linha = linhas[modelo]
        medias = [float(linha[f"{m}_media"]) for m in metricas]
        desvios = [float(linha[f"{m}_desvio"]) for m in metricas]
        eixo.bar(
            x + (indice - 0.5) * largura,
            medias,
            largura,
            yerr=desvios,
            capsize=3,
            label=modelo.capitalize(),
        )
    eixo.set_xticks(x, rotulos, rotation=25, ha="right")
    eixo.set_ylim(0, 1.08)
    eixo.set_ylabel("Media entre folds")
    eixo.set_title(f"Validacao cruzada - {escopo} (media +/- desvio padrao)")
    eixo.grid(axis="y", alpha=0.2)
    eixo.legend()
    figura.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(destino, dpi=170, bbox_inches="tight")
    plt.close(figura)


def _epocas_finais(resultados_treino: Sequence[dict[str, Any]], nome: str) -> int:
    epocas = [int(r["melhor_epoca"]) for r in resultados_treino if r["nome"] == nome and int(r["melhor_epoca"]) > 0]
    if not epocas:
        raise RuntimeError(f"Nao foi possivel determinar a melhor epoca para o modelo {nome}.")
    return max(1, int(round(median(epocas))))


def executar_validacao_cruzada(config: dict[str, Any]) -> dict[str, Any]:
    """Executa StratifiedKFold no desenvolvimento e treina os modelos finais.

    O teste externo nao participa dos folds. Em cada fold, o cenario aumentado
    recebe somente sinteticas derivadas das imagens originais presentes no treino.
    """
    cfg_cv = config.get("validacao_cruzada", {})
    n_splits = int(cfg_cv.get("folds", 5))
    embaralhar = bool(cfg_cv.get("embaralhar", True))
    semente_base = int(config["experimento"]["semente"])
    originais, sinteticas = _carregar_desenvolvimento_e_sinteticas(config)
    if not sinteticas:
        raise RuntimeError(
            "Nenhuma imagem sintetica foi encontrada. Execute 'difusao' e 'mascaras' antes da validacao cruzada."
        )
    folds = gerar_folds_estratificados(originais, n_splits, semente_base, embaralhar)

    resultados_raiz = caminho_configurado(config, "resultados") / "validacao_cruzada"
    modelos_raiz = caminho_configurado(config, "modelos") / "validacao_cruzada"
    resultados_raiz.mkdir(parents=True, exist_ok=True)
    modelos_raiz.mkdir(parents=True, exist_ok=True)
    _salvar_manifesto_folds(config, folds, resultados_raiz / "folds.csv")

    resumos_folds: list[dict[str, Any]] = []
    resultados_treino: list[dict[str, Any]] = []

    # Importacao local evita ciclo na inicializacao dos modulos.
    from .avaliacao import avaliar_checkpoints_em_amostras

    for numero_fold, (treino_originais, validacao) in enumerate(folds, start=1):
        print(f"\n=== Fold {numero_fold}/{n_splits} ===")
        treino_original, treino_aumentado, validacao = construir_cenarios_para_divisao(
            treino_originais, validacao, sinteticas
        )
        pasta_fold_resultados = resultados_raiz / f"fold_{numero_fold:02d}"
        pasta_fold_modelos = modelos_raiz / f"fold_{numero_fold:02d}"
        _salvar_divisao_experimental(
            config,
            treino_original,
            treino_aumentado,
            validacao,
            destino=pasta_fold_resultados / "divisao.csv",
            fold=numero_fold,
        )
        semente_fold = semente_base + numero_fold - 1
        treino_fold: list[dict[str, Any]] = []
        for nome, treino in (("original", treino_original), ("aumentado", treino_aumentado)):
            resultado = treinar_modelo(
                nome,
                treino,
                validacao,
                config,
                pasta_modelos=pasta_fold_modelos,
                pasta_resultados=pasta_fold_resultados / "treinamento",
                semente=semente_fold,
                rotulo_log=f"fold {numero_fold}/{nome}",
            )
            resultado = {"fold": numero_fold, **resultado}
            treino_fold.append(resultado)
            resultados_treino.append(resultado)

        caminhos = {
            "original": pasta_fold_modelos / "unet_original.pt",
            "aumentado": pasta_fold_modelos / "unet_aumentado.pt",
        }
        avaliacao_fold = avaliar_checkpoints_em_amostras(
            config,
            validacao,
            caminhos,
            pasta_fold_resultados / "avaliacao",
            salvar_sobreposicoes=False,
            gerar_analise=False,
            titulo_contexto=f"validacao do fold {numero_fold}",
        )
        for linha in avaliacao_fold["resumo"]:
            resumos_folds.append({"fold": numero_fold, **linha})

    _salvar_csv(resultados_treino, resultados_raiz / "treinamentos_por_fold.csv")
    _salvar_csv(resumos_folds, resultados_raiz / "metricas_por_fold.csv")
    agregados, deltas = _agregar_resumos_folds(resumos_folds)
    _salvar_csv(agregados, resultados_raiz / "resumo_validacao_cruzada.csv")
    _salvar_csv(deltas, resultados_raiz / "comparacao_pareada_por_fold.csv")
    for escopo in ("macro_todas", "macro_regiao_presente", "micro_pixels"):
        _plotar_resumo_kfold(agregados, resultados_raiz / f"comparacao_{escopo}.png", escopo)

    resultado: dict[str, Any] = {
        "folds": n_splits,
        "desenvolvimento": len(originais),
        "sinteticas_disponiveis": len(sinteticas),
        "resumo_validacao_cruzada": agregados,
        "pasta_resultados": str(resultados_raiz),
    }

    if bool(cfg_cv.get("treinar_modelos_finais", True)):
        epocas_original = _epocas_finais(resultados_treino, "original")
        epocas_aumentado = _epocas_finais(resultados_treino, "aumentado")
        ids_desenvolvimento = {a.id for a in originais}
        sinteticas_desenvolvimento = [a for a in sinteticas if a.id_origem in ids_desenvolvimento]
        treino_aumentado_final = [*sorted(originais, key=lambda a: a.id), *sorted(sinteticas_desenvolvimento, key=lambda a: a.id)]
        finais = [
            treinar_modelo_final("original", originais, config, epocas_original, semente=semente_base),
            treinar_modelo_final("aumentado", treino_aumentado_final, config, epocas_aumentado, semente=semente_base),
        ]
        resultado["epocas_finais"] = {
            "original": epocas_original,
            "aumentado": epocas_aumentado,
            "criterio": "mediana da melhor epoca observada nos folds",
        }
        resultado["modelos_finais"] = finais

    return resultado


def treinar_experimento(config: dict[str, Any], modo_simples: bool = False) -> Any:
    if modo_simples:
        return treinar_dois_modelos(config)
    return executar_validacao_cruzada(config)
