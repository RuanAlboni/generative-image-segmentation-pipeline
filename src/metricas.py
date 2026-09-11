from __future__ import annotations

from typing import Any

import numpy as np


def contagens_binarias(predicao: np.ndarray, alvo: np.ndarray) -> dict[str, int]:
    predicao = np.asarray(predicao, dtype=bool)
    alvo = np.asarray(alvo, dtype=bool)
    if predicao.shape != alvo.shape:
        raise ValueError(f"Formas diferentes: predicao={predicao.shape}, alvo={alvo.shape}")
    return {
        "tp": int(np.logical_and(predicao, alvo).sum()),
        "tn": int(np.logical_and(~predicao, ~alvo).sum()),
        "fp": int(np.logical_and(predicao, ~alvo).sum()),
        "fn": int(np.logical_and(~predicao, alvo).sum()),
    }


def somar_contagens(contagens: list[dict[str, int]]) -> dict[str, int]:
    return {chave: sum(c[chave] for c in contagens) for chave in ("tp", "tn", "fp", "fn")}


def metricas_de_contagens(c: dict[str, int]) -> dict[str, float]:
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]

    def divisao(numerador: float, denominador: float, vazio: float = 1.0) -> float:
        return float(numerador / denominador) if denominador else vazio

    iou = divisao(tp, tp + fp + fn)
    f1 = divisao(2 * tp, 2 * tp + fp + fn)
    return {
        "iou": iou,
        "f1": f1,
        "dice": f1,
        "precisao": divisao(tp, tp + fp),
        "revocacao": divisao(tp, tp + fn),
        "especificidade": divisao(tn, tn + fp),
        "acuracia": divisao(tp + tn, tp + tn + fp + fn),
    }


def metricas_binarias(predicao: np.ndarray, alvo: np.ndarray) -> dict[str, Any]:
    contagens = contagens_binarias(predicao, alvo)
    return {**contagens, **metricas_de_contagens(contagens)}
