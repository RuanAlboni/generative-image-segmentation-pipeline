from __future__ import annotations

from typing import Any


def criar_modelo(cfg: dict[str, Any], carregar_pesos_encoder: bool = True):
    try:
        import segmentation_models_pytorch as smp
    except ImportError as erro:
        raise RuntimeError("Instale segmentation-models-pytorch para criar a U-Net.") from erro
    if str(cfg.get("arquitetura", "Unet")).lower() != "unet":
        raise ValueError("Esta versao do experimento implementa a arquitetura Unet.")
    return smp.Unet(
        encoder_name=cfg.get("encoder", "mobilenet_v2"),
        encoder_weights=cfg.get("pesos_encoder", "imagenet") if carregar_pesos_encoder else None,
        in_channels=int(cfg.get("canais_entrada", 3)),
        classes=1,
        activation=None,
    )
