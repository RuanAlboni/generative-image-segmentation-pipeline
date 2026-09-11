from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def carregar_configuracao(caminho: str | Path) -> dict[str, Any]:
    """Carrega o YAML e registra a pasta usada para resolver caminhos relativos."""
    caminho = Path(caminho).expanduser().resolve()
    with caminho.open("r", encoding="utf-8") as arquivo:
        config = yaml.safe_load(arquivo) or {}
    config["_arquivo_config"] = str(caminho)
    config["_diretorio_base"] = str(caminho.parent)
    raiz_configurada = config.get("projeto", {}).get("raiz", ".")
    raiz_projeto = Path(raiz_configurada).expanduser()
    if not raiz_projeto.is_absolute():
        raiz_projeto = (caminho.parent / raiz_projeto).resolve()
    else:
        raiz_projeto = raiz_projeto.resolve()
    config["_raiz_projeto"] = str(raiz_projeto)
    return config


def caminho_do_projeto(config: dict[str, Any], valor: str | Path) -> Path:
    caminho = Path(valor).expanduser()
    if caminho.is_absolute():
        return caminho
    raiz = Path(config.get("_raiz_projeto", config["_diretorio_base"]))
    return (raiz / caminho).resolve()


def caminho_configurado(config: dict[str, Any], nome: str) -> Path:
    return caminho_do_projeto(config, config["caminhos"][nome])


def caminho_portatil(config: dict[str, Any], caminho: str | Path) -> str:
    """Serializa caminhos preferencialmente relativos ao projeto.

    Evita gravar caminhos absolutos com nomes de usuario em CSVs e logs quando
    os arquivos estao dentro da arvore do projeto. Caminhos externos continuam
    funcionais, mas sao reduzidos ao nome do arquivo para nao expor a estrutura
    local da maquina.
    """
    caminho = Path(caminho).expanduser().resolve()
    raiz = Path(config.get("_raiz_projeto", config["_diretorio_base"])).expanduser().resolve()
    try:
        return caminho.relative_to(raiz).as_posix()
    except ValueError:
        return caminho.name


def config_publica(config: dict[str, Any]) -> dict[str, Any]:
    """Copia a configuracao sem campos internos, adequada para checkpoints/logs."""
    copia = deepcopy(config)
    for chave in list(copia):
        if chave.startswith("_"):
            copia.pop(chave)
    return copia
