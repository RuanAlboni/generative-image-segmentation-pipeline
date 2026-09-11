from __future__ import annotations

import csv
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .configuracao import caminho_configurado, caminho_do_projeto
from .dados import (
    Amostra,
    classes_configuradas,
    classes_sem_mascara_configuradas,
    coletar_amostras,
    dividir_originais,
)

EXTENSOES_CHECKPOINT = {".safetensors", ".ckpt"}


def _semente_global(semente: int) -> None:
    random.seed(semente)
    np.random.seed(semente)


def _multiplo(valor: int, base: int = 8) -> int:
    return max(base, round(valor / base) * base)


def _imagem_para_difusao(caminho: Path, multiplo: int) -> tuple[Image.Image, tuple[int, int]]:
    with Image.open(caminho) as arquivo:
        imagem = arquivo.convert("L").convert("RGB")
    tamanho_original = imagem.size
    tamanho_modelo = (_multiplo(imagem.width, multiplo), _multiplo(imagem.height, multiplo))
    if imagem.size != tamanho_modelo:
        imagem = imagem.resize(tamanho_modelo, Image.Resampling.LANCZOS)
    return imagem, tamanho_original


def _prompt(amostra: Amostra, cfg: dict[str, Any]) -> str:
    comum = str(cfg["prompts"].get("comum", "")).strip()
    classe = str(cfg["prompts"].get("por_classe", {}).get(amostra.classe, "")).strip()
    return ", ".join(parte for parte in (comum, classe) if parte)


def _resolver_modelo(config: dict[str, Any], cfg: dict[str, Any]) -> str:
    """Resolve checkpoints/pastas locais em relacao ao config; preserva IDs do Hub."""
    valor = str(cfg["modelo"]).strip()
    candidato = Path(valor).expanduser()
    candidato_projeto = caminho_do_projeto(config, candidato) if not candidato.is_absolute() else candidato
    parece_checkpoint = candidato.suffix.lower() in EXTENSOES_CHECKPOINT
    if parece_checkpoint or candidato.is_absolute() or candidato_projeto.exists():
        if not candidato_projeto.exists():
            raise FileNotFoundError(
                "Modelo local nao encontrado: "
                f"{candidato_projeto}. Coloque o checkpoint nesse caminho ou altere difusao.modelo."
            )
        return str(candidato_projeto.resolve())
    return valor


def _resolver_config_checkpoint(config: dict[str, Any], valor: str | None) -> str | None:
    if not valor:
        return None
    candidato = Path(valor).expanduser()
    candidato_projeto = caminho_do_projeto(config, candidato) if not candidato.is_absolute() else candidato
    if candidato.is_absolute() or candidato_projeto.exists():
        if not candidato_projeto.exists():
            raise FileNotFoundError(f"Configuracao local do checkpoint nao encontrada: {candidato_projeto}")
        return str(candidato_projeto.resolve())
    return valor


def _carregar_pipeline(config: dict[str, Any], cfg: dict[str, Any], modelo: str):
    try:
        import torch
        from diffusers import DPMSolverMultistepScheduler, StableDiffusionImg2ImgPipeline, StableDiffusionXLImg2ImgPipeline
    except ImportError as erro:
        raise RuntimeError("Instale torch, diffusers e transformers antes de executar a difusao.") from erro

    dispositivo_cfg = cfg.get("dispositivo", "auto")
    dispositivo = "cuda" if dispositivo_cfg == "auto" and torch.cuda.is_available() else dispositivo_cfg
    if dispositivo == "auto":
        dispositivo = "cpu"
    if dispositivo == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA foi solicitada, mas nenhuma GPU CUDA esta disponivel.")
    dtype = torch.float16 if dispositivo == "cuda" else torch.float32

    kwargs = {
        "torch_dtype": dtype,
        "use_safetensors": True,
        "local_files_only": bool(cfg.get("somente_arquivos_locais", False)),
    }
    tipo = str(cfg.get("pipeline", "sdxl")).lower()
    caminho_modelo = Path(modelo)
    arquivo_unico = caminho_modelo.is_file() and caminho_modelo.suffix.lower() in EXTENSOES_CHECKPOINT
    if tipo == "sdxl":
        if arquivo_unico:
            kwargs_single_file = {
                "torch_dtype": dtype,
                "local_files_only": bool(cfg.get("somente_arquivos_locais", False)),
                "add_watermarker": False,
            }
            config_checkpoint = _resolver_config_checkpoint(config, cfg.get("config_checkpoint"))
            if config_checkpoint:
                kwargs_single_file["config"] = config_checkpoint
            pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(modelo, **kwargs_single_file)
        else:
            pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(modelo, add_watermarker=False, **kwargs)
    elif tipo in {"sd", "sd15", "stable-diffusion"}:
        if arquivo_unico:
            pipe = StableDiffusionImg2ImgPipeline.from_single_file(
                modelo,
                torch_dtype=dtype,
                local_files_only=bool(cfg.get("somente_arquivos_locais", False)),
                safety_checker=None,
                requires_safety_checker=False,
            )
        else:
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                modelo, safety_checker=None, requires_safety_checker=False, **kwargs
            )
    else:
        raise ValueError("difusao.pipeline deve ser 'sdxl' ou 'sd15'")

    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=bool(cfg.get("karras_sigmas", True))
    )
    if dispositivo == "cuda" and cfg.get("offload_cpu", False):
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(dispositivo)
    if cfg.get("attention_slicing", True):
        pipe.enable_attention_slicing()
    if cfg.get("vae_tiling", True) and hasattr(pipe, "enable_vae_tiling"):
        pipe.enable_vae_tiling()
    return pipe, torch, dispositivo


def gerar_imagens_sinteticas(
    config: dict[str, Any],
    limite: int | None = None,
    simulacao: bool = False,
) -> dict[str, int]:
    """Gera variacoes apenas das amostras pertencentes ao subconjunto de treino."""
    cfg = config["difusao"]
    original = caminho_configurado(config, "original")
    aumentado = caminho_configurado(config, "aumentado")
    classes = classes_configuradas(config)
    classes_sem_mascara = classes_sem_mascara_configuradas(config)
    amostras = coletar_amostras(
        original,
        classes=classes,
        incluir_sinteticas=False,
        classes_sem_mascara=classes_sem_mascara,
    )
    treino, _ = dividir_originais(
        amostras,
        proporcao_treino=float(config["divisao"]["proporcao_treino"]),
        semente=int(config["experimento"]["semente"]),
        classes=classes,
    )
    if limite is not None:
        treino = treino[:limite]
    variacoes = int(cfg.get("variacoes_por_imagem", 1))
    resumo = {"origens_elegiveis": len(treino), "geradas": 0, "ignoradas_existentes": 0}
    if simulacao:
        return resumo

    modelo = _resolver_modelo(config, cfg)
    pipe, torch, dispositivo = _carregar_pipeline(config, cfg, modelo)
    semente = int(config["experimento"]["semente"])
    _semente_global(semente)
    manifesto = aumentado / "manifesto_sinteticas.csv"
    manifesto.parent.mkdir(parents=True, exist_ok=True)
    escrever_cabecalho = not manifesto.exists()

    with manifesto.open("a", encoding="utf-8", newline="") as arquivo:
        campos = [
            "classe", "imagem_origem", "imagem_sintetica", "variacao", "semente",
            "modelo", "strength", "passos", "cfg", "prompt", "tempo_segundos",
        ]
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        if escrever_cabecalho:
            escritor.writeheader()

        for indice, amostra in enumerate(treino):
            imagem_entrada, tamanho_original = _imagem_para_difusao(
                amostra.imagem, int(cfg.get("multiplo_resolucao", 8))
            )
            for variacao in range(1, variacoes + 1):
                saida = aumentado / amostra.classe / f"{amostra.imagem.stem}__sint_{variacao:02d}.png"
                if saida.exists() and cfg.get("ignorar_existentes", True):
                    resumo["ignoradas_existentes"] += 1
                    continue
                saida.parent.mkdir(parents=True, exist_ok=True)
                semente_atual = semente + indice * variacoes + variacao - 1
                gerador = torch.Generator(device=dispositivo).manual_seed(semente_atual)
                inicio = time.perf_counter()
                resultado = pipe(
                    prompt=_prompt(amostra, cfg),
                    negative_prompt=cfg["prompts"].get("negativo", ""),
                    image=imagem_entrada,
                    strength=float(cfg["strength"]),
                    guidance_scale=float(cfg["guidance_scale"]),
                    num_inference_steps=int(cfg["passos"]),
                    generator=gerador,
                ).images[0]
                resultado = resultado.resize(tamanho_original, Image.Resampling.LANCZOS).convert("L")
                resultado.save(saida)
                escritor.writerow({
                    "classe": amostra.classe,
                    "imagem_origem": str(amostra.imagem.relative_to(original)),
                    "imagem_sintetica": str(saida.relative_to(aumentado)),
                    "variacao": variacao,
                    "semente": semente_atual,
                    "modelo": cfg["modelo"],
                    "strength": cfg["strength"],
                    "passos": cfg["passos"],
                    "cfg": cfg["guidance_scale"],
                    "prompt": _prompt(amostra, cfg),
                    "tempo_segundos": round(time.perf_counter() - inicio, 3),
                })
                arquivo.flush()
                resumo["geradas"] += 1
    return resumo
