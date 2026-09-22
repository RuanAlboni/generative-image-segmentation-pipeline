from __future__ import annotations

import csv
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image

EXTENSOES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PADRAO_SINTETICA = re.compile(r"^(?P<origem>.+)__sint_(?P<numero>\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class Amostra:
    imagem: Path
    classe: str
    mascaras: tuple[Path, ...]
    sintetica: bool = False
    origem: str | None = None

    @property
    def id(self) -> str:
        return f"{self.classe}/{self.imagem.stem}"

    @property
    def id_origem(self) -> str:
        return f"{self.classe}/{self.origem or self.imagem.stem}"


def classes_configuradas(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Retorna as classes de casos/imagens declaradas em `dados.classes`."""
    classes = config.get("dados", {}).get("classes")
    if not isinstance(classes, (list, tuple)) or not classes:
        raise ValueError(
            "Defina uma lista nao vazia em dados.classes no config.yaml. "
            "Essas classes organizam/estratificam os casos; a segmentacao continua binaria."
        )
    resultado = tuple(str(classe).strip() for classe in classes)
    if any(not classe for classe in resultado):
        raise ValueError("dados.classes nao pode conter nomes vazios.")
    if len({classe.casefold() for classe in resultado}) != len(resultado):
        raise ValueError("dados.classes contem nomes duplicados (ignorando maiusculas/minusculas).")
    return resultado


def classes_sem_mascara_configuradas(config: Mapping[str, Any]) -> frozenset[str]:
    """Classes nas quais ausencia de arquivo de mascara significa alvo binario vazio."""
    classes = classes_configuradas(config)
    por_nome = {classe.casefold(): classe for classe in classes}
    valores = config.get("dados", {}).get("classes_sem_mascara", []) or []
    if not isinstance(valores, (list, tuple)):
        raise ValueError("dados.classes_sem_mascara deve ser uma lista.")
    resultado: set[str] = set()
    for valor in valores:
        chave = str(valor).strip().casefold()
        if chave not in por_nome:
            raise ValueError(f"Classe em dados.classes_sem_mascara nao declarada em dados.classes: {valor}")
        resultado.add(por_nome[chave])
    return frozenset(resultado)


def mapa_classes_configurado(config: Mapping[str, Any]) -> dict[str, str]:
    """Mapeia nomes/aliases encontrados no ZIP para os nomes internos configurados."""
    classes = classes_configuradas(config)
    mapa = {classe.casefold(): classe for classe in classes}
    aliases = config.get("dados", {}).get("aliases", {}) or {}
    if not isinstance(aliases, Mapping):
        raise ValueError("dados.aliases deve ser um mapa alias: classe.")
    classes_por_nome = {classe.casefold(): classe for classe in classes}
    for alias, destino in aliases.items():
        alias_norm = str(alias).strip().casefold()
        destino_norm = str(destino).strip().casefold()
        if not alias_norm:
            raise ValueError("dados.aliases nao pode conter alias vazio.")
        if destino_norm not in classes_por_nome:
            raise ValueError(f"Alias {alias!r} aponta para classe nao declarada: {destino!r}")
        classe_destino = classes_por_nome[destino_norm]
        existente = mapa.get(alias_norm)
        if existente is not None and existente != classe_destino:
            raise ValueError(f"Alias ambiguo {alias!r}: {existente!r} e {classe_destino!r}")
        mapa[alias_norm] = classe_destino
    return mapa


def e_mascara(caminho: Path) -> bool:
    return "_mask" in caminho.stem.lower()


def identificar_sintetica(caminho: Path) -> tuple[bool, str | None]:
    correspondencia = PADRAO_SINTETICA.match(caminho.stem)
    if not correspondencia:
        return False, None
    return True, correspondencia.group("origem")


def encontrar_mascaras(imagem: Path) -> tuple[Path, ...]:
    candidatos: list[Path] = []
    for caminho in imagem.parent.glob(f"{imagem.stem}_mask*"):
        if caminho.is_file() and caminho.suffix.lower() in EXTENSOES:
            candidatos.append(caminho)
    return tuple(sorted(candidatos))


def descobrir_classes(raiz: str | Path) -> tuple[str, ...]:
    raiz = Path(raiz)
    if not raiz.exists():
        return ()
    return tuple(sorted(p.name for p in raiz.iterdir() if p.is_dir() and not p.name.startswith(".")))


def coletar_amostras(
    raiz: str | Path,
    classes: Sequence[str] | None = None,
    incluir_originais: bool = True,
    incluir_sinteticas: bool = True,
    exigir_mascara: bool = True,
    classes_sem_mascara: Iterable[str] = (),
) -> list[Amostra]:
    """Coleta imagens por classe e associa arquivos `<imagem>_mask*`.

    A ausencia de mascara so e aceita quando `exigir_mascara=False` ou quando a
    classe aparece em `classes_sem_mascara`. Nesses casos a referencia binaria e vazia.
    """
    raiz = Path(raiz)
    classes = tuple(classes) if classes is not None else descobrir_classes(raiz)
    sem_mascara = set(classes_sem_mascara)
    amostras: list[Amostra] = []
    for classe in classes:
        pasta = raiz / classe
        if not pasta.exists():
            continue
        for imagem in sorted(pasta.iterdir()):
            if not imagem.is_file() or imagem.suffix.lower() not in EXTENSOES or e_mascara(imagem):
                continue
            sintetica, origem = identificar_sintetica(imagem)
            if sintetica and not incluir_sinteticas:
                continue
            if not sintetica and not incluir_originais:
                continue
            mascaras = encontrar_mascaras(imagem)
            if exigir_mascara and classe not in sem_mascara and not mascaras:
                raise FileNotFoundError(f"Imagem sem mascara em classe que exige anotacao: {imagem}")
            amostras.append(Amostra(imagem, classe, mascaras, sintetica, origem))
    return amostras


def carregar_mascara_unificada(amostra: Amostra, tamanho: tuple[int, int] | None = None) -> Image.Image:
    """Une anotacoes multiplas; uma amostra sem anotacao gera mascara binaria vazia."""
    if tamanho is None:
        with Image.open(amostra.imagem) as imagem:
            tamanho = imagem.size
    uniao = np.zeros((tamanho[1], tamanho[0]), dtype=np.uint8)
    for caminho in amostra.mascaras:
        with Image.open(caminho) as mascara:
            mascara = mascara.convert("L")
            if mascara.size != tamanho:
                mascara = mascara.resize(tamanho, Image.Resampling.NEAREST)
            uniao |= (np.asarray(mascara) > 0).astype(np.uint8)
    return Image.fromarray(uniao * 255, mode="L")


def validar_proporcoes_divisao(proporcao_treino: float, proporcao_validacao: float | None = None) -> None:
    if not 0 < proporcao_treino < 1:
        raise ValueError("proporcao_treino deve estar entre 0 e 1")
    if proporcao_validacao is not None:
        if not 0 < proporcao_validacao < 1:
            raise ValueError("proporcao_validacao deve estar entre 0 e 1")
        if abs((proporcao_treino + proporcao_validacao) - 1.0) > 1e-9:
            raise ValueError("proporcao_treino + proporcao_validacao deve ser igual a 1")


def dividir_originais(
    amostras: Sequence[Amostra],
    proporcao_treino: float,
    semente: int,
    classes: Sequence[str] | None = None,
) -> tuple[list[Amostra], list[Amostra]]:
    """Divisao estratificada e deterministica por classe."""
    validar_proporcoes_divisao(proporcao_treino)
    classes = tuple(classes) if classes is not None else tuple(sorted({a.classe for a in amostras}))
    rng = Random(semente)
    treino: list[Amostra] = []
    validacao: list[Amostra] = []
    for classe in classes:
        grupo = sorted(
            (a for a in amostras if a.classe == classe and not a.sintetica),
            key=lambda a: a.id,
        )
        if not grupo:
            raise ValueError(
                f"A classe {classe!r} nao possui amostras para a divisao. "
                "Confira dados.classes, aliases e a estrutura do dataset."
            )
        if len(grupo) == 1:
            raise ValueError(
                f"A classe {classe!r} possui apenas 1 amostra e nao pode ser "
                "dividida entre duas particoes. Adicione mais amostras ou ajuste "
                "a composicao do dataset."
            )
        rng.shuffle(grupo)
        quantidade_treino = round(len(grupo) * proporcao_treino)
        # Para classes pequenas, garante ao menos uma amostra em cada lado.
        quantidade_treino = max(1, min(len(grupo) - 1, quantidade_treino))
        treino.extend(grupo[:quantidade_treino])
        validacao.extend(grupo[quantidade_treino:])
    return sorted(treino, key=lambda a: a.id), sorted(validacao, key=lambda a: a.id)


def garantir_estrutura(raiz_projeto: str | Path, classes: Sequence[str]) -> None:
    raiz = Path(raiz_projeto)
    for conjunto in ("original", "aumentado", "teste"):
        for classe in classes:
            (raiz / conjunto / classe).mkdir(parents=True, exist_ok=True)
    for pasta in ("modelos", "modelos_difusao", "resultados"):
        (raiz / pasta).mkdir(parents=True, exist_ok=True)


def _copiar_membro_zip(arquivo: zipfile.ZipFile, membro: zipfile.ZipInfo, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    with arquivo.open(membro) as origem, destino.open("wb") as saida:
        shutil.copyfileobj(origem, saida)


def _classe_membro_zip(partes: Sequence[str], mapa_classes: Mapping[str, str]) -> str | None:
    for parte in partes[:-1]:
        classe = mapa_classes.get(parte.casefold())
        if classe is not None:
            return classe
    return None


def importar_zip_dataset(
    arquivo_zip: str | Path,
    destino: str | Path,
    classes: Sequence[str],
    mapa_classes: Mapping[str, str],
    sobrescrever: bool = False,
) -> dict[str, int]:
    """Importa um ZIP cujas imagens ficam em pastas de classe configuradas.

    Os nomes internos das classes e aliases aceitos sao definidos em `dados` no YAML.
    Um diretorio raiz adicional dentro do ZIP e permitido.
    """
    arquivo_zip = Path(arquivo_zip).expanduser().resolve()
    if not arquivo_zip.exists():
        raise FileNotFoundError(f"ZIP nao encontrado: {arquivo_zip}")

    destino = Path(destino)
    contagem = {classe: 0 for classe in classes}
    with zipfile.ZipFile(arquivo_zip) as arquivo:
        for membro in arquivo.infolist():
            if membro.is_dir():
                continue
            partes = Path(membro.filename).parts
            classe = _classe_membro_zip(partes, mapa_classes)
            if classe is None:
                continue
            nome = Path(membro.filename).name
            if Path(nome).suffix.lower() not in EXTENSOES:
                continue
            saida = destino / classe / nome
            if saida.exists() and not sobrescrever:
                continue
            _copiar_membro_zip(arquivo, membro, saida)
            contagem[classe] += 1

    if sum(contagem.values()) == 0:
        aliases = ", ".join(sorted(mapa_classes))
        raise ValueError(
            "Nenhuma imagem compativel foi encontrada no ZIP. "
            f"As pastas reconhecidas pela configuracao atual sao: {aliases}."
        )
    return contagem


def copiar_originais_para_aumentado(
    original: str | Path,
    aumentado: str | Path,
    classes: Sequence[str],
    sobrescrever: bool = False,
) -> int:
    original, aumentado = Path(original), Path(aumentado)
    copiados = 0
    for classe in classes:
        pasta_origem = original / classe
        (aumentado / classe).mkdir(parents=True, exist_ok=True)
        if not pasta_origem.exists():
            continue
        for origem in pasta_origem.iterdir():
            if not origem.is_file():
                continue
            destino = aumentado / classe / origem.name
            if destino.exists() and not sobrescrever:
                continue
            shutil.copy2(origem, destino)
            copiados += 1
    return copiados


def contar_por_classe(amostras: Iterable[Amostra], classes: Sequence[str] | None = None) -> dict[str, int]:
    amostras = list(amostras)
    classes = tuple(classes) if classes is not None else tuple(sorted({a.classe for a in amostras}))
    resultado = {classe: 0 for classe in classes}
    for amostra in amostras:
        resultado.setdefault(amostra.classe, 0)
        resultado[amostra.classe] += 1
    return resultado


def _copiar_amostra(amostra: Amostra, destino_classe: Path, sobrescrever: bool) -> int:
    destino_classe.mkdir(parents=True, exist_ok=True)
    copiados = 0
    for origem in (amostra.imagem, *amostra.mascaras):
        destino = destino_classe / origem.name
        if destino.exists() and not sobrescrever:
            continue
        shutil.copy2(origem, destino)
        copiados += 1
    return copiados


def salvar_manifesto_divisao(
    treino: Sequence[Amostra],
    validacao: Sequence[Amostra],
    destino: str | Path,
    teste: Sequence[Amostra] = (),
    raiz_referencia: str | Path | None = None,
) -> Path:
    """Salva as particoes do experimento sem copiar as imagens."""
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    raiz = Path(raiz_referencia).expanduser().resolve() if raiz_referencia else None

    def portatil(caminho: Path) -> str:
        resolvido = caminho.expanduser().resolve()
        if raiz is not None:
            try:
                return resolvido.relative_to(raiz).as_posix()
            except ValueError:
                pass
        return resolvido.name

    with destino.open("w", encoding="utf-8", newline="") as arquivo:
        campos = ["particao", "classe", "id", "imagem", "mascaras"]
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        particoes = (("treino", treino), ("validacao", validacao), ("teste", teste))
        for particao, amostras in particoes:
            for amostra in amostras:
                escritor.writerow(
                    {
                        "particao": particao,
                        "classe": amostra.classe,
                        "id": amostra.id,
                        "imagem": portatil(amostra.imagem),
                        "mascaras": ";".join(portatil(m) for m in amostra.mascaras),
                    }
                )
    return destino



def salvar_manifesto_desenvolvimento_teste(
    desenvolvimento: Sequence[Amostra],
    teste: Sequence[Amostra],
    destino: str | Path,
    raiz_referencia: str | Path | None = None,
) -> Path:
    """Salva a separacao fixa entre desenvolvimento e teste externo.

    O conjunto de desenvolvimento sera posteriormente particionado por
    StratifiedKFold durante a validacao cruzada; o teste permanece intocado.
    """
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    raiz = Path(raiz_referencia).expanduser().resolve() if raiz_referencia else None

    def portatil(caminho: Path) -> str:
        resolvido = caminho.expanduser().resolve()
        if raiz is not None:
            try:
                return resolvido.relative_to(raiz).as_posix()
            except ValueError:
                pass
        return resolvido.name

    with destino.open("w", encoding="utf-8", newline="") as arquivo:
        campos = ["particao", "classe", "id", "imagem", "mascaras"]
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        for particao, amostras in (("desenvolvimento", desenvolvimento), ("teste", teste)):
            for amostra in amostras:
                escritor.writerow(
                    {
                        "particao": particao,
                        "classe": amostra.classe,
                        "id": amostra.id,
                        "imagem": portatil(amostra.imagem),
                        "mascaras": ";".join(portatil(m) for m in amostra.mascaras),
                    }
                )
    return destino

def materializar_divisao(
    treino: Sequence[Amostra],
    validacao: Sequence[Amostra],
    destino: str | Path,
    sobrescrever: bool = False,
    limpar_destino: bool = False,
    classes: Sequence[str] | None = None,
) -> dict[str, object]:
    """Copia a divisao para `treino/` e `validacao/`, preservando imagens e mascaras."""
    destino = Path(destino)
    if limpar_destino:
        for nome in ("treino", "validacao"):
            shutil.rmtree(destino / nome, ignore_errors=True)
    elif not sobrescrever:
        existentes = [
            caminho
            for nome in ("treino", "validacao")
            for caminho in (destino / nome).rglob("*")
            if caminho.is_file() and caminho.name != ".gitkeep"
        ]
        if existentes:
            raise FileExistsError(
                f"O destino ja contem uma divisao: {destino}. "
                "Use --sobrescrever para substitui-la."
            )

    copiados = {"treino": 0, "validacao": 0}
    for particao, amostras in (("treino", treino), ("validacao", validacao)):
        for amostra in amostras:
            copiados[particao] += _copiar_amostra(
                amostra,
                destino / particao / amostra.classe,
                sobrescrever=sobrescrever,
            )

    manifesto = salvar_manifesto_divisao(
        treino,
        validacao,
        destino / "manifesto_divisao.csv",
        raiz_referencia=destino,
    )
    return {
        "destino": str(destino.resolve()),
        "amostras_treino": len(treino),
        "amostras_validacao": len(validacao),
        "por_classe_treino": contar_por_classe(treino, classes),
        "por_classe_validacao": contar_por_classe(validacao, classes),
        "arquivos_copiados": copiados,
        "manifesto": str(manifesto.resolve()),
    }


def _conjunto_possui_arquivos(destino: str | Path) -> bool:
    destino = Path(destino)
    return destino.exists() and any(
        caminho.is_file() and caminho.name != ".gitkeep" for caminho in destino.rglob("*")
    )


def limpar_conjunto_dataset(destino: str | Path, classes: Sequence[str]) -> None:
    """Remove dados existentes de um conjunto e recria as pastas de classe."""
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    for caminho in destino.iterdir():
        if caminho.is_dir():
            shutil.rmtree(caminho)
        else:
            caminho.unlink()
    for classe in classes:
        (destino / classe).mkdir(parents=True, exist_ok=True)


def importar_zip_unico_com_divisao(
    arquivo_zip: str | Path,
    destino_desenvolvimento: str | Path,
    destino_teste: str | Path,
    proporcao_teste: float,
    semente: int,
    classes: Sequence[str],
    mapa_classes: Mapping[str, str],
    classes_sem_mascara: Iterable[str] = (),
    sobrescrever: bool = False,
) -> dict[str, object]:
    """Divide um unico ZIP em desenvolvimento/teste e materializa os dois conjuntos."""
    if not 0 < proporcao_teste < 1:
        raise ValueError("proporcao_teste deve estar entre 0 e 1")

    destino_desenvolvimento = Path(destino_desenvolvimento)
    destino_teste = Path(destino_teste)

    if not sobrescrever and (
        _conjunto_possui_arquivos(destino_desenvolvimento)
        or _conjunto_possui_arquivos(destino_teste)
    ):
        raise FileExistsError(
            "Os destinos original/ ou teste/ ja contem dados. "
            "Use --sobrescrever para refazer a divisao a partir do ZIP unico."
        )

    if sobrescrever:
        limpar_conjunto_dataset(destino_desenvolvimento, classes)
        limpar_conjunto_dataset(destino_teste, classes)

    with tempfile.TemporaryDirectory(prefix="dataset_prepare_") as temporario:
        raiz_temporaria = Path(temporario)
        importar_zip_dataset(
            arquivo_zip,
            raiz_temporaria,
            classes=classes,
            mapa_classes=mapa_classes,
            sobrescrever=True,
        )
        amostras = coletar_amostras(
            raiz_temporaria,
            classes=classes,
            incluir_sinteticas=False,
            classes_sem_mascara=classes_sem_mascara,
        )
        desenvolvimento, teste = dividir_originais(
            amostras,
            proporcao_treino=1.0 - proporcao_teste,
            semente=semente,
            classes=classes,
        )

        arquivos_desenvolvimento = sum(
            _copiar_amostra(a, destino_desenvolvimento / a.classe, sobrescrever=True)
            for a in desenvolvimento
        )
        arquivos_teste = sum(
            _copiar_amostra(a, destino_teste / a.classe, sobrescrever=True)
            for a in teste
        )

    return {
        "amostras_desenvolvimento": len(desenvolvimento),
        "amostras_teste": len(teste),
        "por_classe_desenvolvimento": contar_por_classe(desenvolvimento, classes),
        "por_classe_teste": contar_por_classe(teste, classes),
        "arquivos_copiados_desenvolvimento": arquivos_desenvolvimento,
        "arquivos_copiados_teste": arquivos_teste,
    }
