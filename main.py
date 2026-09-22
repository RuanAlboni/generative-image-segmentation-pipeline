from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.configuracao import caminho_configurado, caminho_do_projeto, carregar_configuracao
from src.dados import (
    classes_configuradas,
    classes_sem_mascara_configuradas,
    mapa_classes_configurado,
    coletar_amostras,
    copiar_originais_para_aumentado,
    dividir_originais,
    garantir_estrutura,
    importar_zip_dataset,
    importar_zip_unico_com_divisao,
    limpar_conjunto_dataset,
    materializar_divisao,
    salvar_manifesto_desenvolvimento_teste,
    validar_proporcoes_divisao,
)


def _imprimir(resultado: Any) -> None:
    print(json.dumps(resultado, ensure_ascii=False, indent=2, default=str))


def preparar(args, config: dict[str, Any]) -> dict[str, Any]:
    raiz = Path(config.get("_raiz_projeto", config["_diretorio_base"]))
    classes = classes_configuradas(config)
    classes_sem_mascara = classes_sem_mascara_configuradas(config)
    mapa_classes = mapa_classes_configurado(config)
    garantir_estrutura(raiz, classes)

    fontes = config.get("fontes", {})
    dataset_zip = args.dataset_zip or fontes.get("zip_dataset")
    teste_zip = args.teste_zip or fontes.get("zip_teste")

    resultado: dict[str, Any] = {"estrutura": "criada"}
    semente = int(config["experimento"]["semente"])

    if args.sobrescrever and (dataset_zip or teste_zip):
        limpar_conjunto_dataset(caminho_configurado(config, "aumentado"), classes)
        if dataset_zip and teste_zip:
            limpar_conjunto_dataset(caminho_configurado(config, "original"), classes)
            limpar_conjunto_dataset(caminho_configurado(config, "teste"), classes)
        elif teste_zip and not dataset_zip:
            limpar_conjunto_dataset(caminho_configurado(config, "teste"), classes)

    if dataset_zip and teste_zip:
        resultado["modo_preparacao"] = "zips_separados"
        resultado["itens_desenvolvimento_importados"] = importar_zip_dataset(
            dataset_zip,
            caminho_configurado(config, "original"),
            classes=classes,
            mapa_classes=mapa_classes,
            sobrescrever=args.sobrescrever,
        )
        resultado["itens_teste_importados"] = importar_zip_dataset(
            teste_zip,
            caminho_configurado(config, "teste"),
            classes=classes,
            mapa_classes=mapa_classes,
            sobrescrever=args.sobrescrever,
        )
    elif dataset_zip:
        resultado["modo_preparacao"] = "zip_unico"
        proporcao_teste = float(config["divisao"].get("proporcao_teste", 0.15))
        resultado["divisao_desenvolvimento_teste"] = importar_zip_unico_com_divisao(
            dataset_zip,
            caminho_configurado(config, "original"),
            caminho_configurado(config, "teste"),
            proporcao_teste=proporcao_teste,
            semente=semente,
            classes=classes,
            mapa_classes=mapa_classes,
            classes_sem_mascara=classes_sem_mascara,
            sobrescrever=args.sobrescrever,
        )
    elif teste_zip:
        resultado["modo_preparacao"] = "teste_separado_com_desenvolvimento_existente"
        resultado["itens_teste_importados"] = importar_zip_dataset(
            teste_zip,
            caminho_configurado(config, "teste"),
            classes=classes,
            mapa_classes=mapa_classes,
            sobrescrever=args.sobrescrever,
        )
    else:
        resultado["modo_preparacao"] = "dados_existentes"

    resultado["itens_copiados_para_aumentado"] = copiar_originais_para_aumentado(
        caminho_configurado(config, "original"),
        caminho_configurado(config, "aumentado"),
        classes=classes,
        sobrescrever=args.sobrescrever,
    )

    originais = coletar_amostras(
        caminho_configurado(config, "original"),
        classes=classes,
        incluir_sinteticas=False,
        classes_sem_mascara=classes_sem_mascara,
    )
    if originais:
        teste = coletar_amostras(
            caminho_configurado(config, "teste"),
            classes=classes,
            incluir_sinteticas=False,
            classes_sem_mascara=classes_sem_mascara,
        )
        manifesto = salvar_manifesto_desenvolvimento_teste(
            originais,
            teste,
            caminho_configurado(config, "resultados") / "divisao_preparacao.csv",
            raiz_referencia=raiz,
        )
        resultado["divisao"] = {
            "desenvolvimento": len(originais),
            "teste": len(teste),
            "manifesto": str(manifesto),
        }

        # Opcional: materializa uma unica divisao apenas para depuracao rapida.
        # A validacao experimental principal e feita por StratifiedKFold no treino.
        if args.exportar_divisao:
            proporcao_treino = float(config["divisao"].get("proporcao_treino", 0.8))
            proporcao_validacao = float(config["divisao"].get("proporcao_validacao", 1.0 - proporcao_treino))
            validar_proporcoes_divisao(proporcao_treino, proporcao_validacao)
            treino, validacao = dividir_originais(originais, proporcao_treino, semente, classes=classes)
            destino = caminho_do_projeto(config, args.exportar_divisao)
            resultado["divisao_materializada"] = materializar_divisao(
                treino,
                validacao,
                destino,
                sobrescrever=args.sobrescrever,
                limpar_destino=args.sobrescrever,
                classes=classes,
            )

    return resultado


def validar(config: dict[str, Any]) -> dict[str, Any]:
    classes = classes_configuradas(config)
    classes_sem_mascara = classes_sem_mascara_configuradas(config)
    resultado: dict[str, Any] = {}
    for conjunto in ("original", "aumentado", "teste"):
        raiz = caminho_configurado(config, conjunto)
        amostras = coletar_amostras(
            raiz,
            classes=classes,
            exigir_mascara=conjunto != "aumentado",
            classes_sem_mascara=classes_sem_mascara,
        )
        resultado[conjunto] = {
            "total_imagens": len(amostras),
            "originais": sum(not a.sintetica for a in amostras),
            "sinteticas": sum(a.sintetica for a in amostras),
            "por_classe": {
                classe: sum(a.classe == classe for a in amostras)
                for classe in classes
            },
            "sem_mascara": sum(not a.mascaras for a in amostras),
        }
    return resultado


def construir_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Pipeline experimental para comparar segmentacao treinada somente com "
            "imagens originais contra treinamento com aumento generativo."
        )
    )
    parser.add_argument("--config", default="config.yaml", help="Arquivo YAML de configuracao.")
    sub = parser.add_subparsers(dest="comando", required=True)

    p_preparar = sub.add_parser(
        "preparar",
        help=(
            "Prepara os dados: aceita desenvolvimento+teste separados ou divide "
            "automaticamente um unico ZIP em desenvolvimento e teste externo fixo."
        ),
    )
    p_preparar.add_argument(
        "--dataset-zip",
        help=(
            "ZIP do dataset. Sem --teste-zip, ele e dividido automaticamente em "
            "desenvolvimento e teste; com --teste-zip, e tratado como desenvolvimento."
        ),
    )
    p_preparar.add_argument(
        "--teste-zip",
        help="ZIP opcional reservado exclusivamente para avaliacao final.",
    )
    p_preparar.add_argument(
        "--exportar-divisao",
        nargs="?",
        const="particoes",
        metavar="PASTA",
        help=(
            "Materializa uma divisao treino/validacao simples apenas para depuracao. "
            "A validacao principal usa StratifiedKFold. Sem PASTA, usa ./particoes."
        ),
    )
    p_preparar.add_argument("--sobrescrever", action="store_true")

    sub.add_parser("validar", help="Confere quantidades e mascaras da estrutura de dados.")

    p_difusao = sub.add_parser("difusao", help="Gera imagens sinteticas por img2img.")
    p_difusao.add_argument("--limite", type=int, help="Limite total de origens para teste rapido.")
    p_difusao.add_argument("--simulacao", action="store_true", help="Lista elegiveis sem carregar o modelo.")

    p_masc = sub.add_parser(
        "mascaras",
        help="Cria mascaras das sinteticas por watershed guiado pela mascara original.",
    )
    p_masc.add_argument("--limite", type=int, help="Limita a quantidade de sinteticas processadas.")
    p_masc.add_argument(
        "--debug",
        action="store_true",
        help="Salva um painel e imagens de cada etapa do watershed para inspecao visual.",
    )

    p_treinar = sub.add_parser(
        "treinar",
        help="Executa StratifiedKFold e treina os modelos finais; use --modo-simples para depuracao.",
    )
    p_treinar.add_argument(
        "--modo-simples",
        action="store_true",
        help="Usa uma unica divisao treino/validacao, sem validacao cruzada.",
    )

    p_avaliar = sub.add_parser("avaliar", help="Avalia os dois modelos no conjunto de teste externo.")
    p_avaliar.add_argument("--limite", type=int)

    p_pipeline = sub.add_parser("pipeline", help="Executa difusao, mascaras, treino e avaliacao.")
    p_pipeline.add_argument("--limite-geracao", type=int)
    p_pipeline.add_argument("--limite-avaliacao", type=int)
    p_pipeline.add_argument(
        "--modo-simples",
        action="store_true",
        help="Executa treinamento com uma unica divisao em vez do StratifiedKFold.",
    )
    return parser


def main() -> None:
    parser = construir_parser()
    args = parser.parse_args()
    config = carregar_configuracao(args.config)

    if args.comando == "preparar":
        resultado = preparar(args, config)
    elif args.comando == "validar":
        resultado = validar(config)
    elif args.comando == "difusao":
        from src.difusao import gerar_imagens_sinteticas

        resultado = gerar_imagens_sinteticas(config, args.limite, args.simulacao)
    elif args.comando == "mascaras":
        from src.watershed import gerar_mascaras_watershed

        resultado = gerar_mascaras_watershed(config, args.limite, debug=args.debug)
    elif args.comando == "treinar":
        from src.treinamento import treinar_experimento

        resultado = treinar_experimento(config, modo_simples=args.modo_simples)
    elif args.comando == "avaliar":
        from src.avaliacao import avaliar_modelos

        resultado = avaliar_modelos(config, args.limite)
    else:
        from src.avaliacao import avaliar_modelos
        from src.difusao import gerar_imagens_sinteticas
        from src.treinamento import treinar_experimento
        from src.watershed import gerar_mascaras_watershed

        resultado = {
            "difusao": gerar_imagens_sinteticas(config, args.limite_geracao),
            "mascaras": gerar_mascaras_watershed(config, args.limite_geracao),
            "treinamento": treinar_experimento(config, modo_simples=args.modo_simples),
            "avaliacao": avaliar_modelos(config, args.limite_avaliacao),
        }
    _imprimir(resultado)


if __name__ == "__main__":
    main()
