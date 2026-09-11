from __future__ import annotations

import csv
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from main import preparar
from src.configuracao import caminho_configurado, carregar_configuracao
from src.dados import (
    Amostra,
    carregar_mascara_unificada,
    classes_configuradas,
    classes_sem_mascara_configuradas,
    coletar_amostras,
    dividir_originais,
    identificar_sintetica,
    mapa_classes_configurado,
)
from src.metricas import metricas_binarias


CLASSES_TESTE = ("classe_a", "classe_b", "controle")
PASTAS_ZIP = {"alpha": "classe_a", "beta": "classe_b", "healthy": "controle"}


class TestDados(unittest.TestCase):
    def _criar_zip(self, raiz: Path, nome: str, quantidade_por_classe: int, prefixo: str) -> Path:
        origem = raiz / f"origem_{prefixo}"
        for pasta, classe_interna in PASTAS_ZIP.items():
            (origem / pasta).mkdir(parents=True, exist_ok=True)
            for i in range(quantidade_por_classe):
                base = f"{prefixo}_{pasta}_{i}"
                Image.new("L", (8, 8), 100).save(origem / pasta / f"{base}.png")
                if classe_interna != "controle":
                    Image.new("L", (8, 8), 255).save(origem / pasta / f"{base}_mask.png")

        zip_path = raiz / nome
        with zipfile.ZipFile(zip_path, "w") as arquivo:
            for caminho in origem.rglob("*.png"):
                arquivo.write(caminho, caminho.relative_to(origem))
        return zip_path

    def _config(self, raiz: Path, proporcao_teste: float, proporcao_treino: float) -> dict:
        return {
            "_diretorio_base": str(raiz),
            "experimento": {"semente": 12345},
            "dados": {
                "classes": list(CLASSES_TESTE),
                "aliases": PASTAS_ZIP,
                "classes_sem_mascara": ["controle"],
            },
            "caminhos": {
                "original": "original",
                "aumentado": "aumentado",
                "teste": "teste",
                "modelos": "modelos",
                "resultados": "resultados",
            },
            "fontes": {"zip_dataset": None, "zip_teste": None},
            "divisao": {
                "proporcao_teste": proporcao_teste,
                "proporcao_treino": proporcao_treino,
                "proporcao_validacao": 1.0 - proporcao_treino,
            },
        }

    def test_classes_e_aliases_sao_configuraveis(self):
        config = self._config(Path("."), 0.2, 0.8)
        self.assertEqual(classes_configuradas(config), CLASSES_TESTE)
        self.assertEqual(classes_sem_mascara_configuradas(config), frozenset({"controle"}))
        mapa = mapa_classes_configurado(config)
        self.assertEqual(mapa["alpha"], "classe_a")
        self.assertEqual(mapa["classe_b"], "classe_b")

    def test_config_em_subpasta_pode_definir_raiz_do_projeto(self):
        with tempfile.TemporaryDirectory() as pasta:
            raiz = Path(pasta)
            configs = raiz / "configs"
            configs.mkdir()
            arquivo = configs / "teste.yaml"
            arquivo.write_text(
                "projeto:\n  raiz: ..\ncaminhos:\n  original: original\n",
                encoding="utf-8",
            )
            config = carregar_configuracao(arquivo)
            self.assertEqual(caminho_configurado(config, "original"), raiz / "original")

    def test_identifica_origem_sintetica(self):
        sintetica, origem = identificar_sintetica(Path("caso_1__sint_02.png"))
        self.assertTrue(sintetica)
        self.assertEqual(origem, "caso_1")

    def test_uniao_de_multiplas_mascaras(self):
        with tempfile.TemporaryDirectory() as pasta:
            raiz = Path(pasta)
            imagem = raiz / "caso.png"
            Image.new("L", (4, 4), 0).save(imagem)
            m1 = np.zeros((4, 4), dtype=np.uint8)
            m2 = np.zeros((4, 4), dtype=np.uint8)
            m1[0, 0] = 255
            m2[3, 3] = 255
            p1, p2 = raiz / "caso_mask.png", raiz / "caso_mask_1.png"
            Image.fromarray(m1).save(p1)
            Image.fromarray(m2).save(p2)
            amostra = Amostra(imagem, "classe_a", (p1, p2))
            uniao = np.asarray(carregar_mascara_unificada(amostra)) > 0
            self.assertEqual(int(uniao.sum()), 2)

    def test_divisao_estratificada_reproduzivel_com_classes_genericas(self):
        amostras = [
            Amostra(Path(f"{classe}_{i}.png"), classe, ())
            for classe in CLASSES_TESTE
            for i in range(10)
        ]
        treino_a, validacao_a = dividir_originais(amostras, 0.8, 12345, classes=CLASSES_TESTE)
        treino_b, validacao_b = dividir_originais(amostras, 0.8, 12345, classes=CLASSES_TESTE)
        self.assertEqual(len(treino_a), 24)
        self.assertEqual(len(validacao_a), 6)
        self.assertEqual([a.id for a in treino_a], [a.id for a in treino_b])
        self.assertEqual([a.id for a in validacao_a], [a.id for a in validacao_b])
        for classe in CLASSES_TESTE:
            self.assertEqual(sum(a.classe == classe for a in treino_a), 8)
            self.assertEqual(sum(a.classe == classe for a in validacao_a), 2)

    def test_divisao_garante_amostra_em_cada_particao_para_classe_pequena(self):
        amostras = [
            Amostra(Path("classe_a_0.png"), "classe_a", ()),
            Amostra(Path("classe_a_1.png"), "classe_a", ()),
        ]
        treino, validacao = dividir_originais(amostras, 0.9, 12345, classes=("classe_a",))
        self.assertEqual(len(treino), 1)
        self.assertEqual(len(validacao), 1)

    def test_divisao_rejeita_classe_com_uma_unica_amostra(self):
        amostras = [Amostra(Path("unica.png"), "classe_a", ())]
        with self.assertRaisesRegex(ValueError, "apenas 1 amostra"):
            dividir_originais(amostras, 0.8, 12345, classes=("classe_a",))

    def test_divisao_rejeita_classe_configurada_sem_amostras(self):
        amostras = [
            Amostra(Path("a_0.png"), "classe_a", ()),
            Amostra(Path("a_1.png"), "classe_a", ()),
        ]
        with self.assertRaisesRegex(ValueError, "nao possui amostras"):
            dividir_originais(amostras, 0.8, 12345, classes=("classe_a", "classe_b"))

    def test_classe_configurada_sem_mascara_gera_alvo_vazio(self):
        with tempfile.TemporaryDirectory() as pasta:
            raiz = Path(pasta)
            classe = raiz / "controle"
            classe.mkdir()
            Image.new("L", (4, 4), 100).save(classe / "negativo.png")
            amostras = coletar_amostras(
                raiz,
                classes=CLASSES_TESTE,
                classes_sem_mascara={"controle"},
            )
            self.assertEqual(len(amostras), 1)
            self.assertFalse(np.asarray(carregar_mascara_unificada(amostras[0])).any())

    def test_preparar_um_zip_divide_desenvolvimento_teste_e_treino_validacao(self):
        with tempfile.TemporaryDirectory() as pasta:
            raiz = Path(pasta)
            dataset_zip = self._criar_zip(raiz, "dataset.zip", 10, "unico")
            config = self._config(raiz, proporcao_teste=0.2, proporcao_treino=0.75)
            args = SimpleNamespace(
                dataset_zip=str(dataset_zip),
                teste_zip=None,
                sobrescrever=False,
                exportar_divisao=None,
            )

            resultado = preparar(args, config)
            desenvolvimento = coletar_amostras(
                raiz / "original",
                classes=CLASSES_TESTE,
                incluir_sinteticas=False,
                classes_sem_mascara={"controle"},
            )
            teste = coletar_amostras(
                raiz / "teste",
                classes=CLASSES_TESTE,
                incluir_sinteticas=False,
                classes_sem_mascara={"controle"},
            )

            self.assertEqual(resultado["modo_preparacao"], "zip_unico")
            self.assertEqual(len(desenvolvimento), 24)
            self.assertEqual(len(teste), 6)
            self.assertEqual(resultado["divisao"]["treino"], 18)
            self.assertEqual(resultado["divisao"]["validacao"], 6)
            self.assertEqual(resultado["divisao"]["teste"], 6)
            self.assertTrue({a.id for a in desenvolvimento}.isdisjoint({a.id for a in teste}))
            self.assertTrue(all(a.classe in CLASSES_TESTE for a in desenvolvimento + teste))

            manifesto = raiz / "resultados" / "divisao_preparacao.csv"
            with manifesto.open(encoding="utf-8", newline="") as arquivo:
                linhas = list(csv.DictReader(arquivo))
            self.assertEqual(len(linhas), 30)
            self.assertEqual(sum(l["particao"] == "treino" for l in linhas), 18)
            self.assertEqual(sum(l["particao"] == "validacao" for l in linhas), 6)
            self.assertEqual(sum(l["particao"] == "teste" for l in linhas), 6)
            self.assertTrue(all(not Path(l["imagem"]).is_absolute() for l in linhas))
            self.assertTrue(all(not Path(m).is_absolute() for l in linhas for m in l["mascaras"].split(";") if m))

    def test_preparar_dois_zips_preserva_teste_externo(self):
        with tempfile.TemporaryDirectory() as pasta:
            raiz = Path(pasta)
            desenvolvimento_zip = self._criar_zip(raiz, "desenvolvimento.zip", 6, "dev")
            teste_zip = self._criar_zip(raiz, "teste.zip", 2, "externo")
            config = self._config(raiz, proporcao_teste=0.2, proporcao_treino=0.5)
            args = SimpleNamespace(
                dataset_zip=str(desenvolvimento_zip),
                teste_zip=str(teste_zip),
                sobrescrever=False,
                exportar_divisao=None,
            )

            resultado = preparar(args, config)
            desenvolvimento = coletar_amostras(
                raiz / "original",
                classes=CLASSES_TESTE,
                incluir_sinteticas=False,
                classes_sem_mascara={"controle"},
            )
            teste = coletar_amostras(
                raiz / "teste",
                classes=CLASSES_TESTE,
                incluir_sinteticas=False,
                classes_sem_mascara={"controle"},
            )

            self.assertEqual(resultado["modo_preparacao"], "zips_separados")
            self.assertEqual(len(desenvolvimento), 18)
            self.assertEqual(len(teste), 6)
            self.assertEqual(resultado["divisao"]["treino"], 9)
            self.assertEqual(resultado["divisao"]["validacao"], 9)
            self.assertEqual(resultado["divisao"]["teste"], 6)
            self.assertTrue(all(a.imagem.stem.startswith("externo_") for a in teste))


class TestMetricas(unittest.TestCase):
    def test_predicao_perfeita(self):
        alvo = np.array([[0, 1], [0, 1]], dtype=bool)
        metricas = metricas_binarias(alvo, alvo)
        self.assertEqual(metricas["iou"], 1.0)
        self.assertEqual(metricas["f1"], 1.0)

    def test_duas_mascaras_vazias_sao_acerto(self):
        vazio = np.zeros((3, 3), dtype=bool)
        metricas = metricas_binarias(vazio, vazio)
        self.assertEqual(metricas["iou"], 1.0)
        self.assertEqual(metricas["f1"], 1.0)


class TestAvaliacaoGenerica(unittest.TestCase):
    def test_macro_regiao_presente_nao_depende_do_nome_da_classe(self):
        from src.avaliacao import _resumir_modelo

        base = {
            "original_tp": 1, "original_tn": 3, "original_fp": 0, "original_fn": 0,
            "original_iou": 1.0, "original_f1": 1.0, "original_dice": 1.0,
            "original_precisao": 1.0, "original_revocacao": 1.0,
            "original_especificidade": 1.0, "original_acuracia": 1.0,
        }
        linhas = [
            {"classe": "qualquer_nome", "tem_regiao": True, **base},
            {"classe": "outra_classe", "tem_regiao": False, **base},
        ]
        resumo = _resumir_modelo("original", linhas)
        por_escopo = {linha["escopo"]: linha for linha in resumo}
        self.assertEqual(por_escopo["macro_regiao_presente"]["n"], 1)
        self.assertEqual(por_escopo["macro_todas"]["n"], 2)


if __name__ == "__main__":
    unittest.main()
