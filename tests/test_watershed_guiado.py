from __future__ import annotations

import unittest

import numpy as np

from src.watershed import segmentar_watershed_guiado


class TestWatershedGuiado(unittest.TestCase):
    def test_marcadores_e_resultado_sao_coerentes(self):
        h = w = 128
        yy, xx = np.ogrid[:h, :w]
        centro = (64, 64)
        raio_origem = 20
        mascara = (((xx - centro[1]) ** 2 + (yy - centro[0]) ** 2) <= raio_origem**2)
        mascara = mascara.astype(np.uint8) * 255

        # Fundo claro e regiao escura, com transicao circular proxima da mascara.
        imagem = np.full((h, w), 180, dtype=np.uint8)
        raio_imagem = 23
        regiao = ((xx - centro[1]) ** 2 + (yy - centro[0]) ** 2) <= raio_imagem**2
        imagem[regiao] = 70

        cfg = {
            "dilatacao_externa_kernel": 5,
            "dilatacao_externa_iteracoes": 4,
            "gradiente_dilatacao_kernel": 3,
            "gradiente_imagem": "sobel",
            "sobel_kernel": 3,
            "linha_watershed": True,
        }
        resultado, passos, diagnostico = segmentar_watershed_guiado(imagem, mascara, cfg)

        self.assertGreater(np.count_nonzero(passos["marcador_interno"]), 0)
        self.assertGreater(np.count_nonzero(passos["marcador_externo"]), 0)
        self.assertEqual(
            np.count_nonzero(
                (passos["marcador_interno"] > 0) & (passos["marcador_externo"] > 0)
            ),
            0,
        )
        self.assertGreater(np.count_nonzero(resultado), 0)
        self.assertLess(np.count_nonzero(resultado), h * w)
        self.assertTrue(
            np.all(resultado[passos["marcador_interno"] > 0] > 0),
            "O marcador interno precisa permanecer dentro da bacia da regiao.",
        )
        self.assertEqual(diagnostico["gradiente_imagem"], "sobel")


if __name__ == "__main__":
    unittest.main()
