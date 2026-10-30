import io
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import api


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.cliente = api.app.test_client()
        imagem = Image.new("RGB", (40, 40), "white")
        saida = io.BytesIO()
        imagem.save(saida, format="PNG")
        self.imagem_bytes = saida.getvalue()

    def post(self, rota="/contar", **campos):
        dados = {"imagem": (io.BytesIO(self.imagem_bytes), "foto.png"), **campos}
        return self.cliente.post(rota, data=dados, content_type="multipart/form-data")

    @staticmethod
    def mascara_falsa(imagem, objeto, parametros):
        largura, altura = imagem.size
        mascara = np.zeros((altura, largura), dtype=bool)
        mascara[5:15, 5:15] = True
        mascara[25:35, 25:35] = True
        return mascara

    def test_counts_objects_from_workers_mask(self):
        with patch("api.processar_distribuido", side_effect=self.mascara_falsa):
            resposta = self.post(objeto="caixa de leite")
        self.assertEqual(resposta.status_code, 200)
        corpo = resposta.get_json()
        self.assertEqual(corpo["quantidade"], 2)
        self.assertEqual(corpo["objeto_en"], "milk carton")
        self.assertEqual(len(corpo["instancias"]), 2)
        self.assertEqual(corpo["imagem_original"], {"largura": 40, "altura": 40})
        self.assertNotIn("figura_png_base64", corpo)

    def test_figure_route_returns_png(self):
        with patch("api.processar_distribuido", side_effect=self.mascara_falsa):
            resposta = self.post("/contar/figura", objeto="caixa de leite")
        self.assertEqual(resposta.status_code, 200)
        self.assertEqual(resposta.mimetype, "image/png")
        self.assertEqual(resposta.headers["X-Quantidade"], "2")

    def test_missing_object_is_rejected(self):
        resposta = self.post()
        self.assertEqual(resposta.status_code, 400)

    def test_worker_failure_returns_503(self):
        with patch("api.processar_distribuido", side_effect=RuntimeError("Falha ao processar tiles: [1]")):
            resposta = self.post(objeto="garrafa")
        self.assertEqual(resposta.status_code, 503)
        self.assertIn("Falha ao processar tiles", resposta.get_json()["erro"])


if __name__ == "__main__":
    unittest.main()
