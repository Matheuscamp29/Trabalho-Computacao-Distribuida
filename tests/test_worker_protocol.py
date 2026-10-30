import importlib
import io
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from distribuido import logical_clock


class WorkerProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pipeline = types.ModuleType("visao.pipeline")
        pipeline.gerar_mascara = lambda *args, **kwargs: (np.ones((2, 2), dtype=bool), None, None, "box")
        classifier = types.ModuleType("visao.classifier")
        classifier.carregar_clip = lambda *args, **kwargs: {"nome": "falso", "device": "cpu"}
        with patch.dict(sys.modules, {"visao.pipeline": pipeline, "visao.classifier": classifier}):
            sys.modules.pop("distribuido.worker", None)
            cls.worker = importlib.import_module("distribuido.worker")
        cls.cliente = cls.worker.app.test_client()

    def setUp(self):
        logical_clock.valor = 0
        imagem = Image.new("RGB", (2, 2), "white")
        saida = io.BytesIO()
        imagem.save(saida, format="PNG")
        self.imagem_bytes = saida.getvalue()

    def enviar(self, headers):
        return self.cliente.post(
            "/process_tile",
            data={"objeto": "box", "imagem": (io.BytesIO(self.imagem_bytes), "tile.png")},
            headers=headers,
            content_type="multipart/form-data",
        )

    def test_response_carries_task_id_and_advanced_clock(self):
        resposta = self.enviar({"X-Task-Id": "request-0", "X-Lamport-Timestamp": "7"})
        self.assertEqual(resposta.status_code, 200)
        self.assertEqual(resposta.headers["X-Task-Id"], "request-0")
        self.assertEqual(int(resposta.headers["X-Lamport-Timestamp"]), 9)
        self.assertEqual(resposta.data, bytes([1] * 4))

    def test_missing_lamport_header_is_rejected(self):
        resposta = self.enviar({"X-Task-Id": "request-0"})
        self.assertEqual(resposta.status_code, 422)


if __name__ == "__main__":
    unittest.main()
