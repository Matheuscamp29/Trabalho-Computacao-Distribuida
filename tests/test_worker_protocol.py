import asyncio
import importlib
import io
import sys
import types
import unittest
from unittest.mock import patch

import httpx
import numpy as np
from PIL import Image

from logical_clock import LamportClock


class WorkerProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Evita baixar o modelo CLIP: o teste cobre apenas o protocolo HTTP.
        pipeline = types.ModuleType("pipeline")
        pipeline.gerar_mascara_objetos = lambda *args, **kwargs: (np.ones((2, 2), dtype=bool), "box")
        classifier = types.ModuleType("classifier")
        classifier.ClassificadorCLIP = object
        with patch.dict(sys.modules, {"pipeline": pipeline, "classifier": classifier}):
            cls.worker = importlib.import_module("worker")

    def setUp(self):
        self.worker.CLOCK = LamportClock()
        self.worker.CLASSIFICADOR = object()
        image = Image.new("RGB", (2, 2), "white")
        output = io.BytesIO()
        image.save(output, format="PNG")
        self.image_bytes = output.getvalue()

    def test_response_carries_task_id_and_advanced_clock(self):
        async def request():
            transport = httpx.ASGITransport(app=self.worker.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://worker") as client:
                return await client.post(
                    "/process_tile",
                    data={"objeto": "box"},
                    files={"imagem": ("tile.png", self.image_bytes, "image/png")},
                    headers={"X-Task-Id": "request-0", "X-Lamport-Timestamp": "7"},
                )

        response = asyncio.run(request())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["X-Task-Id"], "request-0")
        self.assertEqual(int(response.headers["X-Lamport-Timestamp"]), 9)
        self.assertEqual(response.content, bytes([1] * 4))

    def test_missing_lamport_header_is_rejected(self):
        async def request():
            transport = httpx.ASGITransport(app=self.worker.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://worker") as client:
                return await client.post(
                    "/process_tile",
                    data={"objeto": "box"},
                    files={"imagem": ("tile.png", self.image_bytes, "image/png")},
                    headers={"X-Task-Id": "request-0"},
                )

        self.assertEqual(asyncio.run(request()).status_code, 422)


if __name__ == "__main__":
    unittest.main()
