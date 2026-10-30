import threading
import types
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

import logical_clock
from distributed_manager import processar_distribuido

WORKERS = ["http://worker1:8001", "http://worker2:8001"]


class DistributedManagerTests(unittest.TestCase):
    def setUp(self):
        logical_clock.valor = 0
        self.imagem = Image.new("RGB", (4, 4), "white")
        self.chamadas = []
        self.trava = threading.Lock()
        self.derrubar_worker2 = False

    def post_falso(self, url, data, files, headers, timeout):
        task_id = headers["X-Task-Id"]
        carimbo = int(headers["X-Lamport-Timestamp"])
        with self.trava:
            self.chamadas.append((url, task_id, carimbo))
        if self.derrubar_worker2 and "worker2" in url:
            return types.SimpleNamespace(status_code=503, content=b"", text="indisponivel", headers={})
        return types.SimpleNamespace(
            status_code=200, content=bytes([1] * 4), text="",
            headers={"X-Task-Id": task_id, "X-Lamport-Timestamp": str(carimbo + 2)},
        )

    def test_tiles_alternate_between_workers_and_clock_advances(self):
        with patch("distributed_manager.requests.post", side_effect=self.post_falso):
            mascara = processar_distribuido(self.imagem, "box", {}, WORKERS)

        por_indice = sorted(self.chamadas, key=lambda c: int(c[1].rsplit("-", 1)[1]))
        self.assertEqual([c[0] for c in por_indice],
                         [f"{w}/process_tile" for w in (WORKERS[0], WORKERS[1], WORKERS[0], WORKERS[1])])
        self.assertEqual(len({c[1] for c in por_indice}), 4)
        self.assertTrue(all(c[2] > 0 for c in por_indice))
        self.assertGreater(logical_clock.atual(), max(c[2] + 2 for c in por_indice))
        np.testing.assert_array_equal(mascara, np.ones((4, 4), dtype=bool))

    def test_failed_tile_does_not_return_partial_count(self):
        self.derrubar_worker2 = True
        with patch("distributed_manager.requests.post", side_effect=self.post_falso):
            with self.assertRaisesRegex(RuntimeError, "Falha ao processar tiles"):
                processar_distribuido(self.imagem, "box", {}, WORKERS)

    def test_requires_two_distinct_worker_addresses(self):
        with self.assertRaises(ValueError):
            processar_distribuido(self.imagem, "box", {}, ["http://worker1:8001"])


if __name__ == "__main__":
    unittest.main()
