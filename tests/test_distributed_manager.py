import asyncio
import io
import unittest
from unittest.mock import patch

import httpx
import numpy as np
from PIL import Image

from distributed_manager import DistributedManager


class DistributedManagerTests(unittest.TestCase):
    def setUp(self):
        image = Image.new("RGB", (4, 4), "white")
        output = io.BytesIO()
        image.save(output, format="PNG")
        self.image_bytes = output.getvalue()
        self.manager = DistributedManager(["http://worker1:8001", "http://worker2:8001"])

    def test_tiles_are_sent_to_both_workers_and_assembled(self):
        requests = []

        def respond(request):
            task_id = request.headers["X-Task-Id"]
            timestamp = int(request.headers["X-Lamport-Timestamp"])
            requests.append((request.url.host, task_id, timestamp))
            return httpx.Response(
                200, content=bytes([1] * 4),
                headers={"X-Task-Id": task_id,
                         "X-Lamport-Timestamp": str(timestamp + 2)},
            )

        original_client = httpx.AsyncClient
        transport = httpx.MockTransport(respond)
        with patch("distributed_manager.httpx.AsyncClient",
                   side_effect=lambda **kwargs: original_client(transport=transport, **kwargs)):
            mask = asyncio.run(self.manager.process_distributed(self.image_bytes, "box", {}))

        self.assertEqual([r[0] for r in requests], ["worker1", "worker2", "worker1", "worker2"])
        self.assertEqual(len({r[1] for r in requests}), 4)
        self.assertTrue(all(r[2] > 0 for r in requests))
        self.assertGreater(self.manager.clock.value, max(r[2] + 2 for r in requests))
        np.testing.assert_array_equal(mask, np.ones((4, 4), dtype=bool))

    def test_failed_tile_does_not_return_partial_count(self):
        def respond(request):
            if request.url.host == "worker2":
                return httpx.Response(503)
            return httpx.Response(
                200, content=bytes([1] * 4),
                headers={"X-Task-Id": request.headers["X-Task-Id"],
                         "X-Lamport-Timestamp": str(int(request.headers["X-Lamport-Timestamp"]) + 2)},
            )

        original_client = httpx.AsyncClient
        transport = httpx.MockTransport(respond)
        with patch("distributed_manager.httpx.AsyncClient",
                   side_effect=lambda **kwargs: original_client(transport=transport, **kwargs)):
            with self.assertRaisesRegex(RuntimeError, "Falha ao processar tiles"):
                asyncio.run(self.manager.process_distributed(self.image_bytes, "box", {}))

    def test_requires_two_distinct_worker_addresses(self):
        with self.assertRaises(ValueError):
            DistributedManager(["http://worker1:8001"])


if __name__ == "__main__":
    unittest.main()
