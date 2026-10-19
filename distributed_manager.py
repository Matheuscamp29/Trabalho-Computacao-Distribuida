import asyncio
import httpx
import numpy as np
from PIL import Image
import io
import os

class DistributedManager:
    def __init__(self, worker_urls: list[str] = None):
        if worker_urls is None:
            configured = os.environ.get(
                "WORKER_URLS", "http://127.0.0.1:8001,http://127.0.0.1:8002"
            )
            worker_urls = configured.split(",")
        self.worker_urls = [url.strip().rstrip("/") for url in worker_urls if url.strip()]
        if len(set(self.worker_urls)) < 2:
            raise ValueError("WORKER_URLS deve conter pelo menos dois workers distintos")

    def _create_tiles(self, image: Image.Image, rows: int, cols: int, overlap_frac: float = 0.2):
        """Divide a imagem em tiles com sobreposição generosa para manter contexto do CLIP."""
        w, h = image.size
        tile_w = w // cols
        tile_h = h // rows

        # Aumentamos a sobreposição para garantir que objetos na borda sejam vistos inteiros
        ov_w = int(tile_w * overlap_frac)
        ov_h = int(tile_h * overlap_frac)

        tiles = []
        for r in range(rows):
            for c in range(cols):
                # Coordenadas do centro (zona útil)
                x_start = c * tile_w
                y_start = r * tile_h
                x_end = (c + 1) * tile_w if c < cols - 1 else w
                y_end = (r + 1) * tile_h if r < rows - 1 else h

                # Expandir para incluir overlap (respeitando bordas da imagem)
                crop_x0 = max(0, x_start - ov_w)
                crop_y0 = max(0, y_start - ov_h)
                crop_x1 = min(w, x_end + ov_w)
                crop_y1 = min(h, y_end + ov_h)

                tile_img = image.crop((crop_x0, crop_y0, crop_x1, crop_y1))

                tiles.append({
                    "image": tile_img,
                    "x_offset": crop_x0,
                    "y_offset": crop_y0,
                    "center_x0": x_start,
                    "center_y0": y_start,
                    "center_x1": x_end,
                    "center_y1": y_end,
                    "width": crop_x1 - crop_x0,
                    "height": crop_y1 - crop_y0
                })
        return tiles

    async def _dispatch_tile(self, client: httpx.AsyncClient, worker_url: str,
                             tile_data: dict, objeto: str, params: dict):
        """Envia um tile para o worker e recebe a máscara."""
        # Converter imagem do tile para bytes
        buf = io.BytesIO()
        tile_data["image"].save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        data = {
            "objeto": objeto,
            "limiar": str(params.get("limiar", 0.6)),
            "segmentos": str(params.get("segmentos", 600)),
            "compacidade": str(params.get("compacidade", 10.0)),
            "janela": str(params.get("janela", 0.08)),
        }
        files = {"imagem": ("tile.jpg", img_bytes, "image/jpeg")}

        try:
            response = await client.post(
                f"{worker_url}/process_tile",
                data=data,
                files=files
            )
            response.raise_for_status()
            # Converte bytes de volta para array numpy (uint8)
            mask = np.frombuffer(response.content, dtype=np.uint8).reshape(tile_data["height"], tile_data["width"])
            return mask
        except Exception as e:
            print(f"Erro ao processar tile no worker {worker_url}: {e}")
            return None

    async def process_distributed(self, imagem_bytes: bytes, objeto: str, params: dict):
        """Orquestra a segmentação distribuída."""
        img = Image.open(io.BytesIO(imagem_bytes)).convert("RGB")
        w, h = img.size

        # Usamos uma grade menor inicialmente para garantir que os tiles sejam grandes o suficiente
        rows, cols = 2, 2
        tiles = self._create_tiles(img, rows, cols)

        async with httpx.AsyncClient(timeout=60.0) as client:
            tasks = []
            for i, tile in enumerate(tiles):
                worker_url = self.worker_urls[i % len(self.worker_urls)]
                print(f"[Manager] Tile {i} enviado para {worker_url}")
                tasks.append(self._dispatch_tile(client, worker_url, tile, objeto, params))

            masks = await asyncio.gather(*tasks)

        failed_tiles = [i for i, mask in enumerate(masks) if mask is None]
        if failed_tiles:
            raise RuntimeError(f"Falha ao processar tiles: {failed_tiles}")

        global_mask = np.zeros((h, w), dtype=bool)

        total_pixels_found = 0

        for i, (tile_data, mask) in enumerate(zip(tiles, masks)):
            # Debug: quantos pixels positivos vieram desse tile?
            positives = np.sum(mask)
            print(f"[Manager] Tile {i} processado. Pixels positivos: {positives}")
            total_pixels_found += positives

            # Coordenadas da zona central (não sobreposta)
            local_center_x0 = tile_data["center_x0"] - tile_data["x_offset"]
            local_center_y0 = tile_data["center_y0"] - tile_data["y_offset"]
            local_center_x1 = tile_data["center_x1"] - tile_data["x_offset"]
            local_center_y1 = tile_data["center_y1"] - tile_data["y_offset"]

            # Recorta a zona útil do worker
            central_mask = mask[local_center_y0:local_center_y1, local_center_x0:local_center_x1]

            # Define onde isso entra na imagem global
            # O centro começa exatamente em center_y0, center_x0
            gy0, gx0 = tile_data["center_y0"], tile_data["center_x0"]
            gy1, gx1 = tile_data["center_y1"], tile_data["center_x1"]

            # Garante que as dimensões batem (crop vs global slice)
            chunk_h, chunk_w = central_mask.shape

            # Insere na máscara global respeitando as dimensões
            global_mask[gy0:gy0+chunk_h, gx0:gx0+chunk_w] = central_mask.astype(bool)

        print(f"[Manager] Total de pixels positivos coletados: {total_pixels_found}")
        print(f"[Manager] Pixels na máscara global final: {np.sum(global_mask)}")

        return global_mask

# Singleton para facilitar integração na API
manager = DistributedManager()
