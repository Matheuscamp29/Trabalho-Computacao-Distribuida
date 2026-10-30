import io
import os
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import numpy as np
import requests

from distribuido import logical_clock

WORKER_URLS = [u.strip().rstrip("/") for u in
               os.environ.get("WORKER_URLS", "http://127.0.0.1:8001,http://127.0.0.1:8002").split(",") if u.strip()]
TIMEOUT_S = 60


def cortar_tiles(imagem, linhas=2, colunas=2, sobreposicao=0.2):
    largura, altura = imagem.size
    tile_largura = largura // colunas
    tile_altura = altura // linhas
    margem_x = int(tile_largura * sobreposicao)
    margem_y = int(tile_altura * sobreposicao)

    tiles = []
    for linha in range(linhas):
        for coluna in range(colunas):
            x0 = coluna * tile_largura
            y0 = linha * tile_altura
            x1 = (coluna + 1) * tile_largura if coluna < colunas - 1 else largura
            y1 = (linha + 1) * tile_altura if linha < linhas - 1 else altura

            rx0 = max(0, x0 - margem_x)
            ry0 = max(0, y0 - margem_y)
            rx1 = min(largura, x1 + margem_x)
            ry1 = min(altura, y1 + margem_y)

            tiles.append({
                "imagem": imagem.crop((rx0, ry0, rx1, ry1)),
                "recorte": (rx0, ry0, rx1, ry1),
                "centro": (x0, y0, x1, y1),
            })
    return tiles


def enviar_tile(url, tile, objeto, parametros, task_id):
    buffer = io.BytesIO()
    tile["imagem"].save(buffer, format="JPEG")

    carimbo = logical_clock.tick()
    print(f"[api] SEND task={task_id} worker={url} lamport={carimbo}", flush=True)
    try:
        resposta = requests.post(
            f"{url}/process_tile",
            data={
                "objeto": objeto,
                "limiar": parametros.get("limiar", 0.6),
                "segmentos": parametros.get("segmentos", 600),
                "compacidade": parametros.get("compacidade", 10.0),
                "janela": parametros.get("janela", 0.08),
            },
            files={"imagem": ("tile.jpg", buffer.getvalue(), "image/jpeg")},
            headers={"X-Task-Id": task_id, "X-Lamport-Timestamp": str(carimbo)},
            timeout=TIMEOUT_S,
        )
        carimbo_remoto = resposta.headers.get("X-Lamport-Timestamp")
        if carimbo_remoto is not None:
            carimbo = logical_clock.receive(int(carimbo_remoto))
            print(f"[api] RECEIVE task={task_id} worker={url} remote={carimbo_remoto} lamport={carimbo}", flush=True)
        if resposta.status_code != 200:
            raise RuntimeError(f"worker respondeu {resposta.status_code}: {resposta.text[:200]}")
        if carimbo_remoto is None or resposta.headers.get("X-Task-Id") != task_id:
            raise RuntimeError("resposta do worker sem identificador ou timestamp")

        rx0, ry0, rx1, ry1 = tile["recorte"]
        return np.frombuffer(resposta.content, dtype=np.uint8).reshape(ry1 - ry0, rx1 - rx0)
    except Exception as e:
        carimbo = logical_clock.tick()
        print(f"[api] ERROR task={task_id} worker={url} lamport={carimbo} error={e}", flush=True)
        return None


def processar_distribuido(imagem, objeto, parametros, worker_urls=None):
    worker_urls = worker_urls or WORKER_URLS
    if len(set(worker_urls)) < 2:
        raise ValueError("WORKER_URLS deve conter pelo menos dois workers distintos")

    tiles = cortar_tiles(imagem)
    pedido = uuid4().hex[:12]

    with ThreadPoolExecutor(max_workers=len(tiles)) as pool:
        tarefas = []
        for i, tile in enumerate(tiles):
            worker = worker_urls[i % len(worker_urls)]
            tarefas.append(pool.submit(enviar_tile, worker, tile, objeto, parametros, f"{pedido}-{i}"))
        mascaras = [t.result() for t in tarefas]

    falhas = [i for i, m in enumerate(mascaras) if m is None]
    if falhas:
        raise RuntimeError(f"Falha ao processar tiles: {falhas}")

    largura, altura = imagem.size
    mascara_global = np.zeros((altura, largura), dtype=bool)
    for tile, mascara in zip(tiles, mascaras):
        rx0, ry0, _, _ = tile["recorte"]
        x0, y0, x1, y1 = tile["centro"]
        mascara_global[y0:y1, x0:x1] = mascara[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0].astype(bool)
    print(f"[api] pedido={pedido} pixels positivos na máscara global: {int(mascara_global.sum())}", flush=True)
    return mascara_global
