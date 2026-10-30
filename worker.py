import os

import numpy as np
from flask import Flask, Response, jsonify, request
from PIL import Image

import logical_clock
from classifier import carregar_clip
from pipeline import gerar_mascara

WORKER_ID = os.environ.get("WORKER_ID", f"worker-{os.getpid()}")

app = Flask(__name__)
clip = carregar_clip()
print(f"[{WORKER_ID}] CLIP '{clip['nome']}' carregado (device={clip['device']})", flush=True)


@app.get("/health")
def health():
    return jsonify(status="ok", worker=WORKER_ID)


@app.post("/process_tile")
def process_tile():
    task_id = request.headers.get("X-Task-Id")
    carimbo_remoto = request.headers.get("X-Lamport-Timestamp", type=int)
    if not task_id or carimbo_remoto is None or carimbo_remoto < 0:
        return jsonify(erro="cabeçalhos X-Task-Id e X-Lamport-Timestamp são obrigatórios"), 422
    if "imagem" not in request.files or "objeto" not in request.form:
        return jsonify(erro="campos 'imagem' e 'objeto' são obrigatórios"), 422

    carimbo = logical_clock.receive(carimbo_remoto)
    print(f"[{WORKER_ID}] RECEIVE task={task_id} remote={carimbo_remoto} lamport={carimbo}", flush=True)

    try:
        imagem = np.array(Image.open(request.files["imagem"].stream).convert("RGB"))
        print(f"[{WORKER_ID}] processando task={task_id} tile={imagem.shape[1]}x{imagem.shape[0]}", flush=True)
        mascara, _, _, _ = gerar_mascara(
            clip, imagem, request.form["objeto"],
            n_segmentos=request.form.get("segmentos", 600, type=int),
            compacidade=request.form.get("compacidade", 10.0, type=float),
            limiar=request.form.get("limiar", 0.6, type=float),
            janela_frac=request.form.get("janela", 0.08, type=float),
        )
        carimbo = logical_clock.tick()
        print(f"[{WORKER_ID}] SEND task={task_id} lamport={carimbo}", flush=True)
        return Response(
            mascara.astype(np.uint8).tobytes(),
            mimetype="application/octet-stream",
            headers={"X-Task-Id": task_id, "X-Lamport-Timestamp": str(carimbo), "X-Worker-Id": WORKER_ID},
        )
    except Exception as e:
        carimbo = logical_clock.tick()
        print(f"[{WORKER_ID}] SEND_ERROR task={task_id} lamport={carimbo} error={e}", flush=True)
        return Response(
            str(e), status=500,
            headers={"X-Task-Id": task_id, "X-Lamport-Timestamp": str(carimbo), "X-Worker-Id": WORKER_ID},
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))
