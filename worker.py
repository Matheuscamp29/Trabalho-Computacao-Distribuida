import os
import io
import numpy as np
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from pipeline import gerar_mascara_objetos
from classifier import ClassificadorCLIP
from logical_clock import LamportClock

app = FastAPI(title="CV Worker")

# Cache do classificador para evitar recarregar a cada request
CLASSIFICADOR = None
CLOCK = LamportClock()
WORKER_ID = os.environ.get("WORKER_ID", f"worker-{os.getpid()}")

@app.on_event("startup")
async def startup_event():
    global CLASSIFICADOR
    modelo = os.environ.get("CLIP_MODELO", "openai/clip-vit-base-patch32")
    CLASSIFICADOR = ClassificadorCLIP(modelo)
    print(f"Worker iniciado com modelo {modelo} no device {CLASSIFICADOR.device}")

@app.get("/health")
def health():
    if CLASSIFICADOR is None:
        raise HTTPException(status_code=503, detail="modelo ainda carregando")
    return {"status": "ok"}

@app.post("/process_tile")
async def process_tile(
    imagem: UploadFile = File(...),
    objeto: str = Form(...),
    limiar: float = Form(0.6),
    segmentos: int = Form(600),
    compacidade: float = Form(10.0),
    janela: float = Form(0.08),
    task_id: str = Header(..., alias="X-Task-Id"),
    lamport_timestamp: int = Header(..., alias="X-Lamport-Timestamp", ge=0),
):
    received_timestamp = CLOCK.receive(lamport_timestamp)
    print(f"[Worker {WORKER_ID}] RECEIVE task={task_id} "
          f"remote={lamport_timestamp} lamport={received_timestamp}", flush=True)
    try:
        # Ler imagem do upload
        dados = await imagem.read()
        img_pil = Image.open(io.BytesIO(dados)).convert("RGB")
        img_np = np.array(img_pil)
        print(f"[Worker {WORKER_ID}] Processando task={task_id} "
              f"tile={img_pil.width}x{img_pil.height}", flush=True)

        # Processar apenas até a máscara
        mascara, _ = gerar_mascara_objetos(
            img_np,
            objeto,
            n_segmentos=segmentos,
            compacidade=compacidade,
            limiar=limiar,
            janela_frac=janela,
            classificador=CLASSIFICADOR
        )

        # Retorna a máscara como bytes (binária) para economizar banda
        # Usamos astype(np.uint8) para garantir que seja 0 ou 1
        send_timestamp = CLOCK.tick()
        print(f"[Worker {WORKER_ID}] SEND task={task_id} lamport={send_timestamp}", flush=True)
        return Response(
            content=mascara.astype(np.uint8).tobytes(),
            media_type="application/octet-stream",
            headers={"X-Task-Id": task_id,
                     "X-Lamport-Timestamp": str(send_timestamp),
                     "X-Worker-Id": WORKER_ID},
        )
    except Exception as e:
        send_timestamp = CLOCK.tick()
        print(f"[Worker {WORKER_ID}] SEND_ERROR task={task_id} "
              f"lamport={send_timestamp} error={e}", flush=True)
        raise HTTPException(
            status_code=500, detail=str(e),
            headers={"X-Task-Id": task_id,
                     "X-Lamport-Timestamp": str(send_timestamp),
                     "X-Worker-Id": WORKER_ID},
        ) from e

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
