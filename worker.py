import os
import io
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from pipeline import gerar_mascara_objetos
from classifier import ClassificadorCLIP

app = FastAPI(title="CV Worker")

# Cache do classificador para evitar recarregar a cada request
CLASSIFICADOR = None

@app.on_event("startup")
async def startup_event():
    global CLASSIFICADOR
    modelo = os.environ.get("CLIP_MODELO", "openai/clip-vit-base-patch32")
    CLASSIFICADOR = ClassificadorCLIP(modelo)
    print(f"Worker iniciado com modelo {modelo} no device {CLASSIFICADOR.device}")

@app.post("/process_tile")
async def process_tile(
    imagem: UploadFile = File(...),
    objeto: str = Form(...),
    limiar: float = Form(0.6),
    segmentos: int = Form(600),
    compacidade: float = Form(10.0),
    janela: float = Form(0.08),
):
    try:
        # Ler imagem do upload
        dados = await imagem.read()
        img_pil = Image.open(io.BytesIO(dados)).convert("RGB")
        img_np = np.array(img_pil)

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
        return Response(
            content=mascara.astype(np.uint8).tobytes(),
            media_type="application/octet-stream"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
