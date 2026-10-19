"""API HTTP do contador de produtos (FastAPI).

Rodar local:   uvicorn api:app --host 0.0.0.0 --port 8000
Via Docker:    docker compose up --build
Docs interativas (Swagger): http://localhost:8000/docs
"""
import base64
import io
import os
import threading
import time
from contextlib import asynccontextmanager

import matplotlib

matplotlib.use("Agg")  # sem janela gráfica dentro do container
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from skimage.measure import regionprops

from classifier import ClassificadorCLIP
from pipeline import contar_objetos, gerar_mascara_objetos
from translations import DICIONARIO
from visualization import montar_figura
from distributed_manager import manager as dist_manager


MODELO = os.environ.get("CLIP_MODELO", "openai/clip-vit-base-patch32")
TAMANHO_MAX_MB = float(os.environ.get("TAMANHO_MAX_MB", "20"))

_estado: dict = {}
_trava = threading.Lock()  # CLIP em CPU: uma inferência por vez, as demais esperam na fila


@asynccontextmanager
async def lifespan(_app: FastAPI):
    t = time.time()
    _estado["classificador"] = ClassificadorCLIP(MODELO)
    print(f"[api] CLIP '{MODELO}' carregado em {time.time() - t:.1f}s "
          f"(device={_estado['classificador'].device})", flush=True)
    yield
    _estado.clear()


app = FastAPI(
    title="Contador de produtos (SLIC + CLIP)",
    description=(
        "Envie uma foto de prateleira e o nome do produto; a API devolve quantas unidades "
        "encontrou e onde cada uma está (caixas em pixels da imagem original)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
# libera chamadas direto do navegador (front-end em outra origem)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ----------------------------------------------------------------------------- modelos de resposta
class Caixa(BaseModel):
    x: int = Field(description="canto superior esquerdo (px da imagem original)")
    y: int
    largura: int
    altura: int


class Ponto(BaseModel):
    x: int
    y: int


class Instancia(BaseModel):
    id: int
    caixa: Caixa
    centro: Ponto
    area_px: int


class Dimensoes(BaseModel):
    largura: int
    altura: int


class RespostaContagem(BaseModel):
    objeto: str = Field(description="texto recebido")
    objeto_en: str = Field(description="texto usado no CLIP (traduzido quando está no dicionário)")
    quantidade: int
    instancias: list[Instancia] = Field(description="uma entrada por unidade encontrada")
    imagem_original: Dimensoes
    imagem_processada: Dimensoes = Field(description="tamanho após o redimensionamento interno (lado_max)")
    parametros: dict
    tempo_ms: int
    figura_png_base64: str | None = Field(
        default=None, description="painel 2x2 com as etapas, só quando figura=true")


class Saude(BaseModel):
    status: str
    modelo: str
    device: str


# ----------------------------------------------------------------------------- núcleo
def _ler_imagem(arquivo: UploadFile) -> tuple[bytes, Dimensoes]:
    dados = arquivo.file.read()
    if not dados:
        raise HTTPException(400, "arquivo de imagem vazio")
    if len(dados) > TAMANHO_MAX_MB * 1024 * 1024:
        raise HTTPException(413, f"imagem maior que {TAMANHO_MAX_MB:.0f} MB")
    try:
        with Image.open(io.BytesIO(dados)) as im:
            im.verify()
        with Image.open(io.BytesIO(dados)) as im:
            largura, altura = im.size
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "arquivo não é uma imagem válida (use JPG, PNG, WEBP...)")
    return dados, Dimensoes(largura=largura, altura=altura)


def _validar(objeto: str, limiar: float, segmentos: int, lado_max: int):
    if not objeto or not objeto.strip():
        raise HTTPException(400, "informe o nome do objeto (ex.: 'caixa de leite')")
    if not 0.0 <= limiar <= 1.0:
        raise HTTPException(400, "limiar deve estar entre 0 e 1")
    if segmentos < 10 or segmentos > 5000:
        raise HTTPException(400, "segmentos deve estar entre 10 e 5000")
    if lado_max < 100 or lado_max > 4000:
        raise HTTPException(400, "lado_max deve estar entre 100 e 4000")


def _processar(imagem: UploadFile, objeto: str, limiar: float, segmentos: int, compacidade: float,
               area_min: float, sensibilidade: float, janela: float, lado_max: int, separar: bool,
               distribuido: bool = False):
    """Roda o pipeline e devolve (Resultado, resposta_sem_figura)."""
    _validar(objeto, limiar, segmentos, lado_max)
    dados, original = _ler_imagem(imagem)
    parametros = dict(limiar=limiar, segmentos=segmentos, compacidade=compacidade, area_min=area_min,
                      sensibilidade=sensibilidade, janela=janela, lado_max=lado_max, separar=separar)

    t = time.time()

    if distribuido:
        # MODO DISTRIBUÍDO
        import asyncio
        import numpy as np
        from pipeline import Resultado
        from counter import separar_e_contar
        from translations import traduzir

        # Executa o manager de forma síncrona dentro da rota FastAPI (simplificação)
        mascara_global = asyncio.run(dist_manager.process_distributed(dados, objeto, parametros))

        # Agora usamos a lógica de contagem local sobre a máscara global
        instancias, contagem = separar_e_contar(
            mascara_global,
            area_minima_px=int(area_min * original.largura * original.altura),
            sensibilidade=sensibilidade,
            separar=separar,
        )

        r = Resultado(
            contagem=contagem,
            imagem=np.array(Image.open(io.BytesIO(dados)).convert("RGB")),
            labels_slic=None,
            probs=None,
            mascara=mascara_global,
            instancias=instancias,
            objeto_en=traduzir(objeto) if 'traduzir' in globals() else objeto
        )
    else:
        # MODO LOCAL
        with _trava:
            r = contar_objetos(
                io.BytesIO(dados), objeto,
                n_segmentos=segmentos, compacidade=compacidade, limiar=limiar,
                area_min_frac=area_min, sensibilidade_separacao=sensibilidade,
                separar=separar, janela_frac=janela, lado_max=lado_max,
                classificador=_estado["classificador"],
            )

    tempo_ms = int((time.time() - t) * 1000)

    h, w = r.imagem.shape[:2]
    ex, ey = original.largura / w, original.altura / h  # volta p/ coordenadas da imagem enviada
    instancias_resp = []
    for p in regionprops(r.instancias):
        minr, minc, maxr, maxc = p.bbox
        cy, cx = p.centroid
        instancias_resp.append(Instancia(
            id=int(p.label),
            caixa=Caixa(x=round(minc * ex), y=round(minr * ey),
                        largura=round((maxc - minc) * ex), altura=round((maxr - minr) * ey)),
            centro=Ponto(x=round(cx * ex), y=round(cy * ey)),
            area_px=round(p.area * ex * ey),
        ))

    resposta = RespostaContagem(
        objeto=objeto, objeto_en=r.objeto_en, quantidade=r.contagem, instancias=instancias_resp,
        imagem_original=original, imagem_processada=Dimensoes(largura=w, altura=h),
        parametros=parametros, tempo_ms=tempo_ms,
    )
    return r, resposta


def _figura_png(r, objeto: str) -> bytes:
    fig = montar_figura(r.imagem, r.labels_slic, r.probs, r.instancias, objeto, r.contagem)
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=120)
    finally:
        plt.close(fig)
    return buf.getvalue()


# ----------------------------------------------------------------------------- rotas
@app.get("/health", response_model=Saude, tags=["infra"])
def health():
    c = _estado.get("classificador")
    if c is None:
        raise HTTPException(503, "modelo ainda carregando")
    return Saude(status="ok", modelo=MODELO, device=c.device)


@app.get("/objetos", tags=["consulta"])
def objetos():
    """Nomes em português que a API já traduz pro CLIP. Qualquer outro texto é usado como
    está (de preferência em inglês, ex.: 'milk carton')."""
    return {"traducoes": dict(sorted(DICIONARIO.items()))}


_DOC_PARAMS = """
Parâmetros opcionais (todos via form-data):

- **limiar** (0.6): prob. mínima p/ um pedaço da imagem ser o objeto. Baixe se faltar objeto, suba se vier objeto errado.
- **segmentos** (600): nº de superpixels SLIC. Mais = bordas mais precisas, mais lento.
- **compacidade** (10): SLIC, maior = formas mais regulares.
- **sensibilidade** (0.06): separação de objetos colados. Menor = separa mais.
- **area_min** (0.003): ignora regiões menores que essa fração da imagem.
- **janela** (0.08): tamanho mínimo da janela de contexto do CLIP (fração do lado menor).
- **lado_max** (1000): a imagem é reduzida pra esse lado máximo antes de processar.
- **separar** (true): false = não tenta separar objetos encostados.
"""


@app.post("/contar", response_model=RespostaContagem, tags=["contagem"],
          description="Conta as unidades do objeto na imagem e devolve JSON com as caixas.\n" + _DOC_PARAMS
          + "\n- **figura** (false): true = inclui o painel 2x2 em PNG base64 na resposta.")
def contar(
    imagem: UploadFile = File(..., description="foto da prateleira (JPG/PNG/WEBP)"),
    objeto: str = Form(..., description='nome do produto, ex.: "caixa de leite"'),
    limiar: float = Form(0.6),
    segmentos: int = Form(600),
    compacidade: float = Form(10.0),
    area_min: float = Form(0.003),
    sensibilidade: float = Form(0.06),
    janela: float = Form(0.08),
    lado_max: int = Form(1000),
    separar: bool = Form(True),
    figura: bool = Form(False),
    distribuido: bool = Form(False),
):
    r, resposta = _processar(imagem, objeto, limiar, segmentos, compacidade, area_min,
                             sensibilidade, janela, lado_max, separar, distribuido=distribuido)
    if figura:
        resposta.figura_png_base64 = base64.b64encode(_figura_png(r, objeto)).decode("ascii")
    return resposta


@app.post("/contar/figura", tags=["contagem"], response_class=Response,
          responses={200: {"content": {"image/png": {}}}},
          description="Mesma entrada do /contar, mas devolve direto o PNG do painel 2x2 "
                      "(original, superpixels, mapa de probabilidade, instâncias numeradas).\n" + _DOC_PARAMS)
def contar_figura(
    imagem: UploadFile = File(...),
    objeto: str = Form(...),
    limiar: float = Form(0.6),
    segmentos: int = Form(600),
    compacidade: float = Form(10.0),
    area_min: float = Form(0.003),
    sensibilidade: float = Form(0.06),
    janela: float = Form(0.08),
    lado_max: int = Form(1000),
    separar: bool = Form(True),
):
    r, resposta = _processar(imagem, objeto, limiar, segmentos, compacidade, area_min,
                             sensibilidade, janela, lado_max, separar)
    return Response(content=_figura_png(r, objeto), media_type="image/png",
                    headers={"X-Quantidade": str(resposta.quantidade),
                             "X-Objeto-En": resposta.objeto_en,
                             "X-Tempo-Ms": str(resposta.tempo_ms)})
