import base64
import io
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
from skimage.measure import regionprops

from visao.counter import separar_e_contar
from distribuido.distributed_manager import WORKER_URLS, processar_distribuido
from visao.translations import DICIONARIO, traduzir
from visao.visualization import montar_figura_distribuida

TAMANHO_MAX_MB = float(os.environ.get("TAMANHO_MAX_MB", "20"))

app = Flask(__name__)
CORS(app)


def ler_bool(valor, padrao):
    if valor is None:
        return padrao
    return valor.strip().lower() in ("1", "true", "t", "yes", "y", "on", "sim")


def ler_parametros(form):
    return {
        "limiar": form.get("limiar", 0.6, type=float),
        "segmentos": form.get("segmentos", 600, type=int),
        "compacidade": form.get("compacidade", 10.0, type=float),
        "area_min": form.get("area_min", 0.003, type=float),
        "sensibilidade": form.get("sensibilidade", 0.06, type=float),
        "janela": form.get("janela", 0.08, type=float),
        "lado_max": form.get("lado_max", 1000, type=int),
        "separar": ler_bool(form.get("separar"), True),
    }


def validar(objeto, parametros):
    if not objeto.strip():
        return "informe o nome do objeto (ex.: 'caixa de leite')"
    if not 0.0 <= parametros["limiar"] <= 1.0:
        return "limiar deve estar entre 0 e 1"
    if not 10 <= parametros["segmentos"] <= 5000:
        return "segmentos deve estar entre 10 e 5000"
    if not 100 <= parametros["lado_max"] <= 4000:
        return "lado_max deve estar entre 100 e 4000"
    return None


def ler_imagem(arquivo):
    if arquivo is None:
        return None, "envie a foto no campo 'imagem'", 400
    dados = arquivo.read()
    if not dados:
        return None, "arquivo de imagem vazio", 400
    if len(dados) > TAMANHO_MAX_MB * 1024 * 1024:
        return None, f"imagem maior que {TAMANHO_MAX_MB:.0f} MB", 413
    try:
        Image.open(io.BytesIO(dados)).verify()
        imagem = Image.open(io.BytesIO(dados)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return None, "arquivo não é uma imagem válida (use JPG, PNG, WEBP...)", 400
    return imagem, None, None


def contar_com_workers(imagem, objeto, parametros):
    inicio = time.time()
    largura_original, altura_original = imagem.size
    imagem.thumbnail((parametros["lado_max"], parametros["lado_max"]), Image.LANCZOS)

    mascara = processar_distribuido(imagem, objeto, parametros)

    largura, altura = imagem.size
    instancias, contagem = separar_e_contar(
        mascara, int(parametros["area_min"] * largura * altura), parametros["sensibilidade"], parametros["separar"])

    return {
        "objeto": objeto,
        "objeto_en": traduzir(objeto),
        "parametros": parametros,
        "contagem": contagem,
        "imagem": np.array(imagem),
        "mascara": mascara,
        "instancias": instancias,
        "original": (largura_original, altura_original),
        "tempo_ms": int((time.time() - inicio) * 1000),
    }


def processar_formulario():
    objeto = request.form.get("objeto", "")
    parametros = ler_parametros(request.form)
    erro = validar(objeto, parametros)
    if erro:
        return None, (jsonify(erro=erro), 400)

    imagem, erro, status = ler_imagem(request.files.get("imagem"))
    if erro:
        return None, (jsonify(erro=erro), status)

    try:
        return contar_com_workers(imagem, objeto, parametros), None
    except (RuntimeError, ValueError) as e:
        return None, (jsonify(erro=str(e)), 503)


def montar_resposta(resultado):
    altura, largura = resultado["imagem"].shape[:2]
    largura_original, altura_original = resultado["original"]
    ex, ey = largura_original / largura, altura_original / altura

    instancias = []
    for p in regionprops(resultado["instancias"]):
        minr, minc, maxr, maxc = p.bbox
        cy, cx = p.centroid
        instancias.append({
            "id": int(p.label),
            "caixa": {"x": int(round(minc * ex)), "y": int(round(minr * ey)),
                      "largura": int(round((maxc - minc) * ex)), "altura": int(round((maxr - minr) * ey))},
            "centro": {"x": int(round(cx * ex)), "y": int(round(cy * ey))},
            "area_px": int(round(p.area * ex * ey)),
        })

    return {
        "objeto": resultado["objeto"],
        "objeto_en": resultado["objeto_en"],
        "quantidade": resultado["contagem"],
        "instancias": instancias,
        "imagem_original": {"largura": largura_original, "altura": altura_original},
        "imagem_processada": {"largura": largura, "altura": altura},
        "parametros": resultado["parametros"],
        "tempo_ms": resultado["tempo_ms"],
    }


def figura_png(resultado):
    fig = montar_figura_distribuida(resultado["imagem"], resultado["mascara"], resultado["instancias"],
                                    resultado["objeto"], resultado["contagem"])
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)
    return buffer.getvalue()


@app.get("/health")
def health():
    workers = {}
    for url in WORKER_URLS:
        try:
            workers[url] = requests.get(f"{url}/health", timeout=2).status_code == 200
        except requests.RequestException:
            workers[url] = False
    return jsonify(status="ok" if all(workers.values()) else "degradado", workers=workers)


@app.get("/objetos")
def objetos():
    return jsonify(traducoes=dict(sorted(DICIONARIO.items())))


@app.post("/contar")
def contar():
    resultado, erro = processar_formulario()
    if erro:
        return erro
    resposta = montar_resposta(resultado)
    if ler_bool(request.form.get("figura"), False):
        resposta["figura_png_base64"] = base64.b64encode(figura_png(resultado)).decode("ascii")
    return jsonify(resposta)


@app.post("/contar/figura")
def contar_figura():
    resultado, erro = processar_formulario()
    if erro:
        return erro
    return Response(
        figura_png(resultado), mimetype="image/png",
        headers={"X-Quantidade": str(resultado["contagem"]),
                 "X-Objeto-En": resultado["objeto_en"],
                 "X-Tempo-Ms": str(resultado["tempo_ms"])},
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
