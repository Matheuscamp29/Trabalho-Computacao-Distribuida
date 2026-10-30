import numpy as np
from PIL import Image

from classifier import pontuar_superpixels
from counter import construir_mascara, separar_e_contar
from segmentation import executar_slic
from translations import traduzir


def carregar_imagem(caminho, lado_max=1000):
    imagem = Image.open(caminho).convert("RGB")
    imagem.thumbnail((lado_max, lado_max), Image.LANCZOS)
    return np.array(imagem)


def gerar_mascara(clip, imagem, objeto, n_segmentos=600, compacidade=10.0, limiar=0.6, janela_frac=0.08):
    objeto_en = traduzir(objeto)
    labels = executar_slic(imagem, n_segmentos, compacidade)
    probs = pontuar_superpixels(clip, imagem, labels, objeto_en, janela_frac=janela_frac)
    mascara = construir_mascara(labels, probs, limiar)
    return mascara, labels, probs, objeto_en


def contar_objetos(clip, caminho_imagem, objeto, n_segmentos=600, compacidade=10.0, limiar=0.6,
                   area_min_frac=0.003, sensibilidade=0.06, separar=True, lado_max=1000, janela_frac=0.08):
    imagem = carregar_imagem(caminho_imagem, lado_max)
    h, w = imagem.shape[:2]
    mascara, labels, probs, objeto_en = gerar_mascara(clip, imagem, objeto, n_segmentos, compacidade, limiar, janela_frac)
    instancias, contagem = separar_e_contar(mascara, int(area_min_frac * h * w), sensibilidade, separar)
    return {
        "contagem": contagem,
        "imagem": imagem,
        "labels": labels,
        "probs": probs,
        "mascara": mascara,
        "instancias": instancias,
        "objeto_en": objeto_en,
    }
