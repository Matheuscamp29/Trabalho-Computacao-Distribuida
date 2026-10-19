"""Orquestra: carregar -> SLIC -> CLIP -> máscara -> contagem."""
from dataclasses import dataclass

import numpy as np
from PIL import Image

from classifier import ClassificadorCLIP
from counter import construir_mascara, separar_e_contar
from segmentation import executar_slic
from translations import traduzir


@dataclass
class Resultado:
    contagem: int
    imagem: np.ndarray
    labels_slic: np.ndarray
    probs: np.ndarray
    mascara: np.ndarray
    instancias: np.ndarray
    objeto_en: str


def carregar_imagem(caminho: str, lado_max: int = 1000) -> np.ndarray:
    img = Image.open(caminho).convert("RGB")
    img.thumbnail((lado_max, lado_max), Image.LANCZOS)   # mantém proporção
    return np.array(img)


def gerar_mascara_objetos(imagem: np.ndarray, objeto: str, *,
                              n_segmentos=600, compacidade=10.0, limiar=0.6,
                              janela_frac=0.08, classificador=None) -> tuple[np.ndarray, str]:
    """Etapa de segmentação e pontuação, retornando a máscara binária."""
    objeto_en = traduzir(objeto)
    labels = executar_slic(imagem, n_segmentos, compacidade)
    classificador = classificador or ClassificadorCLIP()
    probs = classificador.pontuar_superpixels(imagem, labels, objeto_en,
                                              janela_frac=janela_frac)
    mascara = construir_mascara(labels, probs, limiar)
    return mascara, objeto_en



def contar_objetos(caminho_imagem: str, objeto: str, *,
                   n_segmentos=600, compacidade=10.0, limiar=0.6,
                   area_min_frac=0.003, sensibilidade_separacao=0.06,
                   separar=True, lado_max=1000, janela_frac=0.08, classificador=None) -> Resultado:
    imagem = carregar_imagem(caminho_imagem, lado_max)
    h, w = imagem.shape[:2]

    mascara, objeto_en = gerar_mascara_objetos(imagem, objeto,
                                               n_segmentos=n_segmentos, compacidade=compacidade,
                                               limiar=limiar, janela_frac=janela_frac,
                                               classificador=classificador)

    instancias, contagem = separar_e_contar(
        mascara,
        area_minima_px=int(area_min_frac * h * w),
        sensibilidade=sensibilidade_separacao,
        separar=separar,
    )
    return Resultado(contagem, imagem, None, None, mascara, instancias, objeto_en)
