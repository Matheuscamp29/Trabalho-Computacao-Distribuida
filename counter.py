"""Etapa 3: juntar superpixels positivos, separar objetos colados e contar."""
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label as rotular
from skimage.morphology import disk, h_maxima
from skimage.segmentation import watershed


def construir_mascara(labels, probs, limiar):
    positivos = np.where(probs >= limiar)[0]
    mascara = np.isin(labels, positivos)
    # fecha buracos pequenos e remove ruído fino
    mascara = ndi.binary_closing(mascara, structure=disk(3))
    mascara = ndi.binary_opening(mascara, structure=disk(3))
    return mascara


def separar_e_contar(mascara, area_minima_px, sensibilidade=0.12, separar=True):
    """Rotula componentes conexos; opcionalmente separa objetos encostados
    (caixas lado a lado) com watershed sobre a transformada de distância.

    Retorna (mapa_de_instancias, contagem).
    """
    if not mascara.any():
        return np.zeros(mascara.shape, dtype=int), 0

    componentes = rotular(mascara)

    if separar:
        dist = ndi.gaussian_filter(ndi.distance_transform_edt(mascara), 2)
        # máxima-h: só considera "picos" que se destacam o suficiente, evitando
        # dividir um mesmo objeto em vários por causa do platô da distância
        h = max(1.0, sensibilidade * dist.max())
        picos = h_maxima(dist, h) & mascara
        marcadores = rotular(picos)
        instancias = watershed(-dist, marcadores, mask=mascara)
        # componentes sem nenhum pico (muito finos) viram uma instância própria
        faltando = mascara & (instancias == 0)
        if faltando.any():
            extra = rotular(faltando)
            extra[extra > 0] += instancias.max()
            instancias = instancias + extra
    else:
        instancias = componentes

    # descarta regiões pequenas e renumera 1..N
    final = np.zeros_like(instancias)
    n = 0
    for rid in np.unique(instancias):
        if rid == 0:
            continue
        regiao = instancias == rid
        if regiao.sum() >= area_minima_px:
            n += 1
            final[regiao] = n
    return final, n
