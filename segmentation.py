"""Etapa 1: segmentação em superpixels com SLIC."""
import numpy as np
from skimage.segmentation import slic


def executar_slic(imagem_rgb: np.ndarray,
                  n_segmentos: int = 300,
                  compacidade: float = 10.0,
                  sigma: float = 1.0) -> np.ndarray:
    """Retorna um mapa (H, W) com o id (0..N-1) do superpixel de cada pixel.

    n_segmentos : quantidade aproximada de superpixels
    compacidade : maior = superpixels mais quadrados; menor = seguem mais as bordas
    sigma       : suavização gaussiana antes de segmentar
    """
    return slic(
        imagem_rgb,
        n_segments=n_segmentos,
        compactness=compacidade,
        sigma=sigma,
        start_label=0,
        enforce_connectivity=True,
    )
