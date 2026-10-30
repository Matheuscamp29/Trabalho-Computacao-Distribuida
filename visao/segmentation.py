from skimage.segmentation import slic


def executar_slic(imagem_rgb, n_segmentos=300, compacidade=10.0, sigma=1.0):
    return slic(imagem_rgb, n_segments=n_segmentos, compactness=compacidade, sigma=sigma,
                start_label=0, enforce_connectivity=True)
