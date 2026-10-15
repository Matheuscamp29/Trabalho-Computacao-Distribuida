"""Etapa 4: figura de teste/amostragem com todas as etapas."""
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.measure import regionprops
from skimage.segmentation import mark_boundaries


def montar_figura(imagem, labels_slic, probs, instancias, objeto, contagem):
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    ax[0, 0].imshow(imagem)
    ax[0, 0].set_title("1. Imagem original")

    ax[0, 1].imshow(mark_boundaries(imagem, labels_slic, color=(1, 1, 0)))
    ax[0, 1].set_title(f"2. Superpixels SLIC ({labels_slic.max() + 1})")

    ax[1, 0].imshow(imagem)
    im = ax[1, 0].imshow(probs[labels_slic], cmap="jet", alpha=0.5, vmin=0, vmax=1)
    ax[1, 0].set_title(f'3. Probabilidade de ser "{objeto}"')
    fig.colorbar(im, ax=ax[1, 0], fraction=0.046)

    sobreposicao = label2rgb(instancias, image=imagem, alpha=0.4, bg_label=0)
    ax[1, 1].imshow(mark_boundaries(sobreposicao, instancias, color=(1, 1, 1)))
    for r in regionprops(instancias):
        y, x = r.centroid
        ax[1, 1].text(x, y, str(r.label), color="black", fontsize=12, weight="bold",
                      ha="center", va="center",
                      bbox=dict(boxstyle="circle", fc="white", alpha=0.8))
    ax[1, 1].set_title(f'4. Resultado: {contagem} x "{objeto}"')

    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout()
    return fig
