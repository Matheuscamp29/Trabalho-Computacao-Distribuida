import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage.measure import regionprops
from skimage.segmentation import mark_boundaries


def desenhar_instancias(ax, imagem, instancias, objeto, contagem):
    sobreposicao = label2rgb(instancias, image=imagem, alpha=0.4, bg_label=0)
    ax.imshow(mark_boundaries(sobreposicao, instancias, color=(1, 1, 1)))
    for r in regionprops(instancias):
        y, x = r.centroid
        ax.text(x, y, str(r.label), color="black", fontsize=12, weight="bold",
                ha="center", va="center", bbox=dict(boxstyle="circle", fc="white", alpha=0.8))
    ax.set_title(f'Resultado: {contagem} x "{objeto}"')


def montar_figura(imagem, labels, probs, instancias, objeto, contagem):
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))

    ax[0, 0].imshow(imagem)
    ax[0, 0].set_title("1. Imagem original")

    ax[0, 1].imshow(mark_boundaries(imagem, labels, color=(1, 1, 0)))
    ax[0, 1].set_title(f"2. Superpixels SLIC ({labels.max() + 1})")

    ax[1, 0].imshow(imagem)
    im = ax[1, 0].imshow(probs[labels], cmap="jet", alpha=0.5, vmin=0, vmax=1)
    ax[1, 0].set_title(f'3. Probabilidade de ser "{objeto}"')
    fig.colorbar(im, ax=ax[1, 0], fraction=0.046)

    desenhar_instancias(ax[1, 1], imagem, instancias, objeto, contagem)
    ax[1, 1].set_title(f'4. Resultado: {contagem} x "{objeto}"')

    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout()
    return fig


def montar_figura_distribuida(imagem, mascara, instancias, objeto, contagem):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    ax[0].imshow(imagem)
    ax[0].set_title("1. Imagem original")

    ax[1].imshow(imagem)
    ax[1].imshow(mascara, cmap="Reds", alpha=0.5)
    ax[1].set_title("2. Máscara costurada dos workers")

    desenhar_instancias(ax[2], imagem, instancias, objeto, contagem)
    ax[2].set_title(f'3. Resultado: {contagem} x "{objeto}"')

    for a in ax:
        a.axis("off")
    fig.tight_layout()
    return fig
