import os

import numpy as np
import torch
from PIL import Image
from skimage.measure import regionprops
from transformers import CLIPModel, CLIPProcessor

NEGATIVOS = [
    "wooden shelf", "wall", "floor", "blurry background", "price tag",
    "cereal box", "soda can", "cooking oil bottle", "rice bag",
    "glass jar", "pasta jar", "storage basket", "plastic container",
    "cardboard box", "snack package", "plastic bag", "bottle",
]

TEMPLATES = [
    "a photo of a {}.",
    "a photo of a {} on a store shelf.",
    "a close-up photo of a {}.",
]


def carregar_clip(modelo=None):
    modelo = modelo or os.environ.get("CLIP_MODELO", "openai/clip-vit-base-patch32")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return {
        "nome": modelo,
        "modelo": CLIPModel.from_pretrained(modelo).to(device).eval(),
        "processador": CLIPProcessor.from_pretrained(modelo),
        "device": device,
    }


def para_tensor(saida):
    if isinstance(saida, torch.Tensor):
        return saida
    return saida.pooler_output


def embeddings_texto(clip, frases):
    embeddings = []
    with torch.no_grad():
        for frase in frases:
            textos = [t.format(frase) for t in TEMPLATES]
            entrada = clip["processador"](text=textos, return_tensors="pt", padding=True).to(clip["device"])
            e = para_tensor(clip["modelo"].get_text_features(**entrada))
            e = e / e.norm(dim=-1, keepdim=True)
            media = e.mean(dim=0)
            embeddings.append(media / media.norm())
    return torch.stack(embeddings)


def recortar(imagem, cy, cx, lado):
    h, w = imagem.shape[:2]
    lado = max(8, min(lado, h, w))
    y0 = int(np.clip(cy - lado // 2, 0, h - lado))
    x0 = int(np.clip(cx - lado // 2, 0, w - lado))
    return imagem[y0:y0 + lado, x0:x0 + lado]


def pontuar_superpixels(clip, imagem, labels, objeto_en, janela_frac=0.08, batch=32):
    negativos = [n for n in NEGATIVOS if n not in objeto_en and objeto_en not in n]
    textos = embeddings_texto(clip, [objeto_en] + negativos)

    h, w = labels.shape
    janela_min = int(janela_frac * min(h, w))
    probs = np.zeros(labels.max() + 1, dtype=np.float32)

    recortes, ids = [], []
    for p in regionprops(labels + 1):
        cy, cx = map(int, p.centroid)
        minr, minc, maxr, maxc = p.bbox
        lado = max(int(1.5 * max(maxr - minr, maxc - minc)), janela_min)
        recortes.append(Image.fromarray(recortar(imagem, cy, cx, lado)))
        ids.append(p.label - 1)

    with torch.no_grad():
        escala = clip["modelo"].logit_scale.exp()
        for i in range(0, len(recortes), batch):
            entrada = clip["processador"](images=recortes[i:i + batch], return_tensors="pt").to(clip["device"])
            e = para_tensor(clip["modelo"].get_image_features(**entrada))
            e = e / e.norm(dim=-1, keepdim=True)
            probs[ids[i:i + batch]] = (escala * e @ textos.T).softmax(dim=-1)[:, 0].cpu().numpy()
    return probs
