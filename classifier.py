"""Etapa 2: decidir quais superpixels pertencem ao objeto pedido (CLIP zero-shot).

Para cada superpixel é recortada uma janela quadrada centrada nele (com contexto
suficiente para o CLIP reconhecer o objeto). A janela é comparada com o texto do
objeto e com vários textos "negativos" (prateleira, parede, outros produtos...).
A probabilidade (softmax) do texto do objeto é a pontuação do superpixel.
"""
import numpy as np
import torch
from PIL import Image
from skimage.measure import regionprops
from transformers import CLIPModel, CLIPProcessor

NEGATIVOS_PADRAO = [
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


def _para_tensor(saida):
    """Compatível com versões diferentes do transformers."""
    if isinstance(saida, torch.Tensor):
        return saida
    return saida.pooler_output


class ClassificadorCLIP:
    def __init__(self, modelo: str = "openai/clip-vit-base-patch32", device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(modelo).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(modelo)

    @torch.no_grad()
    def _embeddings_texto(self, frases):
        embs = []
        for frase in frases:
            textos = [t.format(frase) for t in TEMPLATES]
            inp = self.processor(text=textos, return_tensors="pt", padding=True).to(self.device)
            e = _para_tensor(self.model.get_text_features(**inp))
            e = e / e.norm(dim=-1, keepdim=True)
            media = e.mean(dim=0)
            embs.append(media / media.norm())
        return torch.stack(embs)

    @staticmethod
    def _recortar(imagem, cy, cx, lado):
        h, w = imagem.shape[:2]
        lado = max(8, min(lado, h, w))
        y0 = int(np.clip(cy - lado // 2, 0, h - lado))
        x0 = int(np.clip(cx - lado // 2, 0, w - lado))
        return imagem[y0:y0 + lado, x0:x0 + lado]

    @torch.no_grad()
    def pontuar_superpixels(self, imagem, labels, objeto_en,
                            negativos=None, janela_frac=0.08, batch=32):
        """Retorna vetor `probs` onde probs[i] = prob. do superpixel i ser o objeto."""
        negativos = negativos or NEGATIVOS_PADRAO
        # um negativo que contém (ou está contido em) o objeto competiria com ele
        # (ex.: "bottle" vs "water bottle"), então é removido
        negativos = [n for n in negativos if n not in objeto_en and objeto_en not in n]
        txt = self._embeddings_texto([objeto_en] + list(negativos))
        escala = self.model.logit_scale.exp()

        h, w = labels.shape
        janela_min = int(janela_frac * min(h, w))
        probs = np.zeros(labels.max() + 1, dtype=np.float32)

        recortes, ids = [], []
        for p in regionprops(labels + 1):          # regionprops ignora o rótulo 0
            cy, cx = map(int, p.centroid)
            minr, minc, maxr, maxc = p.bbox
            lado = max(int(1.5 * max(maxr - minr, maxc - minc)), janela_min)
            recortes.append(Image.fromarray(self._recortar(imagem, cy, cx, lado)))
            ids.append(p.label - 1)

        for i in range(0, len(recortes), batch):
            inp = self.processor(images=recortes[i:i + batch], return_tensors="pt").to(self.device)
            e = _para_tensor(self.model.get_image_features(**inp))
            e = e / e.norm(dim=-1, keepdim=True)
            p_obj = (escala * e @ txt.T).softmax(dim=-1)[:, 0].cpu().numpy()
            probs[ids[i:i + batch]] = p_obj
        return probs
