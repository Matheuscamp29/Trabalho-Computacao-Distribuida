"""Uso:
    python main.py prateleira.jpg "caixa de leite"
    python main.py prateleira.jpg "leite condensado" --limiar 0.4 --saida resultado.png
"""
import argparse

import matplotlib.pyplot as plt

from pipeline import contar_objetos
from visualization import montar_figura


def main():
    ap = argparse.ArgumentParser(description="Conta objetos de cozinha em uma imagem (SLIC + CLIP).")
    ap.add_argument("imagem", help="caminho da imagem")
    ap.add_argument("objeto", help='nome do objeto, ex.: "caixa de leite"')
    ap.add_argument("--segmentos", type=int, default=600, help="nº aproximado de superpixels SLIC")
    ap.add_argument("--compacidade", type=float, default=10.0, help="compacidade do SLIC")
    ap.add_argument("--limiar", type=float, default=0.6, help="prob. mínima p/ superpixel ser do objeto")
    ap.add_argument("--area-min", type=float, default=0.003, help="área mínima (fração da imagem)")
    ap.add_argument("--sensibilidade", type=float, default=0.06,
                    help="separação de objetos colados: menor = separa mais, maior = separa menos")
    ap.add_argument("--janela", type=float, default=0.08,
                    help="tamanho mínimo da janela de contexto do CLIP (fração do lado menor)")
    ap.add_argument("--lado-max", type=int, default=1000, help="redimensiona a imagem p/ esse lado máximo")
    ap.add_argument("--sem-separar", action="store_true", help="não separar objetos encostados")
    ap.add_argument("--saida", default="resultado.png", help="arquivo da figura gerada")
    ap.add_argument("--sem-janela", action="store_true", help="não abrir a janela do matplotlib")
    a = ap.parse_args()

    r = contar_objetos(
        a.imagem, a.objeto,
        n_segmentos=a.segmentos, compacidade=a.compacidade, limiar=a.limiar,
        area_min_frac=a.area_min, sensibilidade_separacao=a.sensibilidade,
        separar=not a.sem_separar, janela_frac=a.janela, lado_max=a.lado_max,
    )

    print(f'Objeto: "{a.objeto}" (busca: "{r.objeto_en}")')
    print(f"Quantidade encontrada: {r.contagem}")

    fig = montar_figura(r.imagem, r.labels_slic, r.probs, r.instancias, a.objeto, r.contagem)
    fig.savefig(a.saida, dpi=120)
    print(f"Figura salva em: {a.saida}")
    if not a.sem_janela:
        plt.show()


if __name__ == "__main__":
    main()
