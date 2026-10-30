import argparse

import matplotlib.pyplot as plt

from visao.classifier import carregar_clip
from visao.pipeline import contar_objetos
from visao.visualization import montar_figura

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conta objetos em uma imagem (SLIC + CLIP) em uma única máquina.")
    parser.add_argument("imagem", help="caminho da imagem")
    parser.add_argument("objeto", help='nome do objeto, ex.: "caixa de leite"')
    parser.add_argument("--segmentos", type=int, default=600, help="número aproximado de superpixels SLIC")
    parser.add_argument("--compacidade", type=float, default=10.0, help="compacidade do SLIC")
    parser.add_argument("--limiar", type=float, default=0.6, help="probabilidade mínima para o superpixel ser do objeto")
    parser.add_argument("--area-min", type=float, default=0.003, help="área mínima (fração da imagem)")
    parser.add_argument("--sensibilidade", type=float, default=0.06, help="separação de objetos colados: menor separa mais")
    parser.add_argument("--janela", type=float, default=0.08, help="janela mínima de contexto do CLIP (fração do lado menor)")
    parser.add_argument("--lado-max", type=int, default=1000, help="redimensiona a imagem para esse lado máximo")
    parser.add_argument("--sem-separar", action="store_true", help="não separar objetos encostados")
    parser.add_argument("--saida", default="resultado.png", help="arquivo da figura gerada")
    parser.add_argument("--sem-janela", action="store_true", help="não abrir a janela do matplotlib")
    a = parser.parse_args()

    clip = carregar_clip()
    r = contar_objetos(clip, a.imagem, a.objeto, n_segmentos=a.segmentos, compacidade=a.compacidade,
                       limiar=a.limiar, area_min_frac=a.area_min, sensibilidade=a.sensibilidade,
                       separar=not a.sem_separar, lado_max=a.lado_max, janela_frac=a.janela)

    print(f'Objeto: "{a.objeto}" (busca: "{r["objeto_en"]}")')
    print(f"Quantidade encontrada: {r['contagem']}")

    fig = montar_figura(r["imagem"], r["labels"], r["probs"], r["instancias"], a.objeto, r["contagem"])
    fig.savefig(a.saida, dpi=120)
    print(f"Figura salva em: {a.saida}")
    if not a.sem_janela:
        plt.show()
