"""Tradução simples PT -> EN dos nomes de objetos.

O CLIP entende melhor inglês. Se o objeto não estiver no dicionário,
o texto digitado é usado como está (você pode digitar direto em inglês,
ex.: "milk carton").
"""
import unicodedata


def _normalizar(texto: str) -> str:
    texto = unicodedata.normalize("NFKD", texto.strip().lower())
    return "".join(c for c in texto if not unicodedata.combining(c))


_DICIONARIO_BRUTO = {
    "caixa de leite": "milk carton",
    "leite": "milk carton",
    "leite condensado": "condensed milk box",
    "caixa de leite condensado": "condensed milk box",
    "lata de leite condensado": "condensed milk can",
    "creme de leite": "cream carton",
    "caixa de creme de leite": "cream carton",
    "leite em po": "powdered milk can",
    "caixa de suco": "juice carton",
    "garrafa de oleo": "cooking oil bottle",
    "oleo": "cooking oil bottle",
    "pacote de arroz": "rice bag",
    "arroz": "rice bag",
    "pacote de feijao": "beans bag",
    "pacote de macarrao": "pasta package",
    "macarrao": "pasta package",
    "pacote de acucar": "sugar bag",
    "acucar": "sugar bag",
    "pacote de cafe": "coffee package",
    "cafe": "coffee package",
    "lata de milho": "can of corn",
    "lata de ervilha": "can of peas",
    "molho de tomate": "tomato sauce package",
    "extrato de tomate": "tomato paste can",
    "lata de sardinha": "sardine can",
    "caixa de cereal": "cereal box",
    "garrafa de agua": "water bottle",
    "garrafinha de agua": "water bottle",
    "garrafinha": "water bottle",
    "agua": "water bottle",
    "garrafa pet": "plastic water bottle",
    "garrafa": "bottle",
    "lata": "can",
    "pote": "jar",
    "ovo": "egg carton",
    "caixa de ovos": "egg carton",
}

DICIONARIO = {_normalizar(k): v for k, v in _DICIONARIO_BRUTO.items()}


def traduzir(nome: str) -> str:
    return DICIONARIO.get(_normalizar(nome), nome.strip().lower())
