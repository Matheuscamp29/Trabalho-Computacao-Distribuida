# Contador de objetos de cozinha (SLIC + CLIP)

## Instalação
    python -m venv venv
    source venv/bin/activate        # Windows: venv\Scripts\activate
    pip install -r requirements.txt

(Na primeira execução o modelo CLIP, ~600 MB, é baixado automaticamente.)

## Uso
    python main.py prateleira.jpg "caixa de leite"
    python main.py prateleira.jpg "leite condensado" --limiar 0.4 --sensibilidade 0.06

## Parâmetros para ajustar
- --limiar        (0.5)  baixe se faltarem objetos; suba se vier objeto errado
- --segmentos     (300)  mais = superpixels menores e bordas mais precisas (mais lento)
- --compacidade   (10)   SLIC: maior = formas mais regulares
- --sensibilidade (0.12) separação de objetos colados: menor = separa mais
- --area-min      (0.002) ignora regiões menores que essa fração da imagem
