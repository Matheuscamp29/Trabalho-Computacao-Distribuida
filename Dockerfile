# Imagem da API do contador de produtos (CPU). Build ~5-10 min na primeira vez:
# baixa o torch (versão CPU, sem CUDA) e o CLIP (~600 MB), que ficam dentro da imagem.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    HF_HOME=/models \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_OFFLINE=0 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# torch só CPU (o pacote padrão do PyPI puxa ~3 GB de CUDA que não serve pra nada aqui)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# pré-baixa o CLIP na build: o container sobe sem precisar de internet
RUN python -c "from transformers import CLIPModel, CLIPProcessor; \
    CLIPModel.from_pretrained('openai/clip-vit-base-patch32'); \
    CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')"

COPY *.py ./

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=90s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
