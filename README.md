# Contador de produtos em prateleira (distribuído)

Você manda uma foto da prateleira e o nome do produto ("caixa de leite", "garrafa de água"...) e a API devolve quantas unidades achou e onde cada uma está. Por baixo é SLIC (segmentação) + CLIP (reconhecimento zero-shot) + watershed pra separar objetos encostados. Não precisa treinar nada.

O processamento pesado não roda na API: ela coordena dois (ou mais) workers, que podem estar em outras máquinas da rede.

## Como funciona

1. A API recebe a foto em `POST /contar`, reduz pra `lado_max` e corta em 4 blocos (grade 2x2) com 20% de sobreposição, pra nenhum objeto da borda ficar cortado ao meio.
2. Manda os 4 blocos em paralelo pros workers, alternando: blocos 0 e 2 pro primeiro, 1 e 3 pro segundo. Cada envio leva os cabeçalhos `X-Task-Id` e `X-Lamport-Timestamp`.
3. Cada worker roda SLIC + CLIP no seu bloco e devolve uma máscara binária (bytes 0/1), sem JSON, pra economizar banda.
4. A API costura só a zona central de cada máscara numa máscara do tamanho da imagem, separa objetos colados e conta.
5. Se qualquer bloco falhar, a resposta é 503. Nunca sai contagem parcial.

Quem é quem:

- `api.py`: a API Flask. Só coordena, não carrega o CLIP.
- `distributed_manager.py`: corta a imagem, envia os blocos, costura as máscaras.
- `worker.py`: servidor Flask que carrega o CLIP e atende `/process_tile`.
- `logical_clock.py`: relógio de Lamport (um contador por processo).
- `pipeline.py`, `segmentation.py`, `classifier.py`, `counter.py`: a parte de visão.
- `visualization.py`: figuras das etapas.
- `main.py`: mesma contagem, mas em uma máquina só, pela linha de comando.

## Rodando na sua máquina

Precisa de Python 3.10 ou mais novo.

```
git clone https://github.com/Matheuscamp29/Trabalho-Computacao-Distribuida.git
cd Trabalho-Computacao-Distribuida
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements-api.txt
```

Se for Linux, instala o torch antes com `pip install torch --index-url https://download.pytorch.org/whl/cpu`, senão o pip baixa 3 GB de CUDA à toa.

Abra três terminais na pasta do projeto, com o venv ativado em cada um:

```text
Terminal 1: PORT=8001 WORKER_ID=worker1 python worker.py
Terminal 2: PORT=8002 WORKER_ID=worker2 python worker.py
Terminal 3: python api.py
```

No PowerShell é `$env:PORT=8001; $env:WORKER_ID="worker1"; python worker.py`.

Na primeira vez cada worker baixa o modelo CLIP (uns 600 MB), então demora um pouco pra subir. Quando aparecer `CLIP ... carregado` no terminal, tá pronto. Cada worker usa uns 1,5 GB de RAM com o modelo carregado, então reserve memória pros dois. A API sobe na hora porque não carrega modelo nenhum. O padrão de `WORKER_URLS` é `http://127.0.0.1:8001,http://127.0.0.1:8002`.

Confere se a API enxerga os workers:

```bash
curl http://localhost:8000/health
```

E manda uma foto:

```bash
curl -F "imagem=@images.jpg" -F "objeto=garrafa de agua" http://localhost:8000/contar
```

No PowerShell, use `curl.exe` no lugar de `curl`. Os logs da API mostram pra qual worker cada bloco foi; os logs dos workers mostram os blocos recebidos.

## Duas máquinas na mesma rede

Rode um worker em cada máquina, libere a porta 8001 no firewall e aponte a API pros IPs deles:

```bash
# em cada máquina
PORT=8001 WORKER_ID=worker1 python worker.py

# na máquina da API
export WORKER_URLS="http://IP_DA_MAQUINA_1:8001,http://IP_DA_MAQUINA_2:8001"
python api.py
```

```powershell
$env:WORKER_URLS = "http://IP_DA_MAQUINA_1:8001,http://IP_DA_MAQUINA_2:8001"
python api.py
```

## Relógio lógico de Lamport

Cada processo tem seu próprio contador, começando em zero. Ao enviar uma mensagem, incrementa o contador (`tick`). Ao receber uma mensagem com timestamp `T`, atualiza pra `max(contador_local, T) + 1` (`receive`). Isso registra a ordem causal das mensagens mesmo que os relógios físicos das máquinas sejam diferentes.

A API identifica cada bloco por `X-Task-Id` e manda seu contador em `X-Lamport-Timestamp`. O worker devolve os mesmos cabeçalhos com seu novo timestamp, mais `X-Worker-Id`. Nos logs, procure as linhas `SEND` e `RECEIVE` com o mesmo `task`: o `RECEIVE` do worker sempre tem timestamp maior que o `SEND` da API, e o `RECEIVE` da API é maior que o `SEND` do worker. Os números exatos dependem da ordem de execução paralela. Exemplo pra um bloco:

```text
[api] SEND task=abc-0 worker=http://127.0.0.1:8001 lamport=1
[worker1] RECEIVE task=abc-0 remote=1 lamport=2
[worker1] SEND task=abc-0 lamport=3
[api] RECEIVE task=abc-0 worker=http://127.0.0.1:8001 remote=3 lamport=4
```

## Usando na sua aplicação

É um POST multipart em `/contar` com dois campos: `imagem` (o arquivo) e `objeto` (o texto).

```js
const form = new FormData();
form.append("imagem", arquivo);          // File do <input type="file">
form.append("objeto", "caixa de leite");

const res = await fetch("http://localhost:8000/contar", { method: "POST", body: form });
const dados = await res.json();
```

Volta assim:

```json
{
  "objeto": "caixa de leite",
  "objeto_en": "milk carton",
  "quantidade": 2,
  "instancias": [
    { "id": 1, "caixa": { "x": 1613, "y": 288, "largura": 180, "altura": 259 }, "centro": { "x": 1696, "y": 430 }, "area_px": 28097 },
    { "id": 2, "caixa": { "x": 1586, "y": 547, "largura": 118, "altura": 182 }, "centro": { "x": 1655, "y": 643 }, "area_px": 16145 }
  ],
  "imagem_original": { "largura": 1800, "altura": 2400 },
  "imagem_processada": { "largura": 750, "altura": 1000 },
  "parametros": { "limiar": 0.6, "segmentos": 600, "compacidade": 10.0, "area_min": 0.003, "sensibilidade": 0.06, "janela": 0.08, "lado_max": 1000, "separar": true },
  "tempo_ms": 4500
}
```

As coordenadas são em pixels da imagem que você mandou, então dá pra desenhar os retângulos direto por cima dela no front. CORS tá liberado, pode chamar do navegador.

O limite de tamanho da foto é 20 MB; dá pra mudar com a variável de ambiente `TAMANHO_MAX_MB` antes de subir a API.

Outras rotas:

- `GET /health`: diz se cada worker está respondendo (`status` vira `degradado` se algum caiu).
- `GET /objetos`: lista os nomes em português que a API já traduz. O que não estiver na lista é usado como veio, então em inglês funciona melhor ("milk carton").
- `POST /contar/figura`: mesma entrada do `/contar`, mas devolve um PNG com a foto, a máscara costurada dos workers e as instâncias numeradas. Mandar `figura=true` no `/contar` inclui esse PNG em base64 no campo `figura_png_base64`.

Parâmetros opcionais, todos via form-data:

- `limiar` (0.6): probabilidade mínima pra um pedaço da imagem ser o objeto. Baixe se faltou objeto, suba se veio objeto errado.
- `segmentos` (600): quantidade de superpixels SLIC. Mais = bordas mais precisas, mais lento.
- `compacidade` (10): SLIC, maior = formas mais regulares.
- `sensibilidade` (0.06): separação de objetos colados. Menor = separa mais.
- `area_min` (0.003): ignora regiões menores que essa fração da imagem.
- `janela` (0.08): tamanho mínimo da janela de contexto do CLIP (fração do lado menor).
- `lado_max` (1000): a imagem é reduzida pra esse lado máximo antes de ser cortada.
- `separar` (true): false = não tenta separar objetos encostados.

## Testes

```bash
python -m unittest discover -s tests
```

Os testes não carregam o CLIP. Eles usam workers falsos pra conferir o protocolo: rodízio dos blocos, cabeçalhos, relógio de Lamport, falha de worker virando 503 e o formato da resposta.

## Uma máquina só (linha de comando)

```
python main.py images.jpg "garrafa de agua"
python main.py prateleira.jpg "leite condensado" --limiar 0.4 --sensibilidade 0.06
```

Roda a mesma contagem sem workers e gera um `resultado.png` com as quatro etapas lado a lado. Serve de base pra comparar com o modo distribuído.
