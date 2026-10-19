# Contador de produtos em prateleira

Você manda uma foto da prateleira e o nome do produto ("caixa de leite", "garrafa de água"...) e ele devolve quantas unidades achou e onde cada uma está. Por baixo é SLIC (segmentação) + CLIP (reconhecimento zero-shot) + watershed pra separar objetos encostados. Não precisa treinar nada.

## Rodando na sua máquina

Precisa de Python 3.10 ou mais novo.

```
git clone https://github.com/Matheuscamp29/Trabalho-Computacao-Distribuida.git
cd Trabalho-Computacao-Distribuida
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements-api.txt
uvicorn api:app --port 8000
```

Na primeira vez ele baixa o modelo CLIP (uns 600 MB), então demora um pouco pra subir. Quando aparecer `[api] CLIP ... carregado` no terminal, tá pronto. Abre http://localhost:8000/docs que dá pra testar mandando uma foto direto pelo navegador.

Se for Linux, instala o torch antes com `pip install torch --index-url https://download.pytorch.org/whl/cpu`, senão o pip baixa 3 GB de CUDA à toa.

Pra atualizar depois é só `git pull` na pasta.

## Modo distribuído (dois workers)

O coordenador divide a imagem em quatro blocos e envia os blocos alternadamente
para dois workers HTTP. Para testar em uma máquina, abra **três terminais** na pasta
do projeto, com o ambiente virtual ativado em cada um:

```text
Terminal 1: python -m uvicorn worker:app --host 0.0.0.0 --port 8001
Terminal 2: python -m uvicorn worker:app --host 0.0.0.0 --port 8002
Terminal 3: python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

O valor padrão de `WORKER_URLS` é
`http://127.0.0.1:8001,http://127.0.0.1:8002`. Para usar duas máquinas na
mesma rede, execute um worker em cada uma, libere a porta 8001 no firewall e
configure na máquina da API os IPs acessíveis dos workers antes de iniciá-la:

```bash
# Linux/macOS
export WORKER_URLS="http://IP_DA_MAQUINA_1:8001,http://IP_DA_MAQUINA_2:8001"
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

```powershell
# PowerShell
$env:WORKER_URLS = "http://IP_DA_MAQUINA_1:8001,http://IP_DA_MAQUINA_2:8001"
python -m uvicorn api:app --host 0.0.0.0 --port 8000
```

Com os três processos prontos, envie a imagem com `distribuido=true`:

```bash
curl -F "imagem=@prateleira.jpg" -F "objeto=caixa de leite" -F "distribuido=true" http://localhost:8000/contar
```

No PowerShell, use `curl.exe` no lugar de `curl`. Os logs do coordenador mostram
o destino de cada bloco; os logs dos workers mostram os blocos recebidos. Uma
falha de worker retorna HTTP 503, evitando uma contagem parcial.

Com Docker, `docker compose up --build` inicia a API e dois workers em contêineres
separados. Depois use o mesmo `curl` acima. O Compose usa os nomes internos
`worker1` e `worker2` como endereços. Cada processo carrega sua própria cópia
do modelo CLIP, então reserve memória suficiente para os três processos.

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
  "tempo_ms": 4500
}
```

As coordenadas são em pixels da imagem que você mandou, então dá pra desenhar os retângulos direto por cima dela no front. CORS tá liberado, pode chamar do navegador.

Outras rotas:
- `GET /health` responde 200 quando o modelo carregou (503 enquanto carrega)
- `GET /objetos` lista os nomes em português que ele já traduz. O que não estiver na lista ele usa como veio, então em inglês funciona melhor ("milk carton")
- `POST /contar/figura` recebe a mesma coisa e devolve um PNG com as etapas, bom pra ver o que ele tá fazendo

Se o resultado vier errado, os ajustes mais úteis são mandar junto no form `limiar` (padrão 0.6; baixa se faltou objeto, sobe se veio objeto errado) e `sensibilidade` (padrão 0.06; menor separa mais objetos colados). Tem mais alguns no `/docs`.

Roda em CPU. O tempo por foto depende da máquina e do modo escolhido; no modo
local, as inferências são serializadas.

## Sem instalar nada

Dá pra consumir pelo túnel de quem já tá com a API rodando: `./servir.sh` sobe a API e abre uma URL pública (`https://alguma-coisa.trycloudflare.com`). Aí é só trocar o `http://localhost:8000` por essa URL. A URL muda toda vez que o túnel reinicia e só funciona com a máquina de origem ligada.

Também tem um `docker-compose.yml` se preferir container: `docker compose up --build`.

## Linha de comando

```
python main.py prateleira.jpg "caixa de leite"
python main.py prateleira.jpg "leite condensado" --limiar 0.4 --sensibilidade 0.06
```

Gera um `resultado.png` com as quatro etapas lado a lado.
