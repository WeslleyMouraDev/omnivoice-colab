# 📖 Guia Completo da API — OmniVoice Colab Studio

Este guia ensina como utilizar a API do **OmniVoice Colab Studio** diretamente a partir de scripts Python, notebooks locais, ferramentas de automação (como n8n, Make ou webhooks) e pipelines de produção de vídeo (como geradores de rascunhos para CapCut e Premiere).

---

## 🚀 Como Funciona

Ao executar o notebook no Google Colab, o Gradio gera uma URL pública temporária com suporte a requisições remotas, no padrão:

```text
https://xxxxxxxx.gradio.live
```

Esse mesmo endereço disponibiliza tanto a interface visual para navegadores quanto a API REST/WebSocket que pode ser consumida pela biblioteca `gradio_client`.

---

## 📦 Instalação do Cliente

No seu ambiente local (ou servidor onde seus scripts de automação rodam), instale a biblioteca oficial do Gradio Client:

```bash
pip install gradio_client
```

---

## 🔍 Inspecionando a API com `view_api()`

Para visualizar os detalhes de todos os endpoints disponíveis na sua instância em tempo real (tipos de dados, parâmetros e retornos):

```python
from gradio_client import Client

GRADIO_URL = "https://SEU-LINK.gradio.live"
client = Client(GRADIO_URL)

# Imprime a documentação completa dos endpoints no terminal
client.view_api()
```

---

## 🎯 Endpoints da API

### 1. `/prepare_voice`
Prepara uma voz a partir de uma amostra de áudio (3 a 15 segundos). O backend utiliza o **faster-whisper** (`small`, int8 na CPU) para transcrever a fala sem consumir VRAM da GPU e cria o prompt de clonagem no **OmniVoice**, armazenando-o em um cache temporário sob um `voice_id` único.

#### Parâmetros de Entrada
| Parâmetro | Tipo | Descrição |
| :--- | :--- | :--- |
| `ref_audio` | `FilePath` (via `handle_file`) | Caminho para o arquivo de áudio de referência (WAV, MP3, M4A, etc.). Recomendado: 3 a 15 segundos de fala limpa. |

#### Saídas Retornadas
| Campo | Tipo | Descrição |
| :--- | :--- | :--- |
| `voice_id` | `str` | Identificador único da voz (ex: `voice_a82f71c922`). Válido durante toda a sessão do Colab. |
| `transcript` | `str` | Transcrição textual automática detectada pelo Whisper. |
| `status` | `str` | Mensagem de confirmação ou erro do processamento. |

#### Exemplo de Código:
```python
from gradio_client import Client, handle_file

client = Client("https://SEU-LINK.gradio.live")

voice_id, transcript, status = client.predict(
    ref_audio=handle_file("minha_locucao.wav"),
    api_name="/prepare_voice"
)

print(f"Status: {status}")
print(f"Voice ID Gerado: {voice_id}")
print(f"Transcrição Detectada: {transcript}")
```

---

### 2. `/generate_tts`
Gera a narração de um roteiro reutilizando uma voz previamente preparada através do `voice_id`. 
- Divide automaticamente roteiros longos em blocos de até `subtitle_max_chars` caracteres.
- Sintetiza cada bloco individualmente no OmniVoice.
- Insere pausas de silêncio configuráveis (`pause_ms`) entre os blocos.
- Gera o arquivo `.srt` com timestamps baseados na contagem real de amostras do áudio.
- Exporta o áudio completo montado em `.mp3` a 192 kbps.

#### Parâmetros de Entrada
| Parâmetro | Tipo | Padrão | Descrição |
| :--- | :--- | :--- | :--- |
| `voice_id` | `str` | Obrigatório | Identificador retornado pelo endpoint `/prepare_voice`. |
| `text` | `str` | Obrigatório | Roteiro ou texto a sintetizar. Pode ter qualquer tamanho. |
| `speed` | `float` | `1.0` | Velocidade da fala (0.5 a 2.0). 1.0 é velocidade normal. |
| `num_step` | `int` | `32` | Passos de inferência de difusão (4 a 64). 32 é ideal em velocidade e qualidade. |
| `guidance_scale` | `float` | `2.0` | Escala CFG (0.0 a 4.0). 2.0 oferece ótima fidelidade à referência. |
| `pause_ms` | `int` | `140` | Duração do silêncio entre frases em milissegundos (0 a 500 ms). |
| `subtitle_max_chars` | `int` | `180` | Limite de caracteres por legenda/bloco de áudio. |

#### Saídas Retornadas
| Campo | Tipo | Descrição |
| :--- | :--- | :--- |
| `mp3` | `FilePath` | Caminho local temporário do arquivo MP3 baixado (192 kbps). |
| `srt` | `FilePath` | Caminho local temporário do arquivo de legendas SRT sincronizado. |
| `status` | `str` | Mensagem de conclusão informando blocos gerados e duração total. |

#### Exemplo de Código:
```python
from gradio_client import Client

client = Client("https://SEU-LINK.gradio.live")

mp3_file, srt_file, status = client.predict(
    voice_id="voice_a82f71c922",
    text="A inteligência artificial está transformando a criação de vídeos. Com o OmniVoice, geramos locuções com alta fidelidade.",
    speed=1.0,
    num_step=32,
    guidance_scale=2.0,
    pause_ms=140,
    subtitle_max_chars=180,
    api_name="/generate_tts"
)

print(f"Status: {status}")
print(f"MP3 salvo em: {mp3_file}")
print(f"SRT salvo em: {srt_file}")
```

---

### 3. `/clone_and_generate`
Executa o fluxo completo em uma única requisição: recebe o áudio de referência, transcreve, gera a voz e sintetiza o roteiro retornando o Voice ID, a transcrição, o MP3 e o SRT.

#### Parâmetros de Entrada
| Parâmetro | Tipo | Padrão | Descrição |
| :--- | :--- | :--- | :--- |
| `ref_audio` | `FilePath` | Obrigatório | Áudio de referência da voz (3 a 15s). |
| `text` | `str` | Obrigatório | Roteiro ou texto a sintetizar. |
| `speed` | `float` | `1.0` | Velocidade da fala. |
| `num_step` | `int` | `32` | Passos de inferência. |
| `guidance_scale` | `float` | `2.0` | Escala CFG. |
| `pause_ms` | `int` | `140` | Pausa entre blocos (ms). |
| `subtitle_max_chars` | `int` | `180` | Limite de caracteres por bloco. |

#### Saídas Retornadas
| Campo | Tipo | Descrição |
| :--- | :--- | :--- |
| `voice_id` | `str` | Identificador da voz gerada. |
| `transcript` | `str` | Transcrição da amostra de voz. |
| `mp3` | `FilePath` | Arquivo MP3 baixado (192 kbps). |
| `srt` | `FilePath` | Arquivo de legendas SRT perfeitamente sincronizado. |
| `status` | `str` | Status de execução. |

#### Exemplo de Código:
```python
from gradio_client import Client, handle_file

client = Client("https://SEU-LINK.gradio.live")

voice_id, transcript, mp3_file, srt_file, status = client.predict(
    ref_audio=handle_file("amostra_voz.wav"),
    text="Este é um teste completo gerando áudio e legendas em uma única chamada de API.",
    speed=1.0,
    num_step=32,
    guidance_scale=2.0,
    pause_ms=140,
    subtitle_max_chars=180,
    api_name="/clone_and_generate"
)

print(f"Voice ID: {voice_id}")
print(f"MP3: {mp3_file}")
print(f"SRT: {srt_file}")
```

---

## 🎬 Integração com Automações de Vídeo (CapCut, Premiere, etc.)

O arquivo `.srt` retornado contém exatamente os textos do roteiro original divididos de acordo com `subtitle_max_chars` e com marcações de tempo que coincidem milissegundo a milissegundo com as pausas inseridas no arquivo `.mp3`:

```text
1
00:00:00,000 --> 00:00:03,420
A inteligência artificial está transformando a criação de vídeos.

2
00:00:03,560 --> 00:00:06,850
Com o OmniVoice, geramos locuções com alta fidelidade.
```

Para automatizar:
1. Chame `/generate_tts` a partir do seu webhook ou script Python.
2. Mova o `.mp3` e o `.srt` para a pasta do seu projeto.
3. Importe ambos na sua linha do tempo para ter legendas e locução 100% sincronizadas instantaneamente.

---

## 🛡️ Segurança e Ética

1. **Consentimento Obrigatório:** Utilize clonagem de voz apenas com sua própria voz ou com consentimento prévio e explícito do proprietário da voz.
2. **Privacidade no Colab:** O link `https://xxxxx.gradio.live` é público na internet enquanto a sessão estiver em execução. Encerre o Colab após o uso para revogar o acesso.
