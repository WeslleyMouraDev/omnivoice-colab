# Prompt para Codex — OmniVoice Colab com API, MP3 e SRT

## Objetivo

Trabalhe no repositório:

`WeslleyMouraDev/omnivoice-colab`

Quero transformar esse fork em uma versão aprimorada do OmniVoice para Google Colab, com foco em clonagem de voz em PT-BR, interface Gradio pública, API reutilizável, saída em MP3 e geração de SRT altamente sincronizado.

Preserve o projeto original sempre que possível e implemente as novas funcionalidades de forma organizada, documentada e fácil de usar.

---

## Requisitos principais

### 1. Interface gráfica Gradio

A aplicação deve manter uma interface gráfica Gradio acessível no navegador.

No Google Colab, ao executar o projeto, deve gerar um link público no formato:

```text
https://xxxxx.gradio.live
```

Esse mesmo link deve servir:

- a interface gráfica;
- a API do Gradio.

A inicialização deve usar algo equivalente a:

```python
demo.queue(max_size=20).launch(
    share=True,
    show_api=True
)
```

---

## 2. Clonagem de voz

A interface deve permitir:

- upload de áudio;
- gravação pelo microfone;
- clonagem de voz usando OmniVoice;
- geração de fala em português brasileiro;
- reutilização da mesma voz em várias gerações.

A amostra recomendada deve ser curta, aproximadamente:

```text
3 a 15 segundos
```

O usuário não deve precisar fornecer manualmente a transcrição da amostra.

---

## 3. Transcrição automática da voz de referência

Use `faster-whisper` para transcrever automaticamente a amostra de voz.

Preferência:

```python
WhisperModel(
    "small",
    device="cpu",
    compute_type="int8"
)
```

Isso evita ocupar VRAM adicional desnecessariamente.

A transcrição deve ser usada automaticamente para criar o prompt de clonagem do OmniVoice.

Exemplo conceitual:

```python
voice_prompt = model.create_voice_clone_prompt(
    ref_audio=ref_audio,
    ref_text=transcript
)
```

A transcrição detectada também deve aparecer na interface.

---

## 4. Sistema de Voice ID

Implemente um cache temporário de vozes.

Depois de preparar uma voz, gere um identificador semelhante a:

```text
voice_a82f71c922
```

Esse `voice_id` deve apontar para o prompt de clonagem armazenado em memória.

Estrutura sugerida:

```python
VOICE_CACHE = {}
VOICE_META = {}
```

Exemplo:

```python
VOICE_CACHE[voice_id] = voice_prompt
VOICE_META[voice_id] = {
    "transcript": transcript,
    "reference_audio": ref_audio
}
```

A voz deve poder ser reutilizada sem reenviar o áudio de referência.

O cache pode ser temporário e desaparecer quando a sessão do Colab terminar.

---

## 5. API pública

Crie endpoints Gradio nomeados.

### Endpoint `/prepare_voice`

Entrada:

```text
ref_audio
```

Saída:

```text
voice_id
transcript
status
```

Exemplo de chamada:

```python
from gradio_client import Client, handle_file

client = Client("https://SEU-LINK.gradio.live")

voice_id, transcript, status = client.predict(
    ref_audio=handle_file("minha_voz.wav"),
    api_name="/prepare_voice"
)
```

---

### Endpoint `/generate_tts`

Entrada:

```text
voice_id
text
speed
num_step
guidance_scale
pause_ms
subtitle_max_chars
```

Saída:

```text
mp3
srt
status
```

Exemplo:

```python
mp3_file, srt_file, status = client.predict(
    voice_id=voice_id,
    text="Hoje vamos entender como a inteligência artificial funciona.",
    speed=1.0,
    num_step=32,
    guidance_scale=2.0,
    pause_ms=140,
    subtitle_max_chars=180,
    api_name="/generate_tts"
)
```

---

### Endpoint `/clone_and_generate`

Esse endpoint deve realizar tudo em uma única chamada:

```text
áudio de referência
↓
transcrição automática
↓
criação da voz
↓
geração do TTS
↓
MP3
↓
SRT
```

Entrada:

```text
ref_audio
text
speed
num_step
guidance_scale
pause_ms
subtitle_max_chars
```

Saída sugerida:

```text
voice_id
transcript
mp3
srt
status
```

---

## 6. Textos longos

A aplicação deve aceitar roteiros longos.

Não envie textos muito grandes diretamente ao modelo.

Implemente divisão inteligente por:

1. final de frase;
2. pontuação;
3. vírgulas e dois-pontos;
4. palavras, somente se necessário.

Função sugerida:

```python
split_for_subtitles(text, max_chars=180)
```

Evite cortar palavras no meio.

---

## 7. SRT altamente sincronizado

Não quero que o SRT final dependa apenas de uma retranscrição por Whisper.

O texto da legenda deve vir diretamente do roteiro original.

Fluxo desejado:

```text
roteiro
↓
divisão em blocos
↓
cada bloco é sintetizado individualmente
↓
medir duração real de cada áudio
↓
montar áudio final
↓
usar essas durações para calcular timestamps do SRT
```

Isso evita erros como:

- palavra trocada;
- pontuação diferente;
- números transcritos incorretamente;
- nomes próprios alterados.

Exemplo:

```python
start_sec = cursor_samples / sample_rate
end_sec = (cursor_samples + len(waveform)) / sample_rate
```

Entrada do SRT:

```python
{
    "start": start_sec,
    "end": end_sec,
    "text": texto_original
}
```

Formato:

```text
1
00:00:00,000 --> 00:00:03,420
Hoje vamos entender como a inteligência artificial funciona.
```

A sincronização precisa corresponder ao áudio final montado.

---

## 8. Pausas

Adicione uma pausa controlável entre os blocos.

Padrão recomendado:

```text
140 ms
```

Parâmetro:

```text
pause_ms
```

Faixa sugerida:

```text
0 a 500 ms
```

A pausa deve ser considerada nos timestamps do SRT.

---

## 9. Remoção de silêncios longos

Utilize o próprio pós-processamento do OmniVoice sempre que possível.

Preferência:

```python
postprocess_output=True
```

Pode usar:

```python
pad_duration=0.08
fade_duration=0.03
```

O objetivo é evitar silêncios excessivos dentro dos trechos sem cortar pausas naturais.

Não aplique uma remoção agressiva que deixe a voz artificial.

---

## 10. Exportação MP3

O arquivo final principal deve ser:

```text
MP3
```

Qualidade:

```text
192 kbps
```

Pode gerar WAV internamente e depois converter usando `pydub` + `ffmpeg`.

Exemplo:

```python
audio.export(
    output_mp3,
    format="mp3",
    bitrate="192k"
)
```

---

## 11. Interface desejada

Crie duas formas de uso.

### Aba 1 — Tudo em uma etapa

Campos:

```text
Amostra de voz
Texto
Velocidade
Inference steps
Guidance scale
Pausa
Tamanho máximo da legenda
```

Botão:

```text
Gerar MP3 + SRT
```

Resultados:

```text
Voice ID
Transcrição da amostra
Áudio
MP3
SRT
Status
```

---

### Aba 2 — Preparar voz e reutilizar

Primeira parte:

```text
Amostra de voz
↓
Preparar voz
↓
Voice ID
Transcrição
```

Segunda parte:

```text
Voice ID
Texto
Configurações
↓
Gerar
↓
MP3 + SRT
```

Essa aba será especialmente útil para automações.

---

## 12. Configuração padrão recomendada

Use inicialmente:

```text
speed = 1.0
num_step = 32
guidance_scale = 2.0
pause_ms = 140
subtitle_max_chars = 180
MP3 = 192 kbps
```

Permita ajuste pela interface.

---

## 13. Dependências

Garanta que o Colab instale automaticamente o necessário.

Provavelmente:

```text
ffmpeg
pydub
soundfile
faster-whisper
gradio
gradio_client
```

Mantenha compatibilidade com as dependências já existentes no fork.

---

## 14. Google Colab

Atualize ou crie um notebook:

```text
OmniVoice_Colab_Studio_API_SRT.ipynb
```

O notebook deve:

1. verificar GPU;
2. instalar dependências;
3. baixar/preparar OmniVoice;
4. iniciar a aplicação;
5. mostrar o link público;
6. documentar os endpoints;
7. incluir exemplos da API.

Preferência de GPU:

```text
T4
```

---

## 15. README

Atualize o `README.md`.

Não apague os créditos originais.

Adicione uma seção em português com:

```text
OmniVoice Colab Studio
```

Explique:

- clonagem;
- PT-BR;
- transcrição automática;
- interface;
- link público;
- API;
- Voice ID;
- MP3;
- SRT;
- limitações do Colab.

Inclua botão:

```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](...)
```

Apontando para o novo notebook do fork.

---

## 16. Guia da API

Crie:

```text
API_PTBR.md
```

Inclua exemplos completos usando:

```python
from gradio_client import Client, handle_file
```

Documente:

```text
/prepare_voice
/generate_tts
/clone_and_generate
```

Inclua também:

```python
client.view_api()
```

---

## 17. Segurança

Adicione aviso claro:

```text
Use clonagem de voz apenas com sua própria voz ou com autorização explícita da pessoa dona da voz.
```

Também explique que:

```text
gradio.live é público enquanto a sessão estiver ativa.
```

---

## 18. Compatibilidade

Antes de alterar funções existentes:

1. leia `app.py`;
2. leia `subtitle.py`;
3. leia `OmniVoice_Colab.ipynb`;
4. preserve funcionalidades úteis do fork original;
5. evite duplicar lógica que já existe.

Reutilize o código existente quando fizer sentido.

---

## 19. Estrutura final desejada

Algo próximo de:

```text
omnivoice-colab/
│
├── README.md
├── API_PTBR.md
├── OmniVoice_Colab.ipynb
├── OmniVoice_Colab_Studio_API_SRT.ipynb
├── app.py
├── subtitle.py
├── requirements.txt
├── colab.txt
└── ...
```

---

## 20. Verificação obrigatória

Antes de considerar o trabalho concluído:

- valide sintaxe Python;
- valide importações principais;
- valide estrutura do notebook;
- confirme que todos os `api_name` estão definidos;
- confirme que os outputs dos endpoints correspondem aos exemplos;
- confirme que o SRT usa os tempos reais do áudio;
- confirme que as pausas estão refletidas nos timestamps;
- confirme que MP3 e SRT são entregues pela interface;
- confirme que a API pode reutilizar `voice_id`;
- preserve os créditos do OmniVoice e do fork original.

Se não for possível executar o modelo por ausência de GPU no ambiente do Codex, faça validação estática e deixe isso explicitamente documentado.

---

## Resultado esperado

Ao final, quero poder:

```text
Abrir o Colab
↓
Executar
↓
Receber link Gradio
↓
Clonar minha voz uma vez
↓
Obter Voice ID
↓
Enviar roteiros pela API
↓
Receber MP3 + SRT sincronizado
```

Fluxo final:

```text
ROTEIRO
   ↓
/generate_tts
   ↓
OmniVoice
   ↓
MP3
+
SRT
   ↓
CapCut Draft Builder / automação / publicação
```

Implemente as mudanças diretamente no fork:

```text
WeslleyMouraDev/omnivoice-colab
```

Crie commits claros e pequenos, por exemplo:

```text
feat: add reusable voice id api
feat: add mp3 and timed srt generation
docs: add pt-br colab and api guide
```
