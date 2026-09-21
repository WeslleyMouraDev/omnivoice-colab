# OmniVoice Colab Studio (PT-BR) 🎙️

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/WeslleyMouraDev/omnivoice-colab/blob/main/OmniVoice_Colab_Studio_API_SRT.ipynb)
[![GitHub](https://img.shields.io/badge/GitHub-WeslleyMouraDev%2Fomnivoice--colab-blue?logo=github)](https://github.com/WeslleyMouraDev/omnivoice-colab)
[![API Guide](https://img.shields.io/badge/API-Guia%20em%20Portugu%C3%AAs-green?logo=markdown)](API_PTBR.md)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)

Versão aprimorada do **OmniVoice** projetada especificamente para execução no **Google Colab** (GPU T4 recomendada), com foco em clonagem de voz em **Português Brasileiro (PT-BR)**, síntese por blocos para roteiros longos, exportação direta em **MP3 (192 kbps)**, legendas **SRT perfeitamente sincronizadas** e uma **API pública reutilizável** via Gradio.

---

## 🇧🇷 OmniVoice Colab Studio — Recursos Principais

- **Clonagem de Voz com Amostras Curtas:** Clone qualquer voz usando uma amostra curta de **3 a 15 segundos** (upload ou gravação direta pelo microfone).
- **Transcrição Automática da Referência:** O modelo `faster-whisper` (`small`, int8 na CPU) transcreve automaticamente a amostra de áudio sem ocupar a VRAM da GPU dedicada ao OmniVoice.
- **Divisão Inteligente de Roteiros Longos:** O algoritmo `split_for_subtitles` quebra textos longos respeitando o fluxo natural de pontuação final (`.`, `!`, `?`), pausas (`:`, `;`, `—`), vírgulas e palavras inteiras (padrão de até 180 caracteres por bloco).
- **Legendas SRT Altamente Sincronizadas:** O texto da legenda vem diretamente do roteiro original (sem alucinações ou palavras trocadas por retranscrição). Os timestamps do SRT são calculados a partir da duração real em amostras de cada bloco sintetizado somada às pausas configuráveis.
- **Pausa Inter-Blocos Ajustável (`pause_ms`):** Controle milimétrico do silêncio entre frases (padrão: 140 ms).
- **Exportação Direta em MP3 (192 kbps):** Áudio final montado e exportado diretamente em alta qualidade para facilitar o uso no CapCut, Premiere, DaVinci Resolve ou automações.
- **Sistema de Cache Temporário com Voice ID:** Prepare uma voz uma única vez para receber um `voice_id` (ex: `voice_a82f71c922`) e reutilize-a em centenas de gerações sem reenviar o áudio de referência.
- **Interface Gradio em Abas:**
  - **Aba 1 (Tudo em 1):** Amostra de voz + Texto → MP3 + SRT em um clique.
  - **Aba 2 (Preparar e Reutilizar):** Criação de Voice ID e síntese reutilizável.
  - **Abas 3 e 4:** Interfaces originais de *Voice Clone* e *Voice Design* do OmniVoice.
- **API Pública do Gradio:** Endpoints nomeados prontos para integração com `gradio_client`, scripts Python e automações web.

---

## 🌐 Endpoints da API

Acesse o guia completo em [API_PTBR.md](API_PTBR.md) para ver exemplos detalhados de código com `gradio_client`.

1. `/prepare_voice`: Envia `ref_audio` e recebe `voice_id`, `transcript` e `status`.
2. `/generate_tts`: Envia `voice_id`, `text`, `speed`, `num_step`, `guidance_scale`, `pause_ms`, `subtitle_max_chars` e recebe `mp3`, `srt` e `status`.
3. `/clone_and_generate`: Executa o fluxo completo em uma única chamada.

### Exemplo Rápido em Python:
```python
from gradio_client import Client, handle_file

client = Client("https://SEU-LINK.gradio.live")

# 1. Preparar voz
voice_id, transcript, status = client.predict(
    ref_audio=handle_file("minha_voz.wav"),
    api_name="/prepare_voice"
)

# 2. Gerar áudio MP3 + SRT com a mesma voz
mp3, srt, status = client.predict(
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

## ⚙️ Limitações no Google Colab

- **Sessão Temporária:** As instâncias gratuitas do Google Colab expiram após inatividade ou atingir o limite de tempo contínuo (geralmente algumas horas). O cache de `Voice ID` e arquivos temporários são redefinidos quando a sessão é encerrada.
- **Link Público:** O link `https://xxxxx.gradio.live` permanece ativo enquanto o comando no Colab estiver em execução.

---

## 🛡️ Segurança e Uso Ético

> ⚠️ **Aviso de Responsabilidade:**
> - Use clonagem de voz apenas com sua própria voz ou mediante **autorização expressa e documentada** do titular da voz clonada.
> - É proibido o uso desta ferramenta para difamação, falsificação de identidade, fraude, desinformação ou qualquer atividade ilegal.
> - Lembre-se de que os links públicos `gradio.live` gerados ficam acessíveis na internet durante a execução do Colab.

---

## 🇺🇸 Run OmniVoice On Google Colab (Original Documentation)

Run **OmniVoice** easily on Google Colab, no complex setup required.

[![Open In Colab (Original)](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NeuralFalconYT/omnivoice-colab/blob/main/OmniVoice_Colab.ipynb)

### 🧠 About
This is a Google Colab version of the original OmniVoice model.
It allows you to quickly generate high-quality speech from text with minimal setup.

### 🔹 Features
* Convert text → speech
* Clone voices from audio
* Support 600+ languages ([language list](https://github.com/k2-fsa/OmniVoice/blob/master/docs/languages.md))
* Add emotions using tags (`[laughter]`, etc.)
* Customize voice (pitch, accent, style)
* Fast and high-quality output
* Generate subtitles (SRT)

### 🖼️ Gradio APP Screenshot 

![voice_clone](https://github.com/user-attachments/assets/ffc989b4-2a5b-478b-a34f-5ae3eb2d9776)

![voice_design](https://github.com/user-attachments/assets/29a2dd1a-5987-4cb8-be4f-681456ddd546)

---

## 🙌 Credits

- Original OmniVoice Project: 👉 [https://github.com/k2-fsa/OmniVoice](https://github.com/k2-fsa/OmniVoice)
- OmniVoice Colab Base: 👉 [https://github.com/NeuralFalconYT/omnivoice-colab](https://github.com/NeuralFalconYT/omnivoice-colab)
- Fork and Colab Studio Enhancements: 👉 [https://github.com/WeslleyMouraDev/omnivoice-colab](https://github.com/WeslleyMouraDev/omnivoice-colab)

---

## 📄 License

This repository is distributed under the [Apache License 2.0](https://github.com/k2-fsa/OmniVoice/blob/master/LICENSE).
