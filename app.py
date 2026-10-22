# %cd /content/omnivoice-colab
import os
import sys
import logging
import tempfile
import uuid
import re
from typing import Any, Dict, Optional, Tuple

import gradio as gr
import numpy as np
import torch
import scipy.io.wavfile as wavfile

# Módulo do estúdio PT-BR com SRT, MP3 e Voice ID
import studio_core

temp_audio_dir = "./Omni_Audio"
os.makedirs(temp_audio_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Setup path to import subtitle_maker from /content/omnivoice-colab/OmniVoice/
OmniVoice_path = f"{os.getcwd()}/OmniVoice/"
sys.path.append(OmniVoice_path)

try:
    from subtitle import subtitle_maker
    from subtitle import LANGUAGE_CODE as WHISPER_LANGUAGE_CODE
except ImportError:
    subtitle_maker = None
    WHISPER_LANGUAGE_CODE = None

try:
    from omnivoice import OmniVoice, OmniVoiceGenerationConfig
    from omnivoice.utils.lang_map import LANG_NAMES, lang_display_name
except ImportError:
    OmniVoice = None
    OmniVoiceGenerationConfig = None
    LANG_NAMES = []
    def lang_display_name(n): return str(n)

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
)
logging.getLogger("omnivoice").setLevel(logging.DEBUG)

# ---------------------------------------------------------------------------
# Model Loading (Global Scope)
# ---------------------------------------------------------------------------
model = None
sampling_rate = 24000

if OmniVoice is not None:
    print("Loading model from k2-fsa/OmniVoice...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    try:
        from hf_mirror import download_model
    except ImportError:
        download_model = None

    try:
        model = OmniVoice.from_pretrained(
            "k2-fsa/OmniVoice",
            device_map=device,
            dtype=dtype,
            load_asr=False,
        )
        sampling_rate = model.sampling_rate
        print(f"Model loaded successfully on {device}!")
    except Exception as e:
        print(f"Standard loading failed: {e}. Trying hf_mirror fallback...")
        if download_model is not None:
            try:
                omnivoice_model_path = download_model(
                    "k2-fsa/OmniVoice",
                    download_folder="./OmniVoice_Model",
                    redownload=False,
                    workers=6,
                    use_snapshot=False,
                )
                model = OmniVoice.from_pretrained(
                    omnivoice_model_path,
                    device_map=device,
                    dtype=dtype,
                    load_asr=False,
                )
                sampling_rate = model.sampling_rate
                print("Model loaded successfully from local snapshot!")
            except Exception as e2:
                print(f"Failed to load model from mirror: {e2}")
        else:
            print("hf_mirror not available.")

# ---------------------------------------------------------------------------
# Event Tags & JS Functions (Original)
# ---------------------------------------------------------------------------
EVENT_TAGS = [
    "[laughter]", "[sigh]", "[confirmation-en]", "[question-en]", 
    "[question-ah]", "[question-oh]", "[question-ei]", "[question-yi]",
    "[surprise-ah]", "[surprise-oh]", "[surprise-wa]", "[surprise-yo]", 
    "[dissatisfaction-hnn]"
]

INSERT_TAG_JS_VC = """
(tag_val, current_text) => {
    const textarea = document.querySelector('#vc_textbox textarea');
    if (!textarea) return current_text + " " + tag_val;
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    let prefix = " ";
    let suffix = " ";
    if (!current_text) return tag_val;
    if (start === 0) prefix = "";
    else if (current_text[start - 1] === ' ') prefix = "";
    if (end < current_text.length && current_text[end] === ' ') suffix = "";
    return current_text.slice(0, start) + prefix + tag_val + suffix + current_text.slice(end);
}
"""

INSERT_TAG_JS_VD = """
(tag_val, current_text) => {
    const textarea = document.querySelector('#vd_textbox textarea');
    if (!textarea) return current_text + " " + tag_val;
    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    let prefix = " ";
    let suffix = " ";
    if (!current_text) return tag_val;
    if (start === 0) prefix = "";
    else if (current_text[start - 1] === ' ') prefix = "";
    if (end < current_text.length && current_text[end] === ' ') suffix = "";
    return current_text.slice(0, start) + prefix + tag_val + suffix + current_text.slice(end);
}
"""

# ---------------------------------------------------------------------------
# UI Configurations & Language Mappings (Original)
# ---------------------------------------------------------------------------
_ALL_LANGUAGES = ["Auto"] + sorted(lang_display_name(n) for n in LANG_NAMES)

_CATEGORIES = {
    "Gender": ["Male", "Female"],
    "Age": ["Child", "Teenager", "Young Adult", "Middle-aged", "Elderly"],
    "Pitch": ["Very Low Pitch", "Low Pitch", "Moderate Pitch", "High Pitch", "Very High Pitch"],
    "Style": ["Whisper"],
    "English Accent": [
        "American Accent", "Australian Accent", "British Accent", "Chinese Accent",
        "Canadian Accent", "Indian Accent", "Korean Accent", "Portuguese Accent",
        "Russian Accent", "Japanese Accent"
    ],
    "Chinese Dialect": [
        "Henan Dialect", "Shaanxi Dialect", "Sichuan Dialect", "Guizhou Dialect",
        "Yunnan Dialect", "Guilin Dialect", "Jinan Dialect", "Shijiazhuang Dialect",
        "Gansu Dialect", "Ningxia Dialect", "Qingdao Dialect", "Northeast Dialect"
    ],
}

DIALECT_MAP = {
    "Henan Dialect": "河南话", "Shaanxi Dialect": "陕西话", "Sichuan Dialect": "四川话",
    "Guizhou Dialect": "贵州话", "Yunnan Dialect": "云南话", "Guilin Dialect": "桂林话",
    "Jinan Dialect": "济南话", "Shijiazhuang Dialect": "石家庄话", "Gansu Dialect": "甘肃话",
    "Ningxia Dialect": "宁夏话", "Qingdao Dialect": "青岛话", "Northeast Dialect": "东北话",
}

_ATTR_INFO = {
    "English Accent": "Only effective for English speech.",
    "Chinese Dialect": "Only effective for Chinese speech.",
}

# ---------------------------------------------------------------------------
# Funções Auxiliares do Projeto Original
# ---------------------------------------------------------------------------
def _is_whisper_supported(lang):
    if not lang or lang == "Auto":
        return True 
    if WHISPER_LANGUAGE_CODE is None:
        return True 
    supported_langs = [str(k).lower() for k in WHISPER_LANGUAGE_CODE.keys()] + \
                      [str(v).lower() for v in WHISPER_LANGUAGE_CODE.values()]
    lang_lower = lang.lower()
    for w_lang in supported_langs:
        if w_lang in lang_lower or lang_lower in w_lang:
            return True
    return False

def generate_subtitles_if_needed(wav_path, lang, want_subs):
    if not want_subs or subtitle_maker is None:
        return None, None, None
    if not _is_whisper_supported(lang):
        logging.warning(f"Language '{lang}' is likely unsupported by Whisper. Skipping subtitle generation.")
        return None, None, None
    try:
        whisper_lang = lang if (lang and lang != "Auto") else None
        whisper_results = subtitle_maker(wav_path, whisper_lang)
        if whisper_results and len(whisper_results) > 3:
            return whisper_results[1], whisper_results[2], whisper_results[3] 
    except Exception as e:
        logging.warning(f"Subtitle generation failed: {e}")
    return None, None, None

def tts_file_name(text, language="en"):
    global temp_audio_dir
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text)
    clean_text = clean_text.lower().strip().replace(" ", "_")
    if not clean_text:
        clean_text = "audio"
    truncated = clean_text[:20]
    lang = re.sub(r'\s+', '_', language.strip().lower()) if language else "unknown"
    rand = uuid.uuid4().hex[:8].upper()
    return f"{temp_audio_dir}/{truncated}_{lang}_{rand}.wav"

def _gen_core(
    text, language, ref_audio, instruct, num_step, guidance_scale, 
    denoise, speed, duration, preprocess_prompt, postprocess_output, mode, ref_text=None
):
    if model is None:
        return None, "OmniVoice model is not loaded."
    if not text or not text.strip():
        return None, "Please enter the text to synthesize."

    if mode == "clone" and ref_audio and not ref_text and subtitle_maker is not None:
        try:
            whisper_lang = language if (language and language != "Auto") else None
            whisper_results = subtitle_maker(ref_audio, whisper_lang)
            if whisper_results and len(whisper_results) > 7:
                ref_text = whisper_results[7]
        except Exception as e:
            logging.warning(f"Fallback transcription failed: {e}")

    gen_config = OmniVoiceGenerationConfig(
        num_step=int(num_step or 32),
        guidance_scale=float(guidance_scale) if guidance_scale is not None else 2.0,
        denoise=bool(denoise) if denoise is not None else True,
        preprocess_prompt=bool(preprocess_prompt),
        postprocess_output=bool(postprocess_output),
    ) if OmniVoiceGenerationConfig is not None else None

    lang = language if (language and language != "Auto") else None
    kw: Dict[str, Any] = dict(text=text.strip(), language=lang)
    if gen_config is not None:
        kw["generation_config"] = gen_config

    if speed is not None and float(speed) != 1.0:
        kw["speed"] = float(speed)
    if duration is not None and float(duration) > 0:
        kw["duration"] = float(duration)

    if mode == "clone":
        if not ref_audio:
            return None, "Please upload a reference audio."
        kw["voice_clone_prompt"] = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=ref_text)
    if mode == "design":
        if instruct and instruct.strip():
            kw["instruct"] = instruct.strip()

    try:
        audio = model.generate(**kw)
    except Exception as e:
        return None, f"Error: {type(e).__name__}: {e}"

    wf = audio[0]
    if hasattr(wf, "cpu"):
        wf = wf.cpu().numpy()
    if hasattr(wf, "squeeze"):
        wf = wf.squeeze()
    waveform = (wf * 32767).astype(np.int16)
    return (sampling_rate, waveform), "Done."


# ---------------------------------------------------------------------------
# Funções de Controle do Estúdio PT-BR e APIs
# ---------------------------------------------------------------------------
def core_prepare_voice(ref_audio: str) -> Tuple[str, str, str]:
    """
    Prepara a voz clonada a partir do áudio de referência:
    1. Transcreve automaticamente com faster-whisper na CPU
    2. Cria o prompt de clonagem no OmniVoice
    3. Registra no cache com Voice ID
    Retorna: (voice_id, transcript, status)
    """
    if not ref_audio:
        return "", "", "Erro: Por favor, envie ou grave um áudio de referência (3 a 15s)."
    
    if model is None:
        return "", "", "Erro: Modelo OmniVoice não carregado na memória."

    try:
        # Transcrição real da referência (faster-whisper na CPU ou ASR do OmniVoice)
        voice_prompt, transcript = studio_core.prepare_voice_prompt(model, ref_audio)
        voice_id = studio_core.register_voice(voice_prompt, transcript, ref_audio)
        return voice_id, transcript, f"Voz preparada com sucesso! ID: {voice_id}"
    except Exception as e:
        logging.exception("Erro em core_prepare_voice")
        return "", "", f"Erro ao preparar voz: {e}"


def core_generate_tts(
    voice_id: str,
    text: str,
    speed: float = 1.0,
    num_step: int = 32,
    guidance_scale: float = 2.0,
    pause_ms: int = 140,
    subtitle_max_chars: int = 180,
) -> Tuple[Optional[Tuple[int, np.ndarray]], Optional[str], Optional[str], str]:
    """
    Gera fala usando o Voice ID previamente registrado, dividindo textos longos,
    calculando pausas e gerando MP3 a 192k e SRT perfeitamente sincronizado.
    """
    if not voice_id or not voice_id.strip():
        return None, None, None, "Erro: Voice ID não informado."
    
    voice_prompt = studio_core.get_cached_voice(voice_id)
    if voice_prompt is None:
        return None, None, None, f"Erro: Voice ID '{voice_id}' não encontrado no cache."

    if not text or not text.strip():
        return None, None, None, "Erro: Informe o texto a sintetizar."

    if model is None:
        return None, None, None, "Erro: Modelo OmniVoice não carregado."

    return studio_core.generate_blocks_tts(
        model=model,
        sampling_rate=sampling_rate,
        voice_prompt=voice_prompt,
        text=text,
        speed=speed,
        num_step=num_step,
        guidance_scale=guidance_scale,
        pause_ms=pause_ms,
        subtitle_max_chars=subtitle_max_chars,
        language="Portuguese",
        generation_config_cls=OmniVoiceGenerationConfig,
    )


def core_clone_and_generate(
    ref_audio: str,
    text: str,
    speed: float = 1.0,
    num_step: int = 32,
    guidance_scale: float = 2.0,
    pause_ms: int = 140,
    subtitle_max_chars: int = 180,
) -> Tuple[str, str, Optional[Tuple[int, np.ndarray]], Optional[str], Optional[str], str]:
    """
    Executa o fluxo completo em uma única chamada:
    Áudio -> Transcrição -> Voice ID -> Síntese por blocos -> MP3 + SRT
    """
    voice_id, transcript, prep_status = core_prepare_voice(ref_audio)
    if not voice_id:
        return "", "", None, None, None, prep_status

    audio_tuple, mp3_path, srt_path, gen_status = core_generate_tts(
        voice_id=voice_id,
        text=text,
        speed=speed,
        num_step=num_step,
        guidance_scale=guidance_scale,
        pause_ms=pause_ms,
        subtitle_max_chars=subtitle_max_chars,
    )

    return voice_id, transcript, audio_tuple, mp3_path, srt_path, gen_status


# Handlers específicos para a API pública do Gradio
def api_prepare_voice_fn(ref_audio):
    v_id, tr, status = core_prepare_voice(ref_audio)
    return v_id, tr, status

def api_generate_tts_fn(voice_id, text, speed, num_step, guidance_scale, pause_ms, subtitle_max_chars):
    _, mp3_path, srt_path, status = core_generate_tts(
        voice_id=voice_id,
        text=text,
        speed=speed,
        num_step=num_step,
        guidance_scale=guidance_scale,
        pause_ms=pause_ms,
        subtitle_max_chars=subtitle_max_chars,
    )
    return mp3_path, srt_path, status

def api_clone_and_generate_fn(ref_audio, text, speed, num_step, guidance_scale, pause_ms, subtitle_max_chars):
    v_id, tr, _, mp3_path, srt_path, status = core_clone_and_generate(
        ref_audio=ref_audio,
        text=text,
        speed=speed,
        num_step=num_step,
        guidance_scale=guidance_scale,
        pause_ms=pause_ms,
        subtitle_max_chars=subtitle_max_chars,
    )
    return v_id, tr, mp3_path, srt_path, status


# ---------------------------------------------------------------------------
# Gradio UI Construction
# ---------------------------------------------------------------------------
theme = gr.themes.Soft(font=["Inter", "Arial", "sans-serif"])
css = """
.gradio-container {max-width: 100% !important; font-size: 16px !important;}
.gradio-container h1 {font-size: 1.5em !important;}
.gradio-container .prose {font-size: 1.1em !important;}
.compact-audio audio {height: 60px !important;}
.compact-audio .waveform {min-height: 80px !important;}

/* Alerta de Segurança e Boas Práticas */
.ethics-banner {
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 5px solid #2563eb;
    border-radius: 8px;
    padding: 12px 18px;
    margin-bottom: 20px;
    font-size: 0.95em;
    color: #1e293b;
}

/* CSS for Event Tags */
.tag-container {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 8px !important;
    margin-top: 5px !important;
    margin-bottom: 10px !important;
    border: none !important;
    background: transparent !important;
}
.tag-btn {
    min-width: fit-content !important;
    width: auto !important;
    height: 32px !important;
    font-size: 13px !important;
    background: #eef2ff !important;
    border: 1px solid #c7d2fe !important;
    color: #3730a3 !important;
    border-radius: 6px !important;
    padding: 0 10px !important;
    margin: 0 !important;
    box-shadow: none !important;
}
.tag-btn:hover {
    background: #c7d2fe !important;
    transform: translateY(-1px);
}
"""

def _lang_dropdown(label="Language (optional)", value="Auto"):
    return gr.Dropdown(
        label=label, choices=_ALL_LANGUAGES, value=value,
        allow_custom_value=False, interactive=True,
    )

def _gen_settings():
    with gr.Accordion("Generation Settings (optional)", open=False):
        sp = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Speed", info="1.0 = normal. >1 faster, <1 slower.")
        du = gr.Number(value=None, label="Duration (seconds)", info="Set a fixed duration to override speed.")
        ns = gr.Slider(4, 64, value=32, step=1, label="Inference Steps", info="Lower = faster, higher = better quality.")
        dn = gr.Checkbox(label="Denoise", value=True)
        gs = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="Guidance Scale (CFG)")
        pp = gr.Checkbox(label="Preprocess Prompt", value=True, info="Applies silence removal and trims reference audio.")
        po = gr.Checkbox(label="Postprocess Output", value=True, info="Removes long silences from generated audio.")
    return ns, gs, dn, sp, du, pp, po


with gr.Blocks(theme=theme, css=css, title="OmniVoice Colab Studio") as demo:
    gr.HTML("""
        <div style="text-align: center; margin: 15px auto; max-width: 850px;">
            <h1 style="font-size: 2.2em; margin-bottom: 5px;">🎙️ OmniVoice Colab Studio</h1>
            <p style="color: #475569;">Clonagem de voz em Português Brasileiro, síntese por blocos para textos longos, exportação em MP3 (192 kbps), legendas SRT perfeitamente sincronizadas e API Gradio reutilizável.</p>
        </div>
        <div class="ethics-banner">
            🛡️ <b>Aviso Ético e de Responsabilidade:</b> Use clonagem de voz apenas com sua própria voz ou com autorização explícita da pessoa dona da voz. O link público <code>gradio.live</code> gerado é acessível publicamente enquanto a sessão do Colab permanecer ativa.
        </div>
    """)

    with gr.Tabs():
        # ==============================================================
        # Aba 1 — Estúdio PT-BR (Tudo em 1 etapa)
        # ==============================================================
        with gr.TabItem("Estúdio PT-BR (Tudo em 1)"):
            with gr.Row():
                with gr.Column(scale=1):
                    tab1_ref_audio = gr.Audio(
                        label="Amostra de Voz de Referência (3 a 15 segundos)",
                        type="filepath",
                        sources=["upload", "microphone"],
                        elem_classes="compact-audio",
                    )
                    tab1_text = gr.Textbox(
                        label="Texto / Roteiro para Síntese",
                        lines=5,
                        placeholder="Cole aqui seu texto ou roteiro completo. Textos longos serão divididos automaticamente em frases e blocos com legendas sincronizadas.",
                    )
                    with gr.Accordion("Configurações Avançadas", open=False):
                        tab1_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Velocidade (Speed)", info="1.0 = normal.")
                        tab1_num_step = gr.Slider(4, 64, value=32, step=1, label="Inference Steps", info="32 = padrão equilibrado.")
                        tab1_guidance_scale = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="Guidance Scale (CFG)", info="2.0 = recomendado.")
                        tab1_pause_ms = gr.Slider(0, 500, value=140, step=10, label="Pausa entre blocos (ms)", info="Silêncio inserido entre frases. Padrão: 140 ms.")
                        tab1_max_chars = gr.Slider(50, 300, value=180, step=10, label="Tamanho Máximo da Legenda (caracteres)", info="Padrão: 180 caracteres.")

                    tab1_btn = gr.Button("⚡ Gerar MP3 + SRT", variant="primary")

                with gr.Column(scale=1):
                    tab1_out_voice_id = gr.Textbox(label="Voice ID Gerado (Reutilizável)", lines=1)
                    tab1_out_transcript = gr.Textbox(label="Transcrição da Amostra (Whisper Automático)", lines=2)
                    tab1_out_audio = gr.Audio(label="Prévia do Áudio Gerado", type="numpy")
                    with gr.Row():
                        tab1_out_mp3 = gr.File(label="Áudio Final (MP3 192kbps)")
                        tab1_out_srt = gr.File(label="Legendas Sincronizadas (SRT)")
                    tab1_out_status = gr.Textbox(label="Status do Processamento", lines=1)

            def _ui_tab1_clone_and_generate(ref_aud, txt, spd, n_step, cfg, pause, max_c):
                v_id, tr, aud_tuple, mp3, srt, st = core_clone_and_generate(
                    ref_audio=ref_aud,
                    text=txt,
                    speed=spd,
                    num_step=n_step,
                    guidance_scale=cfg,
                    pause_ms=pause,
                    subtitle_max_chars=max_c,
                )
                return v_id, tr, aud_tuple, mp3, srt, st

            tab1_btn.click(
                fn=_ui_tab1_clone_and_generate,
                inputs=[tab1_ref_audio, tab1_text, tab1_speed, tab1_num_step, tab1_guidance_scale, tab1_pause_ms, tab1_max_chars],
                outputs=[tab1_out_voice_id, tab1_out_transcript, tab1_out_audio, tab1_out_mp3, tab1_out_srt, tab1_out_status],
            )

        # ==============================================================
        # Aba 2 — Estúdio PT-BR (Preparar Voz e Reutilizar)
        # ==============================================================
        with gr.TabItem("Estúdio PT-BR (Reutilizar Voice ID)"):
            with gr.Row():
                # Coluna Esquerda: Preparação da Voz
                with gr.Column(scale=1):
                    gr.Markdown("### 1️⃣ Preparar Voz de Referência")
                    tab2_prep_audio = gr.Audio(
                        label="Amostra de Voz (3 a 15 segundos)",
                        type="filepath",
                        sources=["upload", "microphone"],
                        elem_classes="compact-audio",
                    )
                    tab2_prep_btn = gr.Button("🎯 Preparar Voz (Obter Voice ID)", variant="secondary")
                    tab2_prep_voice_id = gr.Textbox(label="Voice ID Gerado", lines=1)
                    tab2_prep_transcript = gr.Textbox(label="Transcrição Detectada", lines=2)
                    tab2_prep_status = gr.Textbox(label="Status da Preparação", lines=1)

                # Coluna Direita: Síntese usando o Voice ID
                with gr.Column(scale=1):
                    gr.Markdown("### 2️⃣ Gerar Roteiros com Voice ID")
                    tab2_gen_voice_id = gr.Textbox(
                        label="Voice ID",
                        placeholder="Ex: voice_a82f71c922",
                        lines=1,
                    )
                    tab2_gen_text = gr.Textbox(
                        label="Texto / Roteiro",
                        lines=4,
                        placeholder="Insira o texto que esta voz deve narrar...",
                    )
                    with gr.Accordion("Configurações de Síntese", open=False):
                        tab2_speed = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Velocidade", info="1.0 = normal.")
                        tab2_num_step = gr.Slider(4, 64, value=32, step=1, label="Inference Steps", info="32 = padrão.")
                        tab2_guidance_scale = gr.Slider(0.0, 4.0, value=2.0, step=0.1, label="Guidance Scale (CFG)", info="2.0 = recomendado.")
                        tab2_pause_ms = gr.Slider(0, 500, value=140, step=10, label="Pausa entre blocos (ms)", info="Padrão: 140 ms.")
                        tab2_max_chars = gr.Slider(50, 300, value=180, step=10, label="Tamanho Máximo da Legenda", info="Padrão: 180 caracteres.")

                    tab2_gen_btn = gr.Button("🚀 Sintetizar com Voice ID", variant="primary")
                    tab2_gen_audio = gr.Audio(label="Prévia do Áudio", type="numpy")
                    with gr.Row():
                        tab2_gen_mp3 = gr.File(label="Arquivo MP3 (192kbps)")
                        tab2_gen_srt = gr.File(label="Legenda Sincronizada (SRT)")
                    tab2_gen_status = gr.Textbox(label="Status da Geração", lines=1)

            def _ui_tab2_prep(ref_aud):
                v_id, tr, st = core_prepare_voice(ref_aud)
                # Preenche automaticamente o campo de Voice ID na seção de geração
                return v_id, tr, st, v_id

            tab2_prep_btn.click(
                fn=_ui_tab2_prep,
                inputs=[tab2_prep_audio],
                outputs=[tab2_prep_voice_id, tab2_prep_transcript, tab2_prep_status, tab2_gen_voice_id],
            )

            def _ui_tab2_generate(v_id, txt, spd, n_step, cfg, pause, max_c):
                aud_tuple, mp3, srt, st = core_generate_tts(
                    voice_id=v_id,
                    text=txt,
                    speed=spd,
                    num_step=n_step,
                    guidance_scale=cfg,
                    pause_ms=pause,
                    subtitle_max_chars=max_c,
                )
                return aud_tuple, mp3, srt, st

            tab2_gen_btn.click(
                fn=_ui_tab2_generate,
                inputs=[tab2_gen_voice_id, tab2_gen_text, tab2_speed, tab2_num_step, tab2_guidance_scale, tab2_pause_ms, tab2_max_chars],
                outputs=[tab2_gen_audio, tab2_gen_mp3, tab2_gen_srt, tab2_gen_status],
            )

        # ==============================================================
        # Aba 3 — Voice Clone (Original)
        # ==============================================================
        with gr.TabItem("Voice Clone (Original)"):
            with gr.Row():
                with gr.Column(scale=1):
                    vc_text = gr.Textbox(label="Text to Synthesize", lines=4, placeholder="Enter the text to synthesize...", elem_id="vc_textbox")
                    
                    with gr.Row(elem_classes=["tag-container"]):
                        for tag in EVENT_TAGS:
                            btn = gr.Button(tag, elem_classes=["tag-btn"])
                            btn.click(
                                fn=None,
                                inputs=[btn, vc_text],
                                outputs=vc_text,
                                js=INSERT_TAG_JS_VC
                            )

                    with gr.Row():
                        vc_lang = _lang_dropdown("Language (optional)")
                        vc_want_subs = gr.Checkbox(label="Want Subtitles ?", value=False)
                    vc_ref_audio = gr.Audio(label="Reference Audio (3–10 seconds audio)", type="filepath", elem_classes="compact-audio")
                    
                    vc_ref_text = gr.Textbox(
                        label="Reference Text", lines=2, 
                        placeholder="Auto-transcribed upon audio upload. You can manually edit it if Whisper gets it wrong."
                    )
                                        
                    vc_btn = gr.Button("Generate", variant="primary")
                    vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po = _gen_settings()
                
                with gr.Column(scale=1):
                    vc_audio = gr.Audio(label="Output Audio", type="numpy")
                    vc_status = gr.Textbox(label="Status", lines=1)
                    
                    with gr.Accordion("Download files", open=False):
                        vc_out_wav = gr.File(label="Generated Audio (WAV)")
                        vc_out_custom_srt = gr.File(label="Sentence Level SRT")
                        vc_out_word_srt = gr.File(label="Word Level SRT")
                        vc_out_shorts_srt = gr.File(label="Shorts SRT")

            def _auto_transcribe(audio_path, lang):
                if not audio_path or subtitle_maker is None:
                    return gr.update(value="")
                try:
                    whisper_lang = lang if lang != "Auto" else None
                    whisper_results = subtitle_maker(audio_path, whisper_lang)
                    if whisper_results and len(whisper_results) > 7:
                        return gr.update(value=whisper_results[7])
                except Exception as e:
                    logging.warning(f"Auto-transcription failed: {e}")
                return gr.update(value="")

            vc_ref_audio.change(
                fn=_auto_transcribe,
                inputs=[vc_ref_audio, vc_lang],
                outputs=[vc_ref_text]
            )

            def _clone_fn(text, lang, ref_aud, ref_text, want_subs, ns, gs, dn, sp, du, pp, po):
                res = _gen_core(text, lang, ref_aud, None, ns, gs, dn, sp, du, pp, po, mode="clone", ref_text=ref_text)
                if res[0] is None:
                    return None, res[1], None, None, None, None

                audio_tuple, status = res
                sr, waveform = audio_tuple
                # Encurta silêncios antes do Whisper para o SRT seguir o áudio final.
                waveform = studio_core.compress_silences(waveform, sr)
                audio_tuple = (sr, waveform)
                tmp_wav = tts_file_name(text, language=lang)
                wavfile.write(tmp_wav, sr, waveform)
                
                c_srt, w_srt, s_srt = generate_subtitles_if_needed(tmp_wav, lang, want_subs)
                return audio_tuple, status, tmp_wav, c_srt, w_srt, s_srt

            vc_btn.click(
                _clone_fn,
                inputs=[vc_text, vc_lang, vc_ref_audio, vc_ref_text, vc_want_subs, vc_ns, vc_gs, vc_dn, vc_sp, vc_du, vc_pp, vc_po],
                outputs=[vc_audio, vc_status, vc_out_wav, vc_out_custom_srt, vc_out_word_srt, vc_out_shorts_srt],
            )

        # ==============================================================
        # Aba 4 — Voice Design (Original)
        # ==============================================================
        with gr.TabItem("Voice Design (Original)"):
            with gr.Row():
                with gr.Column(scale=1):
                    vd_text = gr.Textbox(label="Text to Synthesize", lines=4, placeholder="Enter the text to synthesize...", elem_id="vd_textbox")
                    
                    with gr.Row(elem_classes=["tag-container"]):
                        for tag in EVENT_TAGS:
                            btn = gr.Button(tag, elem_classes=["tag-btn"])
                            btn.click(
                                fn=None,
                                inputs=[btn, vd_text],
                                outputs=vd_text,
                                js=INSERT_TAG_JS_VD
                            )

                    with gr.Row():
                        vd_lang = _lang_dropdown(value='Auto')
                        vd_want_subs = gr.Checkbox(label="Want Subtitles ?", value=False)
                    vd_btn = gr.Button("Generate", variant="primary")
                    with gr.Accordion("Character Voice Design", open=False):
                        vd_groups = []
                        for _cat, _choices in _CATEGORIES.items():
                            default_val = "Auto"
                            if _cat == "Gender":
                                default_val = "Female"
                            elif _cat == "Age":
                                default_val = "Young Adult"
                            vd_groups.append(
                                gr.Dropdown(label=_cat, choices=["Auto"] + _choices, value=default_val, info=_ATTR_INFO.get(_cat))
                            )
                        
                    vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po = _gen_settings()
                
                with gr.Column(scale=1):
                    vd_audio = gr.Audio(label="Output Audio", type="numpy")
                    vd_status = gr.Textbox(label="Status", lines=1)
                    
                    with gr.Accordion("Download files", open=False):
                        vd_out_wav = gr.File(label="Generated Audio (WAV)")
                        vd_out_custom_srt = gr.File(label="Sentence Level SRT")
                        vd_out_word_srt = gr.File(label="Word Level SRT")
                        vd_out_shorts_srt = gr.File(label="Shorts SRT")

            def _build_instruct(groups):
                selected = [g for g in groups if g and g != "Auto"]
                if not selected: return None
                return ", ".join([DIALECT_MAP.get(v, v) for v in selected])

            def _design_fn(text, lang, want_subs, ns, gs, dn, sp, du, pp, po, *groups):
                instruct = _build_instruct(groups)
                res = _gen_core(text, lang, None, instruct, ns, gs, dn, sp, du, pp, po, mode="design")
                if res[0] is None:
                    return None, res[1], None, None, None, None
                
                audio_tuple, status = res
                sr, waveform = audio_tuple
                tmp_wav = tts_file_name(text, language=lang)
                wavfile.write(tmp_wav, sr, waveform)
                
                c_srt, w_srt, s_srt = generate_subtitles_if_needed(tmp_wav, lang, want_subs)
                return audio_tuple, status, tmp_wav, c_srt, w_srt, s_srt

            vd_btn.click(
                _design_fn,
                inputs=[vd_text, vd_lang, vd_want_subs, vd_ns, vd_gs, vd_dn, vd_sp, vd_du, vd_pp, vd_po] + vd_groups,
                outputs=[vd_audio, vd_status, vd_out_wav, vd_out_custom_srt, vd_out_word_srt, vd_out_shorts_srt],
            )

    # ==================================================================
    # Endpoints Públicos da API Gradio (Gradio Client)
    # ==================================================================
    with gr.Group(visible=False):
        # 1. Endpoint /prepare_voice
        # Entrada: ref_audio | Saída: voice_id, transcript, status
        api_prep_ref_audio = gr.Audio(type="filepath")
        api_prep_btn = gr.Button()
        api_prep_voice_id = gr.Textbox()
        api_prep_transcript = gr.Textbox()
        api_prep_status = gr.Textbox()

        api_prep_btn.click(
            fn=api_prepare_voice_fn,
            inputs=[api_prep_ref_audio],
            outputs=[api_prep_voice_id, api_prep_transcript, api_prep_status],
            api_name="prepare_voice",
        )

        # 2. Endpoint /generate_tts
        # Entrada: voice_id, text, speed, num_step, guidance_scale, pause_ms, subtitle_max_chars
        # Saída: mp3, srt, status
        api_gen_voice_id = gr.Textbox()
        api_gen_text = gr.Textbox()
        api_gen_speed = gr.Slider(0.5, 2.0, value=1.0)
        api_gen_num_step = gr.Slider(4, 64, value=32)
        api_gen_guidance_scale = gr.Slider(0.0, 4.0, value=2.0)
        api_gen_pause_ms = gr.Slider(0, 500, value=140)
        api_gen_max_chars = gr.Slider(50, 300, value=180)
        api_gen_btn = gr.Button()
        api_gen_mp3 = gr.File()
        api_gen_srt = gr.File()
        api_gen_status = gr.Textbox()

        api_gen_btn.click(
            fn=api_generate_tts_fn,
            inputs=[
                api_gen_voice_id, api_gen_text, api_gen_speed,
                api_gen_num_step, api_gen_guidance_scale,
                api_gen_pause_ms, api_gen_max_chars
            ],
            outputs=[api_gen_mp3, api_gen_srt, api_gen_status],
            api_name="generate_tts",
        )

        # 3. Endpoint /clone_and_generate
        # Entrada: ref_audio, text, speed, num_step, guidance_scale, pause_ms, subtitle_max_chars
        # Saída: voice_id, transcript, mp3, srt, status
        api_all_ref_audio = gr.Audio(type="filepath")
        api_all_text = gr.Textbox()
        api_all_speed = gr.Slider(0.5, 2.0, value=1.0)
        api_all_num_step = gr.Slider(4, 64, value=32)
        api_all_guidance_scale = gr.Slider(0.0, 4.0, value=2.0)
        api_all_pause_ms = gr.Slider(0, 500, value=140)
        api_all_max_chars = gr.Slider(50, 300, value=180)
        api_all_btn = gr.Button()
        api_all_voice_id = gr.Textbox()
        api_all_transcript = gr.Textbox()
        api_all_mp3 = gr.File()
        api_all_srt = gr.File()
        api_all_status = gr.Textbox()

        api_all_btn.click(
            fn=api_clone_and_generate_fn,
            inputs=[
                api_all_ref_audio, api_all_text, api_all_speed,
                api_all_num_step, api_all_guidance_scale,
                api_all_pause_ms, api_all_max_chars
            ],
            outputs=[api_all_voice_id, api_all_transcript, api_all_mp3, api_all_srt, api_all_status],
            api_name="clone_and_generate",
        )


if __name__ == "__main__":
    demo.queue(max_size=20).launch(share=True, show_api=True)
