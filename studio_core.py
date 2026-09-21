"""
studio_core.py - Núcleo do OmniVoice Colab Studio
Responsável por:
- Divisão inteligente de roteiros longos para legendas (split_for_subtitles)
- Transcrição automática do áudio de referência com faster-whisper (CPU, int8)
- Sistema de cache de vozes (Voice ID temporário)
- Síntese de fala por blocos individuais com pausas controláveis
- Cálculo exato de timestamps para geração de SRT sincronizado com o áudio final
- Exportação de áudio em MP3 a 192 kbps
"""

import os
import re
import uuid
import time
import logging
import numpy as np
import scipy.io.wavfile as wavfile
from typing import Dict, Any, List, Tuple, Optional

logger = logging.getLogger("omnivoice.studio")

# ---------------------------------------------------------------------------
# Estruturas de Cache em Memória para Vozes
# ---------------------------------------------------------------------------
VOICE_CACHE: Dict[str, Any] = {}
VOICE_META: Dict[str, Dict[str, Any]] = {}

STUDIO_OUTPUT_DIR = "./Omni_Studio_Output"
os.makedirs(STUDIO_OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Lazy-loading do Whisper (CPU / int8 para economizar VRAM)
# ---------------------------------------------------------------------------
_whisper_model = None

def get_whisper_cpu_model(model_size: str = "small"):
    """
    Carrega o faster-whisper na CPU com compute_type='int8' para evitar
    consumir VRAM que deve ficar dedicada ao OmniVoice na GPU.
    """
    global _whisper_model
    if _whisper_model is None:
        try:
            from faster_whisper import WhisperModel
            logger.info("Inicializando faster-whisper (small, cpu, int8)...")
            _whisper_model = WhisperModel(model_size, device="cpu", compute_type="int8")
        except Exception as e:
            logger.error(f"Falha ao carregar faster-whisper na CPU: {e}")
            raise e
    return _whisper_model

def transcribe_reference_audio(audio_path: str, lang: str = "pt") -> str:
    """
    Transcreve o áudio de referência de 3 a 15 segundos usando faster-whisper na CPU.
    """
    if not audio_path or not os.path.exists(audio_path):
        return ""
    try:
        model = get_whisper_cpu_model("small")
        whisper_lang = lang if lang and lang.lower() != "auto" else None
        segments, _ = model.transcribe(audio_path, language=whisper_lang, beam_size=5)
        text_parts = [seg.text.strip() for seg in segments if seg.text and seg.text.strip()]
        return " ".join(text_parts).strip()
    except Exception as e:
        logger.warning(f"Erro na transcrição do áudio de referência: {e}")
        return ""


# ---------------------------------------------------------------------------
# Divisão Inteligente de Texto para Legendas e Síntese
# ---------------------------------------------------------------------------
def _split_into_atomic_units(text: str, max_chars: int) -> List[str]:
    """
    Divide recursivamente o texto em fragmentos menores que max_chars,
    seguindo a prioridade:
    1. Pontuação final de frases (. ? ! \n)
    2. Pausas fortes (: ; — –)
    3. Vírgulas (,)
    4. Limites de palavras (espaço)
    Nunca corta palavras ao meio (a menos que a palavra seja maior que max_chars).
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # Nível 1: Quebras de linha e pontuação de fim de frase
    pattern_level1 = r'(?<=[.!?\n])\s+'
    chunks = [c.strip() for c in re.split(pattern_level1, text) if c.strip()]
    if len(chunks) > 1 and all(len(c) < len(text) for c in chunks):
        result = []
        for c in chunks:
            result.extend(_split_into_atomic_units(c, max_chars))
        return result

    # Nível 2: Pausas fortes (dois-pontos, ponto e vírgula, travessões)
    pattern_level2 = r'(?<=[;:—–])\s+'
    chunks = [c.strip() for c in re.split(pattern_level2, text) if c.strip()]
    if len(chunks) > 1 and all(len(c) < len(text) for c in chunks):
        result = []
        for c in chunks:
            result.extend(_split_into_atomic_units(c, max_chars))
        return result

    # Nível 3: Vírgulas
    pattern_level3 = r'(?<=,)\s+'
    chunks = [c.strip() for c in re.split(pattern_level3, text) if c.strip()]
    if len(chunks) > 1 and all(len(c) < len(text) for c in chunks):
        result = []
        for c in chunks:
            result.extend(_split_into_atomic_units(c, max_chars))
        return result

    # Nível 4: Espaços (palavras)
    words = text.split()
    if len(words) <= 1:
        # Palavra individual muito longa
        return [text]

    result = []
    current_chunk = []
    current_len = 0
    for w in words:
        w_len = len(w) + (1 if current_chunk else 0)
        if current_chunk and (current_len + w_len > max_chars):
            result.append(" ".join(current_chunk))
            current_chunk = [w]
            current_len = len(w)
        else:
            current_chunk.append(w)
            current_len += w_len
    if current_chunk:
        result.append(" ".join(current_chunk))
    return result


def split_for_subtitles(text: str, max_chars: int = 180) -> List[str]:
    """
    Divide roteiros de qualquer tamanho em blocos inteligentes para legendas
    e síntese por blocos no OmniVoice.
    Garante que cada bloco não ultrapasse max_chars (respeitando palavras inteiras)
    e agrupa fragmentos compatíveis de forma natural.
    """
    if not text or not text.strip():
        return []

    # Normalizar quebras de linha
    normalized = re.sub(r'\r\n|\r', '\n', text.strip())
    paragraphs = [p.strip() for p in normalized.split('\n\n') if p.strip()]

    final_blocks = []

    for para in paragraphs:
        # Se o parágrafo contém quebras de linha simples, divide nelas primeiro
        sub_paras = [sp.strip() for sp in para.split('\n') if sp.strip()]
        for sub_p in sub_paras:
            atomic_units = _split_into_atomic_units(sub_p, max_chars)

            # Agrupar atomic_units até atingir o limite max_chars sem estourar
            current_block = []
            current_len = 0

            for unit in atomic_units:
                added_len = len(unit) + (1 if current_block else 0)
                if current_block and (current_len + added_len > max_chars):
                    final_blocks.append(" ".join(current_block))
                    current_block = [unit]
                    current_len = len(unit)
                else:
                    current_block.append(unit)
                    current_len += added_len

            if current_block:
                final_blocks.append(" ".join(current_block))

    return final_blocks


# ---------------------------------------------------------------------------
# Formatação e Criação de Legendas SRT
# ---------------------------------------------------------------------------
def format_srt_timestamp(seconds: float) -> str:
    """
    Converte segundos com fração para formato de timestamp SRT padrão (HH:MM:SS,mmm).
    """
    total_millis = int(round(seconds * 1000))
    hours = total_millis // 3600000
    remainder = total_millis % 3600000
    minutes = remainder // 60000
    remainder = remainder % 60000
    secs = remainder // 1000
    millis = remainder % 1000

    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def build_srt_file(subtitles: List[Dict[str, Any]], output_path: str) -> str:
    """
    Gera um arquivo .srt a partir da lista de segmentos sincronizados:
    [{ "index": 1, "start": 0.0, "end": 3.42, "text": "Texto original" }, ...]
    """
    lines = []
    for sub in subtitles:
        start_str = format_srt_timestamp(sub["start"])
        end_str = format_srt_timestamp(sub["end"])
        idx = sub["index"]
        txt = sub["text"].strip()
        lines.append(f"{idx}\n{start_str} --> {end_str}\n{txt}\n")

    content = "\n".join(lines).strip() + "\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    return output_path


# ---------------------------------------------------------------------------
# Gestão de Voice ID
# ---------------------------------------------------------------------------
def register_voice(voice_prompt: Any, transcript: str, ref_audio_path: str) -> str:
    """
    Armazena o prompt de clonagem no cache de vozes temporário e retorna um voice_id único.
    """
    voice_id = f"voice_{uuid.uuid4().hex[:10]}"
    VOICE_CACHE[voice_id] = voice_prompt
    VOICE_META[voice_id] = {
        "transcript": transcript,
        "reference_audio": ref_audio_path,
        "created_at": time.time(),
    }
    return voice_id


def get_cached_voice(voice_id: str) -> Optional[Any]:
    """Retorna o prompt de voz armazenado ou None se não existir."""
    if not voice_id:
        return None
    return VOICE_CACHE.get(voice_id.strip())


# ---------------------------------------------------------------------------
# Conversão para MP3 a 192 kbps
# ---------------------------------------------------------------------------
def convert_wav_to_mp3(wav_path: str, mp3_path: str, bitrate: str = "192k") -> str:
    """
    Converte um arquivo WAV para MP3 na taxa de 192 kbps utilizando pydub.
    """
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_wav(wav_path)
        audio.export(mp3_path, format="mp3", bitrate=bitrate)
        return mp3_path
    except ImportError:
        logger.warning("pydub não instalado. O arquivo WAV será mantido como saída.")
        import shutil
        shutil.copyfile(wav_path, mp3_path)
        return mp3_path
    except Exception as e:
        logger.warning(f"Erro ao converter para MP3 com pydub: {e}. O arquivo WAV será mantido como saída.")
        import shutil
        shutil.copyfile(wav_path, mp3_path)
        return mp3_path


# ---------------------------------------------------------------------------
# Síntese Completa por Blocos com SRT e MP3
# ---------------------------------------------------------------------------
def generate_blocks_tts(
    model: Any,
    sampling_rate: int,
    voice_prompt: Any,
    text: str,
    speed: float = 1.0,
    num_step: int = 32,
    guidance_scale: float = 2.0,
    pause_ms: int = 140,
    subtitle_max_chars: int = 180,
    language: str = "Portuguese",
    generation_config_cls: Any = None,
) -> Tuple[Optional[Tuple[int, np.ndarray]], Optional[str], Optional[str], str]:
    """
    Executa a síntese de fala bloco a bloco:
    1. Divide o texto em blocos inteligentes usando split_for_subtitles
    2. Sintetiza cada bloco individualmente com o OmniVoice
    3. Mede a duração real de cada áudio em amostras
    4. Concatena os áudios inserindo o silêncio exato definido por pause_ms
    5. Constrói os timestamps do SRT baseando-se nas durações reais acumuladas
    6. Salva o áudio final montado em WAV e exporta para MP3 a 192 kbps
    7. Retorna ((sampling_rate, waveform), mp3_path, srt_path, status)
    """
    blocks = split_for_subtitles(text, max_chars=int(subtitle_max_chars or 180))
    if not blocks:
        return None, None, None, "Texto vazio ou inválido para síntese."

    # Configuração de geração do OmniVoice
    cfg_kwargs = {
        "num_step": int(num_step or 32),
        "guidance_scale": float(guidance_scale) if guidance_scale is not None else 2.0,
        "denoise": True,
        "preprocess_prompt": True,
        "postprocess_output": True,
    }
    
    if generation_config_cls is not None:
        try:
            gen_config = generation_config_cls(**cfg_kwargs, pad_duration=0.08, fade_duration=0.03)
        except TypeError:
            gen_config = generation_config_cls(**cfg_kwargs)
    else:
        gen_config = None

    pause_sec = max(0.0, float(pause_ms or 0) / 1000.0)
    pause_samples = int(round(pause_sec * sampling_rate))
    silence_array = np.zeros(pause_samples, dtype=np.int16) if pause_samples > 0 else np.array([], dtype=np.int16)

    waveforms: List[np.ndarray] = []
    subtitles: List[Dict[str, Any]] = []
    cursor_samples = 0

    for idx, block_text in enumerate(blocks, start=1):
        gen_kwargs: Dict[str, Any] = {
            "text": block_text.strip(),
            "language": language,
            "voice_clone_prompt": voice_prompt,
        }
        if gen_config is not None:
            gen_kwargs["generation_config"] = gen_config
        if speed is not None and float(speed) != 1.0:
            gen_kwargs["speed"] = float(speed)

        try:
            audio_out = model.generate(**gen_kwargs)
        except Exception as e:
            return None, None, None, f"Erro ao sintetizar bloco {idx}/{len(blocks)}: {e}"

        # Tratar tensor / ndarray
        wf = audio_out[0]
        if hasattr(wf, "cpu"):
            wf = wf.cpu().numpy()
        if hasattr(wf, "squeeze"):
            wf = wf.squeeze()
        
        # Converter para int16
        if wf.dtype != np.int16:
            wf = (wf * 32767).astype(np.int16)

        block_len = len(wf)
        start_sec = cursor_samples / float(sampling_rate)
        end_sec = (cursor_samples + block_len) / float(sampling_rate)

        subtitles.append({
            "index": idx,
            "start": start_sec,
            "end": end_sec,
            "text": block_text.strip(),
        })

        waveforms.append(wf)
        cursor_samples += block_len

        # Adicionar pausa entre blocos (exceto após o último bloco)
        if idx < len(blocks) and pause_samples > 0:
            waveforms.append(silence_array)
            cursor_samples += pause_samples

    if not waveforms:
        return None, None, None, "Nenhum áudio gerado."

    full_waveform = np.concatenate(waveforms)

    # Nomes únicos de saída
    session_id = uuid.uuid4().hex[:8]
    wav_path = os.path.join(STUDIO_OUTPUT_DIR, f"tts_{session_id}.wav")
    mp3_path = os.path.join(STUDIO_OUTPUT_DIR, f"tts_{session_id}.mp3")
    srt_path = os.path.join(STUDIO_OUTPUT_DIR, f"tts_{session_id}.srt")

    # Salvar WAV
    wavfile.write(wav_path, sampling_rate, full_waveform)

    # Salvar SRT
    build_srt_file(subtitles, srt_path)

    # Converter para MP3 192k
    try:
        convert_wav_to_mp3(wav_path, mp3_path, bitrate="192k")
    except Exception as e:
        logger.warning(f"Falha ao exportar MP3 com pydub: {e}. Usando WAV como fallback.")
        mp3_path = wav_path

    status = f"Sucesso! {len(blocks)} blocos gerados. Duração total: {format_srt_timestamp(cursor_samples / sampling_rate)}."
    return (sampling_rate, full_waveform), mp3_path, srt_path, status
