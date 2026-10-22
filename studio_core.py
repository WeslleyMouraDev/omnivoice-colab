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


# Fala natural em português fica perto de 12 a 16 caracteres por segundo. Abaixo
# deste limite a transcrição não descreve o áudio de referência, e o OmniVoice
# estimaria durações várias vezes maiores para cada frase (fala esticada).
MIN_REFERENCE_CHARS_PER_SECOND = 4.0


def _reference_seconds(model: Any, voice_prompt: Any) -> Optional[float]:
    try:
        tokens = voice_prompt.ref_audio_tokens
        chunks = tokens if isinstance(tokens, (list, tuple)) else [tokens]
        frames = sum(chunk.shape[-1] for chunk in chunks)
        return float(frames) / float(model.audio_tokenizer.config.frame_rate)
    except Exception:  # noqa: BLE001 - sem metadados, a checagem é ignorada
        return None


def prepare_voice_prompt(model: Any, ref_audio: str, transcribe=None) -> Tuple[Any, str]:
    """
    Cria o prompt de clonagem com a transcrição real do áudio de referência.
    1. Tenta o faster-whisper na CPU.
    2. Se ele falhar, deixa o OmniVoice transcrever com o ASR embutido (GPU).
    3. Recusa transcrições incompatíveis com a duração do áudio, porque elas
       fazem o modelo esticar a fala de todo o roteiro.
    Nunca usa um texto genérico no lugar da transcrição.
    """
    transcribe = transcribe or transcribe_reference_audio
    transcript = (transcribe(ref_audio, lang="pt") or "").strip()
    if transcript:
        voice_prompt = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=transcript)
    else:
        logger.warning("faster-whisper não transcreveu a referência; usando o ASR do OmniVoice.")
        voice_prompt = model.create_voice_clone_prompt(ref_audio=ref_audio, ref_text=None)
        transcript = str(getattr(voice_prompt, "ref_text", "") or "").strip()
    if not transcript:
        raise ValueError("Não foi possível transcrever o áudio de referência.")
    seconds = _reference_seconds(model, voice_prompt)
    if seconds and len(transcript) / seconds < MIN_REFERENCE_CHARS_PER_SECOND:
        raise ValueError(
            f"A transcrição do áudio de referência tem {len(transcript)} caracteres para "
            f"{seconds:.1f} s de fala e parece incompleta: \"{transcript}\". "
            "Use um áudio com fala contínua de 5 a 15 segundos."
        )
    return voice_prompt, transcript


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
# Remoção de silêncio do áudio final
# ---------------------------------------------------------------------------
MAX_PAUSE_MS = int(os.environ.get("OMNIVOICE_MAX_PAUSE_MS", "300") or 300)
EDGE_SILENCE_MS = int(os.environ.get("OMNIVOICE_EDGE_SILENCE_MS", "80") or 80)
SILENCE_THRESHOLD_DB = float(os.environ.get("OMNIVOICE_SILENCE_THRESHOLD_DB", "-42") or -42)
# No modo Studio a pausa entre blocos já é inserida por pause_ms.
STUDIO_BLOCK_EDGE_MS = int(os.environ.get("OMNIVOICE_STUDIO_BLOCK_EDGE_MS", "20") or 20)


def compress_silences(
    waveform: np.ndarray,
    sampling_rate: int,
    max_pause_ms: Optional[int] = None,
    edge_ms: Optional[int] = None,
    threshold_db: Optional[float] = None,
    frame_ms: int = 20,
) -> np.ndarray:
    """
    Encurta silêncios do áudio final sem cortar fala:
    - pausas internas maiores que max_pause_ms ficam com max_pause_ms
      (metade preservada no fim da fala anterior e metade antes da próxima);
    - silêncio no início e no fim fica com edge_ms.
    Deve rodar antes de gerar o SRT, para as legendas acompanharem o áudio.
    """
    max_pause_ms = MAX_PAUSE_MS if max_pause_ms is None else max_pause_ms
    edge_ms = EDGE_SILENCE_MS if edge_ms is None else edge_ms
    threshold_db = SILENCE_THRESHOLD_DB if threshold_db is None else threshold_db
    samples = np.asarray(waveform)
    if samples.ndim != 1 or samples.size == 0:
        return samples

    frame = max(1, int(sampling_rate * frame_ms / 1000))
    frames = samples.size // frame
    if frames == 0:
        return samples
    scale = 32768.0 if samples.dtype == np.int16 else 1.0
    body = samples[: frames * frame].astype(np.float64).reshape(frames, frame) / scale
    rms = np.sqrt(np.mean(body * body, axis=1))
    silent = 20 * np.log10(np.maximum(rms, 1e-10)) < threshold_db
    if silent.all():
        return samples[:0]

    keep_pause = int(sampling_rate * max_pause_ms / 1000)
    keep_edge = int(sampling_rate * edge_ms / 1000)
    pieces: List[np.ndarray] = []
    index = 0
    while index < frames:
        end = index
        while end < frames and silent[end] == silent[index]:
            end += 1
        start_sample = index * frame
        end_sample = end * frame if end < frames else samples.size
        segment = samples[start_sample:end_sample]
        if not silent[index]:
            pieces.append(segment)
        elif index == 0:
            pieces.append(segment[max(0, segment.size - keep_edge):])
        elif end >= frames:
            pieces.append(segment[:keep_edge])
        elif segment.size > keep_pause:
            half = keep_pause // 2
            pieces.append(segment[:half])
            pieces.append(segment[segment.size - (keep_pause - half):])
        else:
            pieces.append(segment)
        index = end
    return np.concatenate(pieces)


# ---------------------------------------------------------------------------
# Síntese em lotes na GPU
# ---------------------------------------------------------------------------
DEFAULT_TTS_BATCH_SIZE = int(os.environ.get("OMNIVOICE_TTS_BATCH_SIZE", "8") or 8)


def _to_int16(waveform: Any) -> np.ndarray:
    if hasattr(waveform, "cpu"):
        waveform = waveform.cpu().numpy()
    if hasattr(waveform, "squeeze"):
        waveform = waveform.squeeze()
    if waveform.dtype != np.int16:
        waveform = (waveform * 32767).astype(np.int16)
    return waveform


def _is_out_of_memory(error: Exception) -> bool:
    return "out of memory" in str(error).lower()


def _release_gpu_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - liberar memória é apenas uma tentativa
        pass


def _synthesize_in_batches(
    model: Any,
    blocks: List[str],
    language: str,
    voice_prompt: Any,
    gen_config: Any,
    speed: Optional[float],
    batch_size: Optional[int],
) -> List[np.ndarray]:
    """
    Gera vários blocos por chamada de model.generate, que processa a lista em
    paralelo na GPU. A ordem dos blocos é preservada. Se a GPU ficar sem memória,
    o lote é reduzido pela metade e o mesmo trecho é refeito.
    """
    size = max(1, int(batch_size or DEFAULT_TTS_BATCH_SIZE))
    audios: List[np.ndarray] = []
    start = 0
    while start < len(blocks):
        chunk = [block.strip() for block in blocks[start:start + size]]
        gen_kwargs: Dict[str, Any] = {
            "text": chunk,
            "language": language,
            "voice_clone_prompt": voice_prompt,
        }
        if gen_config is not None:
            gen_kwargs["generation_config"] = gen_config
        if speed is not None and float(speed) != 1.0:
            gen_kwargs["speed"] = float(speed)
        try:
            outputs = model.generate(**gen_kwargs)
        except Exception as error:  # noqa: BLE001 - OOM vira lote menor
            if _is_out_of_memory(error) and size > 1:
                _release_gpu_memory()
                size = max(1, size // 2)
                logger.warning(f"GPU sem memória; reduzindo o lote para {size} blocos.")
                continue
            raise RuntimeError(f"bloco {start + 1}/{len(blocks)}: {error}") from error
        if len(outputs) != len(chunk):
            raise RuntimeError(
                f"o modelo devolveu {len(outputs)} áudios para {len(chunk)} blocos a partir do bloco {start + 1}"
            )
        audios.extend(_to_int16(output) for output in outputs)
        logger.info(f"Blocos sintetizados: {min(start + len(chunk), len(blocks))}/{len(blocks)}")
        start += len(chunk)
    return audios


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
    batch_size: Optional[int] = None,
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

    try:
        block_audios = _synthesize_in_batches(
            model, blocks, language, voice_prompt, gen_config, speed, batch_size
        )
    except Exception as e:
        return None, None, None, f"Erro ao sintetizar os blocos: {e}"

    for idx, (block_text, wf) in enumerate(zip(blocks, block_audios), start=1):
        # Silêncio removido antes de medir o bloco: o SRT usa a duração já encurtada.
        wf = compress_silences(wf, sampling_rate, edge_ms=STUDIO_BLOCK_EDGE_MS)
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
