import os
import unittest
import numpy as np
import tempfile
from studio_core import (
    split_for_subtitles,
    format_srt_timestamp,
    build_srt_file,
    register_voice,
    get_cached_voice,
    generate_blocks_tts,
    prepare_voice_prompt,
    compress_silences,
    VOICE_CACHE,
    VOICE_META,
)

class TestStudioCore(unittest.TestCase):

    def test_split_for_subtitles_basic(self):
        text = "Hoje vamos entender como a inteligência artificial funciona."
        blocks = split_for_subtitles(text, max_chars=180)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0], text)

    def test_split_for_subtitles_long_text(self):
        text = (
            "A inteligência artificial está transformando a criação de conteúdo em vídeo no mundo inteiro. "
            "Ferramentas modernas de síntese de voz permitem gerar locuções perfeitas em português brasileiro, "
            "com alta fidelidade emocional e entonação natural. "
            "Além disso, a sincronização precisa de legendas no formato SRT economiza horas de edição manual no CapCut, "
            "tornando o fluxo de produção de vídeos curtos e longos extremamente ágil e produtivo."
        )
        blocks = split_for_subtitles(text, max_chars=120)
        self.assertTrue(len(blocks) > 1)
        for b in blocks:
            self.assertLessEqual(len(b), 120)
            self.assertTrue(len(b) > 0)
        
        # Garantir que todas as palavras originais estão presentes
        original_words = set(text.replace(".", "").replace(",", "").split())
        reconstructed = " ".join(blocks)
        reconstructed_words = set(reconstructed.replace(".", "").replace(",", "").split())
        self.assertEqual(original_words, reconstructed_words)

    def test_split_for_subtitles_respects_paragraphs(self):
        text = "Primeiro parágrafo curto.\n\nSegundo parágrafo curto."
        blocks = split_for_subtitles(text, max_chars=180)
        self.assertEqual(len(blocks), 2)
        self.assertEqual(blocks[0], "Primeiro parágrafo curto.")
        self.assertEqual(blocks[1], "Segundo parágrafo curto.")

    def test_format_srt_timestamp(self):
        self.assertEqual(format_srt_timestamp(0.0), "00:00:00,000")
        self.assertEqual(format_srt_timestamp(1.5), "00:00:01,500")
        self.assertEqual(format_srt_timestamp(63.456), "00:01:03,456")
        self.assertEqual(format_srt_timestamp(3661.025), "01:01:01,025")

    def test_build_srt_file(self):
        subtitles = [
            {"index": 1, "start": 0.0, "end": 2.5, "text": "Primeira frase da legenda."},
            {"index": 2, "start": 2.64, "end": 5.12, "text": "Segunda frase sincronizada."},
        ]
        with tempfile.NamedTemporaryFile(suffix=".srt", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            build_srt_file(subtitles, tmp_path)
            with open(tmp_path, "r", encoding="utf-8") as f:
                content = f.read()
            expected = (
                "1\n00:00:00,000 --> 00:00:02,500\nPrimeira frase da legenda.\n\n"
                "2\n00:00:02,640 --> 00:00:05,120\nSegunda frase sincronizada."
            )
            self.assertIn("00:00:00,000 --> 00:00:02,500", content)
            self.assertIn("00:00:02,640 --> 00:00:05,120", content)
            self.assertIn("Primeira frase da legenda.", content)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_voice_cache(self):
        mock_prompt = {"embedding": [0.1, 0.2, 0.3]}
        voice_id = register_voice(mock_prompt, "Transcrição de teste", "audio.wav")
        self.assertTrue(voice_id.startswith("voice_"))
        cached = get_cached_voice(voice_id)
        self.assertEqual(cached, mock_prompt)
        self.assertIn(voice_id, VOICE_META)
        self.assertEqual(VOICE_META[voice_id]["transcript"], "Transcrição de teste")

    def test_generate_blocks_tts_mock(self):
        # Mock do modelo OmniVoice
        class MockModel:
            def generate(self, **kwargs):
                # Retorna 1 segundo de seno a 24000Hz como int16 para cada texto
                sr = 24000
                t = np.linspace(0, 1.0, sr, endpoint=False)
                waveform = (np.sin(2 * np.pi * 440 * t) * 10000).astype(np.int16)
                texts = kwargs["text"] if isinstance(kwargs["text"], list) else [kwargs["text"]]
                return [waveform for _ in texts]

        mock_model = MockModel()
        sampling_rate = 24000
        text = "Bloco um de teste. Bloco dois de teste."
        audio_tuple, mp3_path, srt_path, status = generate_blocks_tts(
            model=mock_model,
            sampling_rate=sampling_rate,
            voice_prompt="dummy_prompt",
            text=text,
            speed=1.0,
            num_step=32,
            pause_ms=200,
            subtitle_max_chars=30, # Forçar divisão em 2 blocos
        )

        self.assertIsNotNone(audio_tuple)
        sr, waveform = audio_tuple
        self.assertEqual(sr, sampling_rate)
        
        # 2 blocos de 1 segundo (24000 amostras cada) + 1 pausa de 200ms (4800 amostras)
        # Total esperado = 24000 + 4800 + 24000 = 52800 amostras
        self.assertEqual(len(waveform), 24000 + 4800 + 24000)
        self.assertTrue(os.path.exists(srt_path))
        self.assertTrue(os.path.exists(mp3_path))
        self.assertIn("Sucesso!", status)

        # Validar conteúdo do SRT
        with open(srt_path, "r", encoding="utf-8") as f:
            srt_content = f.read()
        self.assertIn("1\n00:00:00,000 --> 00:00:01,000", srt_content)
        # Bloco 2 deve iniciar exatamente em 1s + 200ms = 1.2s -> 00:00:01,200
        self.assertIn("2\n00:00:01,200 --> 00:00:02,200", srt_content)

    def test_compress_silences_shortens_pauses_and_edges_but_keeps_speech(self):
        sr = 1000
        speech = (np.sin(np.arange(500) / 3.0) * 12000).astype(np.int16)
        silence = lambda ms: np.zeros(ms, dtype=np.int16)
        audio = np.concatenate([silence(900), speech, silence(1500), speech, silence(200), speech, silence(1200)])
        out = compress_silences(audio, sr, max_pause_ms=300, edge_ms=80, threshold_db=-42, frame_ms=20)
        self.assertEqual(len(out), 80 + 500 + 300 + 500 + 200 + 500 + 80)
        speech_samples = np.count_nonzero(np.abs(out) > 1000)
        self.assertEqual(speech_samples, 3 * np.count_nonzero(np.abs(speech) > 1000))

    def test_generate_blocks_tts_removes_block_silence_and_keeps_srt_in_sync(self):
        sr = 1000
        speech = (np.sin(np.arange(500) / 3.0) * 12000).astype(np.int16)
        silence = lambda ms: np.zeros(ms, dtype=np.int16)
        noisy_block = np.concatenate([silence(400), speech, silence(900), speech, silence(600)])

        class SilentEdgesModel:
            def generate(self, **kwargs):
                return [noisy_block for _ in kwargs["text"]]

        audio_tuple, _, srt_path, status = generate_blocks_tts(
            model=SilentEdgesModel(),
            sampling_rate=sr,
            voice_prompt="dummy_prompt",
            text="Primeira frase.\n\nSegunda frase.",
            pause_ms=140,
            subtitle_max_chars=180,
        )
        self.assertIn("Sucesso!", status)
        block = 20 + 500 + 300 + 500 + 20
        self.assertEqual(len(audio_tuple[1]), block + 140 + block)
        with open(srt_path, "r", encoding="utf-8") as f:
            srt_content = f.read()
        self.assertIn(f"00:00:00,000 --> {format_srt_timestamp(block / sr)}", srt_content)
        second_start = (block + 140) / sr
        self.assertIn(
            f"{format_srt_timestamp(second_start)} --> {format_srt_timestamp(second_start + block / sr)}",
            srt_content,
        )

    def test_compress_silences_keeps_audio_without_long_silences(self):
        sr = 1000
        speech = (np.sin(np.arange(500) / 3.0) * 12000).astype(np.int16)
        audio = np.concatenate([speech, np.zeros(100, dtype=np.int16), speech])
        out = compress_silences(audio, sr, max_pause_ms=300, edge_ms=80, frame_ms=20)
        self.assertTrue(np.array_equal(out, audio))

    def _voice_model(self, asr_text, seconds=12.75, frame_rate=25):
        calls = []

        class Prompt:
            def __init__(self, ref_text):
                self.ref_text = ref_text
                self.ref_audio_tokens = np.zeros((8, int(seconds * frame_rate)))

        class VoiceModel:
            class audio_tokenizer:
                class config:
                    pass

            def create_voice_clone_prompt(self, ref_audio, ref_text=None):
                calls.append(ref_text)
                return Prompt(ref_text if ref_text is not None else asr_text)

        VoiceModel.audio_tokenizer.config.frame_rate = frame_rate
        return VoiceModel(), calls

    def test_prepare_voice_prompt_uses_omnivoice_asr_when_whisper_fails(self):
        real = "Oi, eu sou o criador do canal e hoje vou te mostrar como o salário some antes do dia trinta chegar."
        model, calls = self._voice_model(real)
        prompt, transcript = prepare_voice_prompt(model, "voz.mp3", transcribe=lambda *_a, **_k: "")
        self.assertEqual(calls, [None], "sem transcrição, o OmniVoice deve transcrever a referência")
        self.assertEqual(transcript, real)
        self.assertEqual(prompt.ref_text, real)

    def test_prepare_voice_prompt_never_uses_generic_placeholder(self):
        model, _ = self._voice_model("Áudio de referência")
        with self.assertRaises(ValueError):
            prepare_voice_prompt(model, "voz.mp3", transcribe=lambda *_a, **_k: "")

    def test_prepare_voice_prompt_keeps_whisper_transcript(self):
        real = "Oi, eu sou o criador do canal e hoje vou te mostrar como o salário some antes do dia trinta chegar."
        model, calls = self._voice_model("não usado")
        _, transcript = prepare_voice_prompt(model, "voz.mp3", transcribe=lambda *_a, **_k: real)
        self.assertEqual(calls, [real])
        self.assertEqual(transcript, real)

    def _sized_model(self, fail_above=None):
        # Cada bloco vira um áudio com tamanho proporcional ao texto, para
        # verificar ordem e timestamps depois da síntese em lotes.
        calls = []

        class SizedModel:
            def generate(self, **kwargs):
                texts = kwargs["text"]
                calls.append(list(texts))
                if fail_above is not None and len(texts) > fail_above:
                    raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
                return [np.full(len(text) * 100, 1000, dtype=np.int16) for text in texts]

        return SizedModel(), calls

    def test_generate_blocks_tts_batches_blocks_in_order(self):
        model, calls = self._sized_model()
        text = "Um. Dois dois. Três três três. Quatro quatro quatro quatro. Cinco."
        blocks = split_for_subtitles(text, max_chars=20)
        audio_tuple, _, srt_path, status = generate_blocks_tts(
            model=model,
            sampling_rate=1000,
            voice_prompt="dummy_prompt",
            text=text,
            pause_ms=0,
            subtitle_max_chars=20,
            batch_size=2,
        )
        self.assertIn("Sucesso!", status)
        self.assertGreater(len(blocks), 2)
        expected_sizes = [min(2, len(blocks) - start) for start in range(0, len(blocks), 2)]
        self.assertEqual([len(call) for call in calls], expected_sizes)
        self.assertEqual([text for call in calls for text in call], blocks)
        self.assertEqual(len(audio_tuple[1]), sum(len(block) * 100 for block in blocks))
        with open(srt_path, "r", encoding="utf-8") as f:
            srt_content = f.read()
        first_end = len(blocks[0]) * 100 / 1000
        self.assertIn(f"00:00:00,000 --> {format_srt_timestamp(first_end)}", srt_content)

    def test_generate_blocks_tts_halves_batch_on_gpu_out_of_memory(self):
        model, calls = self._sized_model(fail_above=2)
        text = "Um. Dois dois. Três três três. Quatro quatro quatro quatro. Cinco."
        blocks = split_for_subtitles(text, max_chars=20)
        audio_tuple, _, _, status = generate_blocks_tts(
            model=model,
            sampling_rate=1000,
            voice_prompt="dummy_prompt",
            text=text,
            pause_ms=0,
            subtitle_max_chars=20,
            batch_size=4,
        )
        self.assertIn("Sucesso!", status)
        successful = [call for call in calls if len(call) <= 2]
        self.assertEqual([text for call in successful for text in call], blocks)
        self.assertEqual(len(audio_tuple[1]), sum(len(block) * 100 for block in blocks))


if __name__ == "__main__":
    unittest.main()
