"""Offline Arabic speech synthesis using the downloaded MMS checkpoint."""

import io
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class ArabicSpeech:
    def __init__(self, checkpoint=None):
        from transformers import AutoTokenizer, VitsModel

        self.root = Path(checkpoint or ROOT / "checkpoints/tts/mms-ara")
        self.tokenizer = AutoTokenizer.from_pretrained(self.root)
        self.model = VitsModel.from_pretrained(self.root).eval()

    def wav_bytes(self, text):
        import torch

        if not text.strip() or len(text) > 500:
            raise ValueError("Speech text must contain 1–500 characters")
        tokens = self.tokenizer(text, return_tensors="pt")
        with torch.inference_mode():
            audio = self.model(**tokens).waveform[0].cpu().numpy()
        stream = io.BytesIO()
        pcm = (np.clip(np.asarray(audio), -1, 1) * 32767).astype("<i2")
        with wave.open(stream, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(self.model.config.sampling_rate)
            wav.writeframes(pcm.tobytes())
        return stream.getvalue()
