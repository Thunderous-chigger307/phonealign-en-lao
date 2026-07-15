"""Acoustic model wrapper: wav2vec2 CTC emissions over espeak IPA phones."""

from __future__ import annotations

import numpy as np
import torch

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
SAMPLE_RATE = 16_000


class AcousticModel:
    """Lazy-loaded wav2vec2 phoneme-CTC model producing log-prob emissions."""

    def __init__(self, model_id: str = MODEL_ID, device: str = "cpu"):
        self.model_id = model_id
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)
        self._model = None
        self._vocab: dict[str, int] | None = None
        self._blank_id: int | None = None

    def _load(self):
        if self._model is not None:
            return
        # The wav2vec2 phoneme tokenizer instantiates phonemizer/espeak on
        # load (even though we never use its phonemize()); make sure the
        # bundled espeak-ng library is registered first.
        from phonalign.g2p import ensure_espeak_library

        ensure_espeak_library()
        from transformers import AutoModelForCTC, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCTC.from_pretrained(self.model_id).to(self.device).eval()
        self._vocab = tokenizer.get_vocab()
        self._blank_id = tokenizer.pad_token_id

    @property
    def vocab(self) -> dict[str, int]:
        self._load()
        return self._vocab

    @property
    def blank_id(self) -> int:
        self._load()
        return self._blank_id

    def emissions(self, waveform: np.ndarray) -> torch.Tensor:
        """Log-prob emissions [num_frames, vocab_size] for 16 kHz mono float audio."""
        return self.batch_emissions([waveform])[0]

    @torch.inference_mode()
    def batch_emissions(self, waveforms: list[np.ndarray]) -> list[torch.Tensor]:
        """Emissions for several utterances in one padded forward pass.

        Waveforms are normalized individually, zero-padded to the longest item,
        and masked so padding never influences attention; each returned emission
        matrix is sliced back to that utterance's true frame count.
        """
        if not waveforms:
            return []
        self._load()
        xs = []
        for w in waveforms:
            x = torch.from_numpy(np.ascontiguousarray(w, dtype=np.float32))
            # wav2vec2-lv-60 was trained on zero-mean/unit-var normalized audio
            xs.append((x - x.mean()) / (x.std() + 1e-7))
        lengths = torch.tensor([len(x) for x in xs], dtype=torch.long)
        max_len = int(lengths.max())
        batch = torch.zeros(len(xs), max_len)
        mask = torch.zeros(len(xs), max_len, dtype=torch.long)
        for i, x in enumerate(xs):
            batch[i, : len(x)] = x
            mask[i, : len(x)] = 1
        logits = self._model(batch.to(self.device), attention_mask=mask.to(self.device)).logits
        n_frames = self._model._get_feat_extract_output_lengths(lengths)
        return [
            torch.log_softmax(logits[i, : int(n_frames[i])].float(), dim=-1).cpu()
            for i in range(len(xs))
        ]


def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int, float]:
    """Load audio as mono float32 at target_sr.

    Returns (waveform, original_sample_rate, duration_seconds). Duration is
    computed from the original file so downstream frame math matches the
    untouched audio.
    """
    import soundfile as sf
    import soxr

    data, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    duration = len(mono) / sr
    if sr != target_sr:
        mono = soxr.resample(mono, sr, target_sr)
    return mono, sr, duration
