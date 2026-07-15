"""Forced alignment core.

Pipeline: G2P phones -> map each phone onto one or more acoustic-model vocab
tokens -> CTC Viterbi forced alignment over the emission matrix -> merge token
spans back into per-phone (and per-word) time intervals.

The Viterbi implementation replaces torchaudio.functional.forced_align
(torchaudio's newest release no longer tracks current torch); it is the
standard CTC trellis over the blank-interleaved target sequence.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

import torch

from phonalign.acoustic import SAMPLE_RATE, AcousticModel, load_audio
from phonalign.errors import AlignmentError, UnmappablePhoneError
from phonalign.g2p import G2PBackend, get_g2p

#: Manual escape hatch for phones the automatic mapping can't place.
#: Maps G2P phone -> list of acoustic-model vocab tokens.
FALLBACK_PHONE_MAP: dict[str, list[str]] = {
    "ɤ": ["ʌ"],  # close-mid back unrounded (Lao) — nearest vocab vowel
    "ɤː": ["ɜː"],
    "g": ["ɡ"],  # ASCII g (epitran) -> IPA script g (model vocab)
}

#: Per-language phone -> vocab-token overrides, applied before every other
#: mapping route (including exact vocab hits). The model concentrates its
#: probability mass on the token spellings its training transcripts used, so
#: an exact-but-rare IPA token can align far worse than the token the model
#: actually emits for that sound. Output labels are unaffected — this only
#: changes which tokens Viterbi aligns against.
#:
#: Lao entries were chosen empirically on the bundled examples: the model
#: renders Lao aspirated stops with espeak's digraph tokens (ph/th) or x, and
#: its vowels with the tone-language tokens (i5, ɑ5, ...) from its
#: Mandarin-style training transcripts.
LANG_TOKEN_OVERRIDES: dict[str, dict[str, list[str]]] = {
    "lo": {
        "pʰ": ["ph"],
        "tʰ": ["th"],
        "kʰ": ["x"],
        "iː": ["i5"],
        "aː": ["ɑ5"],
        "a": ["a5"],
        "eː": ["ei5"],
        "ɯː": ["əɜ"],
    },
}

#: Stress / tone / delimiter / tie-bar marks that carry no segmental identity
#: for the acoustic model (its affricates are single tokens without tie bars).
#: Stripped when looking up vocab tokens; preserved in output labels.
_IGNORABLE_MARKS = "ˈˌ˥˦˧˨˩'’‿·͜͡"


@dataclass
class Phone:
    label: str
    start: float
    end: float
    score: float

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class Word:
    label: str
    start: float
    end: float
    phones: list[Phone] = field(default_factory=list)


@dataclass
class AlignmentResult:
    phones: list[Phone]
    words: list[Word]
    audio_duration: float
    language: str
    text: str

    def full_timeline(self, min_silence: float = 0.04, silence_label: str = "sil") -> list[Phone]:
        """Gap-free phone sequence covering [0, audio_duration].

        Gaps >= min_silence become explicit silence phones; smaller gaps are
        absorbed into the neighboring phone. TTS duration targets need every
        audio frame accounted for.
        """
        if not self.phones:
            return [Phone(silence_label, 0.0, self.audio_duration, 1.0)]
        out: list[Phone] = []
        cursor = 0.0
        for p in self.phones:
            gap = p.start - cursor
            if gap >= min_silence:
                out.append(Phone(silence_label, cursor, p.start, 1.0))
                out.append(Phone(p.label, p.start, p.end, p.score))
            else:
                out.append(Phone(p.label, cursor, p.end, p.score))
            cursor = out[-1].end
        tail = self.audio_duration - cursor
        if tail >= min_silence:
            out.append(Phone(silence_label, cursor, self.audio_duration, 1.0))
        else:
            out[-1] = Phone(out[-1].label, out[-1].start, self.audio_duration, out[-1].score)
        return out


def ctc_forced_align(
    log_probs: torch.Tensor, targets: torch.Tensor, blank: int
) -> list[tuple[int, int, int, float]]:
    """Viterbi forced alignment of a CTC target sequence to emissions.

    Args:
        log_probs: [T, V] frame log-probabilities.
        targets: [N] vocab ids, no blanks.
        blank: blank (pad) token id.

    Returns one (target_index, start_frame, end_frame_exclusive, mean_prob)
    per target token.
    """
    T, _ = log_probs.shape
    N = len(targets)
    if N == 0:
        raise AlignmentError("empty target sequence")
    if T < N:
        raise AlignmentError(f"audio too short: {T} frames for {N} tokens")

    # Blank-interleaved state sequence: [b, y1, b, y2, ..., yN, b]
    S = 2 * N + 1
    z = torch.full((S,), blank, dtype=torch.long)
    z[1::2] = targets
    frame_z = log_probs[:, z]  # [T, S]

    NEG_INF = float("-inf")
    # Skip transition s-2 -> s is legal only into a non-blank state whose
    # token differs from the token two states back.
    can_skip = torch.zeros(S, dtype=torch.bool)
    can_skip[3::2] = z[3::2] != z[1:-2:2]

    dp = torch.full((S,), NEG_INF)
    dp[0] = frame_z[0, 0]
    dp[1] = frame_z[0, 1]
    ptr = torch.zeros((T, S), dtype=torch.int8)  # 0=stay, 1=from s-1, 2=from s-2

    for t in range(1, T):
        prev = dp
        from_prev = torch.cat([torch.tensor([NEG_INF]), prev[:-1]])
        from_skip = torch.cat([torch.tensor([NEG_INF, NEG_INF]), prev[:-2]])
        from_skip = torch.where(can_skip, from_skip, torch.tensor(NEG_INF))
        stacked = torch.stack([prev, from_prev, from_skip])  # [3, S]
        best, arg = stacked.max(dim=0)
        dp = best + frame_z[t]
        ptr[t] = arg.to(torch.int8)

    # Path must end in the last token or the trailing blank.
    s = S - 1 if dp[S - 1] >= dp[S - 2] else S - 2
    if dp[s] == NEG_INF:
        raise AlignmentError("no valid alignment path (audio/text mismatch too severe)")

    states = torch.empty(T, dtype=torch.long)
    for t in range(T - 1, -1, -1):
        states[t] = s
        s -= int(ptr[t, s])

    probs = frame_z.exp()
    spans: list[tuple[int, int, int, float]] = []
    t = 0
    while t < T:
        s = int(states[t])
        t_end = t
        while t_end + 1 < T and int(states[t_end + 1]) == s:
            t_end += 1
        if s % 2 == 1:  # non-blank state
            score = float(probs[t : t_end + 1, s].mean())
            spans.append((s // 2, t, t_end + 1, score))
        t = t_end + 1
    return spans


class PhoneVocabMapper:
    """Maps G2P phone labels onto acoustic-model vocab token id sequences."""

    def __init__(self, vocab: dict[str, int], overrides: dict[str, list[str]] | None = None):
        self._vocab = vocab
        self._overrides = overrides or {}
        self._max_token_len = max(len(t) for t in vocab)
        self._cache: dict[str, list[int]] = {}

    def map_phone(self, phone: str, word: str = "?", language: str = "?") -> list[int]:
        if phone in self._cache:
            return self._cache[phone]
        ids = self._try_map(phone)
        if ids is None:
            raise UnmappablePhoneError(phone, word, language)
        self._cache[phone] = ids
        return ids

    def _try_map(self, phone: str) -> list[int] | None:
        toks = self._overrides.get(phone)
        if toks is not None and all(t in self._vocab for t in toks):
            return [self._vocab[t] for t in toks]
        for cand in self._candidates(phone):
            if cand in self._vocab:
                return [self._vocab[cand]]
        if phone in FALLBACK_PHONE_MAP:
            toks = FALLBACK_PHONE_MAP[phone]
            if all(t in self._vocab for t in toks):
                return [self._vocab[t] for t in toks]
        return self._greedy_segment(phone)

    @staticmethod
    def _candidates(phone: str):
        yield phone
        stripped = "".join(c for c in phone if c not in _IGNORABLE_MARKS)
        if stripped and stripped != phone:
            yield stripped
        for form in ("NFC", "NFD"):
            norm = unicodedata.normalize(form, stripped or phone)
            if norm != phone:
                yield norm
        if stripped.endswith("ː"):
            yield stripped[:-1]

    def _greedy_segment(self, phone: str) -> list[int] | None:
        """Longest-match segmentation of a phone string into vocab tokens.

        Handles compound G2P output like affricates or diphthongs the model
        vocab only has as parts. Combining/modifier marks that can't match
        are dropped.
        """
        s = "".join(c for c in phone if c not in _IGNORABLE_MARKS)
        ids: list[int] = []
        i = 0
        while i < len(s):
            match = None
            for ln in range(min(self._max_token_len, len(s) - i), 0, -1):
                piece = s[i : i + ln]
                if piece in self._vocab:
                    match = (piece, ln)
                    break
            if match:
                ids.append(self._vocab[match[0]])
                i += match[1]
                continue
            ch = s[i]
            cat = unicodedata.category(ch)
            if cat in ("Mn", "Sk", "Lm"):  # combining mark / modifier — droppable
                i += 1
                continue
            return None
        return ids or None


@dataclass
class _PreparedText:
    """G2P output flattened into vocab token ids, with the bookkeeping needed
    to merge aligned token spans back into phone and word intervals."""

    words_g2p: list
    token_ids: list[int] = field(default_factory=list)
    tokens_per_phone: list[int] = field(default_factory=list)
    flat_phone_labels: list[str] = field(default_factory=list)
    phones_per_word: list[int] = field(default_factory=list)


class Aligner:
    """End-to-end aligner: text + audio file -> phone/word timestamps."""

    def __init__(
        self,
        lang: str = "en-us",
        device: str = "cpu",
        model_id: str | None = None,
        preserve_stress: bool = False,
        g2p: G2PBackend | None = None,
    ):
        self.lang = lang
        self.g2p = g2p or get_g2p(lang, preserve_stress=preserve_stress)
        kwargs = {"device": device}
        if model_id:
            kwargs["model_id"] = model_id
        self.acoustic = AcousticModel(**kwargs)
        self._mapper: PhoneVocabMapper | None = None

    @property
    def mapper(self) -> PhoneVocabMapper:
        if self._mapper is None:
            self._mapper = PhoneVocabMapper(
                self.acoustic.vocab, overrides=LANG_TOKEN_OVERRIDES.get(self.lang)
            )
        return self._mapper

    def align(self, wav_path: str, text: str) -> AlignmentResult:
        prep = self._prepare(text)
        waveform, _, duration = load_audio(wav_path)
        emissions = self.acoustic.emissions(waveform)
        return self._merge(emissions, duration, prep, text)

    def align_batch(
        self, items: list[tuple[str, str]], batch_size: int = 8
    ) -> list[AlignmentResult | Exception]:
        """Align (wav_path, text) pairs, batching model forward passes.

        Runs `batch_size` utterances per forward pass (padded to the longest
        in the batch). Per-utterance failures never poison the batch: the
        returned list matches `items` positionally, holding an
        AlignmentResult on success or the raised exception on failure.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        results: list[AlignmentResult | Exception] = [None] * len(items)  # type: ignore[list-item]
        for chunk_start in range(0, len(items), batch_size):
            chunk = items[chunk_start : chunk_start + batch_size]
            prepared: dict[int, _PreparedText] = {}
            audio: dict[int, tuple] = {}
            for j, (wav_path, text) in enumerate(chunk):
                i = chunk_start + j
                try:
                    prepared[i] = self._prepare(text)
                    waveform, _, duration = load_audio(wav_path)
                    audio[i] = (waveform, duration)
                except Exception as exc:
                    results[i] = exc
            ready = sorted(prepared.keys() & audio.keys())
            if not ready:
                continue
            emissions = self.acoustic.batch_emissions([audio[i][0] for i in ready])
            for i, em in zip(ready, emissions):
                try:
                    results[i] = self._merge(em, audio[i][1], prepared[i], items[i][1])
                except Exception as exc:
                    results[i] = exc
        return results

    def _prepare(self, text: str) -> _PreparedText:
        """G2P + vocab mapping; everything that can fail before touching audio."""
        words_g2p = self.g2p.word_phones(text)
        if not words_g2p:
            raise AlignmentError(f"G2P produced no phones for text: {text!r}")

        # Flatten phones; remember how many vocab tokens each phone expands to
        # and how many phones each word has, so spans can be merged back.
        prep = _PreparedText(words_g2p=words_g2p)
        for wg in words_g2p:
            prep.phones_per_word.append(len(wg.phones))
            for ph in wg.phones:
                ids = self.mapper.map_phone(ph, word=wg.word, language=self.lang)
                prep.token_ids.extend(ids)
                prep.tokens_per_phone.append(len(ids))
                prep.flat_phone_labels.append(ph)
        return prep

    def _merge(
        self, emissions: torch.Tensor, duration: float, prep: _PreparedText, text: str
    ) -> AlignmentResult:
        """Viterbi over the emissions, then token spans -> phone/word intervals."""
        frame_dur = duration / emissions.shape[0]
        spans = ctc_forced_align(
            emissions, torch.tensor(prep.token_ids, dtype=torch.long), self.acoustic.blank_id
        )
        if len(spans) != len(prep.token_ids):
            raise AlignmentError(
                f"alignment returned {len(spans)} spans for {len(prep.token_ids)} tokens"
            )

        phones: list[Phone] = []
        idx = 0
        for label, n_tok in zip(prep.flat_phone_labels, prep.tokens_per_phone):
            group = spans[idx : idx + n_tok]
            idx += n_tok
            start = group[0][1] * frame_dur
            end = group[-1][2] * frame_dur
            score = sum(g[3] for g in group) / len(group)
            phones.append(Phone(label, round(start, 4), round(end, 4), round(score, 4)))

        words: list[Word] = []
        idx = 0
        for wg, n_ph in zip(prep.words_g2p, prep.phones_per_word):
            group = phones[idx : idx + n_ph]
            idx += n_ph
            words.append(Word(wg.word, group[0].start, group[-1].end, group))

        return AlignmentResult(
            phones=phones,
            words=words,
            audio_duration=round(duration, 4),
            language=self.lang,
            text=text,
        )
