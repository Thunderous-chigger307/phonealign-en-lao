import math

import numpy as np
import pytest
import torch

from phonalign.align import (
    Aligner,
    AlignmentResult,
    Phone,
    PhoneVocabMapper,
    Word,
    ctc_forced_align,
)
from phonalign.errors import AlignmentError, G2PError, UnmappablePhoneError
from phonalign.g2p import WordG2P

BLANK = 0


def make_emissions(frame_tokens, vocab_size=6, p=0.9):
    """Emissions where each frame strongly prefers the given token id."""
    T = len(frame_tokens)
    rest = (1.0 - p) / (vocab_size - 1)
    probs = torch.full((T, vocab_size), rest)
    for t, tok in enumerate(frame_tokens):
        probs[t, tok] = p
    return probs.log()


class TestCtcForcedAlign:
    def test_clean_three_token_alignment(self):
        frames = [1, 1, 1, 0, 0, 2, 2, 2, 3, 3, 0, 0]
        spans = ctc_forced_align(make_emissions(frames), torch.tensor([1, 2, 3]), BLANK)
        assert [s[0] for s in spans] == [0, 1, 2]
        starts = [s[1] for s in spans]
        ends = [s[2] for s in spans]
        assert starts == [0, 5, 8]
        assert ends == [3, 8, 10]
        assert all(s[3] > 0.5 for s in spans)

    def test_repeated_token_needs_blank(self):
        frames = [1, 1, 0, 1, 1]
        spans = ctc_forced_align(make_emissions(frames), torch.tensor([1, 1]), BLANK)
        assert len(spans) == 2
        assert spans[0][2] <= spans[1][1]

    def test_empty_targets_raises(self):
        with pytest.raises(AlignmentError):
            ctc_forced_align(make_emissions([0, 0]), torch.tensor([], dtype=torch.long), BLANK)

    def test_too_short_audio_raises(self):
        with pytest.raises(AlignmentError):
            ctc_forced_align(make_emissions([1]), torch.tensor([1, 2, 3]), BLANK)

    def test_spans_cover_all_targets_in_order(self):
        frames = [1, 2, 3, 4, 5, 0]
        spans = ctc_forced_align(
            make_emissions(frames, vocab_size=7), torch.tensor([1, 2, 3, 4, 5]), BLANK
        )
        assert [s[0] for s in spans] == [0, 1, 2, 3, 4]


class TestPhoneVocabMapper:
    VOCAB = {"<pad>": 0, "a": 1, "b": 2, "aː": 3, "tʃ": 4, "t": 5, "ʃ": 6, "k": 7}

    def setup_method(self):
        self.mapper = PhoneVocabMapper(self.VOCAB)

    def test_exact(self):
        assert self.mapper.map_phone("a") == [1]
        assert self.mapper.map_phone("aː") == [3]
        assert self.mapper.map_phone("tʃ") == [4]

    def test_stress_stripped(self):
        assert self.mapper.map_phone("ˈa") == [1]

    def test_length_fallback(self):
        # 'bː' not in vocab -> falls back to 'b'
        assert self.mapper.map_phone("bː") == [2]

    def test_greedy_segmentation(self):
        assert self.mapper.map_phone("ab") == [1, 2]
        # aspiration modifier dropped when unmatchable
        assert self.mapper.map_phone("kʰ") == [7]

    def test_tie_bar_stripped(self):
        # t͡ʃ (with tie bar) must hit the single 'tʃ' token, not t+ʃ
        assert self.mapper.map_phone("t͡ʃ") == [4]

    def test_overrides_beat_exact_match(self):
        mapper = PhoneVocabMapper(self.VOCAB, overrides={"a": ["b"], "t": ["t", "ʃ"]})
        assert mapper.map_phone("a") == [2]
        assert mapper.map_phone("t") == [5, 6]
        # non-overridden phones still map normally
        assert mapper.map_phone("b") == [2]

    def test_override_with_missing_token_falls_through(self):
        mapper = PhoneVocabMapper(self.VOCAB, overrides={"a": ["nope"]})
        assert mapper.map_phone("a") == [1]

    def test_unmappable_raises(self):
        with pytest.raises(UnmappablePhoneError):
            self.mapper.map_phone("ɸ", word="x", language="xx")

    def test_fallback_map(self):
        # ɤ (Lao) falls back to ʌ; ASCII g maps to IPA script ɡ
        vocab = dict(self.VOCAB, **{"ʌ": 8, "ɜː": 9, "ɡ": 10})
        mapper = PhoneVocabMapper(vocab)
        assert mapper.map_phone("ɤ") == [8]
        assert mapper.map_phone("ɤː") == [9]
        assert mapper.map_phone("g") == [10]


class FakeAcoustic:
    """Stands in for AcousticModel: fixed vocab, synthetic emissions."""

    vocab = {"<pad>": 0, "a": 1, "b": 2}
    blank_id = 0

    def batch_emissions(self, waveforms):
        return [make_emissions([1, 1, 1, 0, 2, 2], vocab_size=3) for _ in waveforms]

    def emissions(self, waveform):
        return self.batch_emissions([waveform])[0]


class FakeG2P:
    def word_phones(self, text):
        if text == "boom":
            raise G2PError("bad text")
        return [WordG2P(word=text, phones=["a", "b"])]


class TestAlignBatch:
    @pytest.fixture
    def wav(self, tmp_path):
        import soundfile as sf

        path = tmp_path / "utt.wav"
        rng = np.random.default_rng(0)
        sf.write(path, rng.standard_normal(1600).astype(np.float32) * 0.1, 16000)
        return str(path)

    @pytest.fixture
    def aligner(self):
        a = Aligner(lang="xx", g2p=FakeG2P())
        a.acoustic = FakeAcoustic()
        return a

    def test_batch_matches_single(self, aligner, wav):
        single = aligner.align(wav, "hello")
        (batched,) = aligner.align_batch([(wav, "hello")])
        assert isinstance(batched, AlignmentResult)
        assert [(p.label, p.start, p.end, p.score) for p in batched.phones] == [
            (p.label, p.start, p.end, p.score) for p in single.phones
        ]

    def test_results_positional_and_errors_isolated(self, aligner, wav):
        outcomes = aligner.align_batch(
            [(wav, "one"), (wav, "boom"), (wav, "two")], batch_size=3
        )
        assert isinstance(outcomes[0], AlignmentResult)
        assert isinstance(outcomes[1], G2PError)
        assert isinstance(outcomes[2], AlignmentResult)
        assert outcomes[0].text == "one" and outcomes[2].text == "two"

    def test_missing_wav_isolated(self, aligner, wav):
        outcomes = aligner.align_batch([(wav, "one"), ("no_such.wav", "two")])
        assert isinstance(outcomes[0], AlignmentResult)
        assert isinstance(outcomes[1], Exception)
        assert not isinstance(outcomes[1], AlignmentResult)

    def test_all_items_fail(self, aligner):
        outcomes = aligner.align_batch([("no_such.wav", "boom"), ("gone.wav", "x")])
        assert all(isinstance(o, Exception) for o in outcomes)

    def test_chunking_covers_all_items(self, aligner, wav):
        outcomes = aligner.align_batch([(wav, f"w{i}") for i in range(5)], batch_size=2)
        assert len(outcomes) == 5
        assert all(isinstance(o, AlignmentResult) for o in outcomes)
        assert [o.text for o in outcomes] == [f"w{i}" for i in range(5)]

    def test_bad_batch_size_rejected(self, aligner, wav):
        with pytest.raises(ValueError):
            aligner.align_batch([(wav, "one")], batch_size=0)


class TestFullTimeline:
    def make_result(self, phones, duration):
        return AlignmentResult(
            phones=phones, words=[], audio_duration=duration, language="en-us", text=""
        )

    def test_gaps_become_silence(self):
        result = self.make_result(
            [Phone("a", 0.5, 0.6, 1.0), Phone("b", 0.62, 0.7, 1.0)], duration=1.0
        )
        tl = result.full_timeline(min_silence=0.04)
        assert tl[0].label == "sil" and tl[0].start == 0.0
        labels = [p.label for p in tl]
        assert labels == ["sil", "a", "b", "sil"]
        # small 0.02 gap absorbed into 'b'
        assert math.isclose(tl[2].start, 0.6, abs_tol=1e-9)
        # full coverage, no holes
        assert tl[0].start == 0.0 and tl[-1].end == 1.0
        for prev, nxt in zip(tl, tl[1:]):
            assert math.isclose(prev.end, nxt.start, abs_tol=1e-9)

    def test_no_phones_all_silence(self):
        tl = self.make_result([], 2.0).full_timeline()
        assert len(tl) == 1 and tl[0].label == "sil" and tl[0].end == 2.0
