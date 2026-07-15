import json
import random

import numpy as np

from phonalign.align import AlignmentResult, Phone, Word
from phonalign.writers import ManifestWriter, VitsFilelistWriter, write_durations, write_textgrid
from phonalign.writers.durations import phone_durations, total_frames
from phonalign.writers.vits import phoneme_text


def make_result(seed=0, duration=3.0, n_phones=12):
    rng = random.Random(seed)
    phones = []
    t = rng.uniform(0.0, 0.3)
    for i in range(n_phones):
        d = rng.uniform(0.02, 0.25)
        end = min(t + d, duration - 0.01)
        phones.append(Phone(f"p{i}", round(t, 4), round(end, 4), rng.uniform(0.3, 1.0)))
        t = end + (rng.uniform(0.0, 0.1) if rng.random() < 0.3 else 0.0)
    words = [
        Word("w1", phones[0].start, phones[5].end, phones[:6]),
        Word("w2", phones[6].start, phones[-1].end, phones[6:]),
    ]
    return AlignmentResult(
        phones=phones, words=words, audio_duration=duration, language="en-us", text="w1 w2"
    )


class TestDurations:
    def test_sum_invariant_many_seeds(self):
        for seed in range(25):
            result = make_result(seed=seed, duration=2.0 + seed * 0.13)
            labels, durs = phone_durations(result, sample_rate=22050, hop_length=256)
            expected = total_frames(result.audio_duration, 22050, 256)
            assert int(durs.sum()) == expected, f"seed {seed}"
            assert len(labels) == len(durs)
            assert (durs > 0).all()

    def test_write_files(self, tmp_path):
        result = make_result()
        npy = write_durations(result, "utt1", tmp_path)
        durs = np.load(npy)
        sidecar = json.loads((tmp_path / "utt1.json").read_text(encoding="utf-8"))
        assert sidecar["total_frames"] == int(durs.sum())
        assert len(sidecar["phones"]) == len(durs)


class TestTextGrid:
    def test_roundtrip(self, tmp_path):
        from praatio import textgrid as ptg

        result = make_result()
        path = write_textgrid(result, tmp_path / "utt1.TextGrid")
        tg = ptg.openTextgrid(str(path), includeEmptyIntervals=False)
        assert set(tg.tierNames) == {"words", "phones"}
        phone_labels = [e.label for e in tg.getTier("phones").entries]
        assert phone_labels == [p.label for p in result.phones]


class TestVits:
    def test_filelist_format(self, tmp_path):
        w = VitsFilelistWriter(tmp_path)
        result = make_result()
        w.add("wavs/utt1.wav", result)
        files = w.close()
        line = files[0].read_text(encoding="utf-8").strip()
        wav, phones = line.split("|")
        assert wav == "wavs/utt1.wav"
        assert phones == phoneme_text(result)
        assert " " in phones  # two words

    def test_speaker_and_split(self, tmp_path):
        w = VitsFilelistWriter(tmp_path, val_count=2)
        for i in range(10):
            w.add(f"wavs/u{i}.wav", make_result(seed=i), speaker=f"spk{i % 2}")
        files = {p.name: p for p in w.close()}
        assert set(files) == {"filelist_all.txt", "filelist_train.txt", "filelist_val.txt"}
        assert len(files["filelist_val.txt"].read_text(encoding="utf-8").strip().splitlines()) == 2
        assert len(files["filelist_train.txt"].read_text(encoding="utf-8").strip().splitlines()) == 8
        first = files["filelist_all.txt"].read_text(encoding="utf-8").splitlines()[0]
        assert first.split("|")[1] == "spk0"


    def test_order_key_makes_output_processing_order_invariant(self, tmp_path):
        # same 10 utterances added in two different processing orders
        # -> identical filelists, including the seeded train/val split
        perm = [3, 7, 0, 9, 1, 5, 8, 2, 6, 4]
        outs = []
        for name, order in (("a", range(10)), ("b", perm)):
            w = VitsFilelistWriter(tmp_path / name, val_count=2)
            for i in order:
                w.add(f"wavs/u{i}.wav", make_result(seed=i), order=i)
            files = {p.name: p.read_text(encoding="utf-8") for p in w.close()}
            outs.append(files)
        assert outs[0] == outs[1]
        first = outs[0]["filelist_all.txt"].splitlines()[0]
        assert first.startswith("wavs/u0.wav")


class TestManifest:
    def test_jsonl_and_per_utt(self, tmp_path):
        m = ManifestWriter(tmp_path)
        result = make_result()
        m.add("utt1", "wavs/utt1.wav", result)
        m.add("utt2", "wavs/utt2.wav", make_result(seed=2))
        m.close()
        lines = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        rec = json.loads(lines[0])
        assert rec["id"] == "utt1"
        assert len(rec["phones"]) == len(result.phones)
        assert {"phone", "start", "end", "score"} <= set(rec["phones"][0])
        per_utt = json.loads((tmp_path / "json" / "utt1.json").read_text(encoding="utf-8"))
        assert per_utt == rec

    def test_out_of_order_adds_restore_corpus_order(self, tmp_path):
        m = ManifestWriter(tmp_path)
        for i in (2, 0, 1):
            m.add(f"utt{i}", f"wavs/utt{i}.wav", make_result(seed=i), order=i)
        m.close()
        lines = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").strip().splitlines()
        assert [json.loads(ln)["id"] for ln in lines] == ["utt0", "utt1", "utt2"]
