"""VITS-style pipe-delimited training filelists.

Line format:  <wav_path>|<phoneme text>            (single speaker)
              <wav_path>|<speaker_id>|<phoneme text>  (multi speaker)

Phoneme text: phones concatenated within a word, words separated by spaces —
the layout VITS-style char-level tokenizers expect for pre-phonemized input.
"""

from __future__ import annotations

import random
from pathlib import Path

from phonalign.align import AlignmentResult


def phoneme_text(result: AlignmentResult) -> str:
    return " ".join("".join(p.label for p in w.phones) for w in result.words)


class VitsFilelistWriter:
    def __init__(self, out_dir: str | Path, val_count: int = 0, seed: int = 1234):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.val_count = val_count
        self.seed = seed
        self._rows: list[tuple[int, str]] = []

    def add(
        self,
        wav_path: str,
        result: AlignmentResult,
        speaker: str | None = None,
        order: int | None = None,
    ) -> None:
        """`order` fixes the row's position in the filelists; batched pipelines
        pass the corpus index so files (and the seeded val split) come out
        identical no matter what order utterances were processed in."""
        fields = [str(wav_path)]
        if speaker is not None:
            fields.append(speaker)
        fields.append(phoneme_text(result))
        key = order if order is not None else len(self._rows)
        self._rows.append((key, "|".join(fields)))

    def close(self) -> list[Path]:
        rows = [line for _, line in sorted(self._rows, key=lambda t: t[0])]
        written = []
        all_path = self.out_dir / "filelist_all.txt"
        all_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
        written.append(all_path)
        if self.val_count > 0 and len(rows) > self.val_count:
            random.Random(self.seed).shuffle(rows)
            val, train = rows[: self.val_count], rows[self.val_count :]
            for name, subset in (("train", train), ("val", val)):
                p = self.out_dir / f"filelist_{name}.txt"
                p.write_text("\n".join(subset) + "\n", encoding="utf-8")
                written.append(p)
        return written
