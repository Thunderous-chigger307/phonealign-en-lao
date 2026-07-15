"""JSON output: one file per utterance plus a corpus-level JSONL manifest."""

from __future__ import annotations

import json
from pathlib import Path

from phonalign.align import AlignmentResult


def alignment_record(utt_id: str, wav_path: str, result: AlignmentResult) -> dict:
    return {
        "id": utt_id,
        "wav": str(wav_path),
        "text": result.text,
        "language": result.language,
        "duration": result.audio_duration,
        "phones": [
            {"phone": p.label, "start": p.start, "end": p.end, "score": p.score}
            for p in result.phones
        ],
        "words": [{"word": w.label, "start": w.start, "end": w.end} for w in result.words],
    }


class ManifestWriter:
    """Writes per-utterance JSON files and appends to manifest.jsonl."""

    def __init__(self, out_dir: str | Path):
        self.json_dir = Path(out_dir) / "json"
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = Path(out_dir) / "manifest.jsonl"
        self._fh = open(self.manifest_path, "w", encoding="utf-8")
        self._orders: list[int] = []

    def add(
        self, utt_id: str, wav_path: str, result: AlignmentResult, order: int | None = None
    ) -> None:
        """`order` fixes the record's line position in manifest.jsonl; batched
        pipelines pass the corpus index so the manifest comes out identical no
        matter what order utterances were processed in."""
        record = alignment_record(utt_id, wav_path, result)
        with open(self.json_dir / f"{utt_id}.json", "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._orders.append(order if order is not None else len(self._orders))

    def close(self) -> None:
        self._fh.close()
        # Records were streamed in processing order; put the file in corpus
        # order if the two differ (length-sorted batched runs).
        if self._orders != sorted(self._orders):
            lines = self.manifest_path.read_text(encoding="utf-8").splitlines()
            ordered = [line for _, line in sorted(zip(self._orders, lines), key=lambda t: t[0])]
            self.manifest_path.write_text("\n".join(ordered) + "\n", encoding="utf-8")
