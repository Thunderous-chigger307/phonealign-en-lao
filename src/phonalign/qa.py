"""Per-utterance quality report over alignment confidence scores."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from phonalign.align import AlignmentResult
from phonalign.errors import UnmappablePhoneError


@dataclass
class QARow:
    utt_id: str
    status: str  # ok | flagged | error
    n_phones: int = 0
    duration: float = 0.0
    mean_score: float = 0.0
    min_score: float = 0.0
    detail: str = ""


def evaluate(utt_id: str, result: AlignmentResult, flag_threshold: float = 0.5) -> QARow:
    scores = [p.score for p in result.phones]
    mean_score = sum(scores) / len(scores)
    min_score = min(scores)
    flagged = mean_score < flag_threshold
    return QARow(
        utt_id=utt_id,
        status="flagged" if flagged else "ok",
        n_phones=len(scores),
        duration=result.audio_duration,
        mean_score=round(mean_score, 4),
        min_score=round(min_score, 4),
        detail="low mean confidence — check transcript/audio match" if flagged else "",
    )


def error_row(utt_id: str, exc: Exception) -> QARow:
    """Turn a per-utterance failure into a report row so the run can continue."""
    if isinstance(exc, UnmappablePhoneError):
        detail = (
            f"unmappable phone {exc.phone!r} in word {exc.word!r} "
            f"(language {exc.language}) — add it to FALLBACK_PHONE_MAP or LANG_TOKEN_OVERRIDES"
        )
    else:
        detail = f"{type(exc).__name__}: {exc}"
    return QARow(utt_id=utt_id, status="error", detail=detail)


def write_report(rows: list[QARow], out_dir: str | Path) -> Path:
    path = Path(out_dir) / "report.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "status", "n_phones", "duration_s", "mean_score", "min_score", "detail"])
        for r in rows:
            writer.writerow(
                [r.utt_id, r.status, r.n_phones, r.duration, r.mean_score, r.min_score, r.detail]
            )
    return path
