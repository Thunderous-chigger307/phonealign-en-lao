"""phonalign command-line interface."""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress

# IPA phones are far outside cp1252; make Windows consoles/pipes UTF-8-safe.
if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

app = typer.Typer(
    name="phonalign",
    help="Audio-to-phoneme forced alignment for TTS dataset preprocessing.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
console = Console()
err_console = Console(stderr=True, style="bold red")

FORMATS = ("textgrid", "json", "vits", "durations")


@app.command()
def align(
    input: Path = typer.Option(..., "--input", "-i", help="Corpus dir or metadata.csv"),
    out: Path = typer.Option(..., "--out", "-o", help="Output directory"),
    lang: str = typer.Option("en-us", "--lang", "-l", help="Language code (e.g. en-us, lo)"),
    formats: str = typer.Option(
        "textgrid,json,vits,durations", "--formats", "-f",
        help=f"Comma-separated output formats: {','.join(FORMATS)}",
    ),
    sample_rate: int = typer.Option(22050, help="Target TTS sample rate for duration frames"),
    hop_length: int = typer.Option(256, help="Mel hop length for duration frames"),
    device: str = typer.Option("cpu", help="cpu or cuda"),
    batch_size: int = typer.Option(
        0, "--batch-size", "-b", min=0,
        help="Utterances per model forward pass. 0 = auto: 1 on cpu (torch already "
        "uses all cores per utterance), 8 on cuda. Batched runs process in "
        "length-sorted order to minimize padding.",
    ),
    speaker_column: bool = typer.Option(
        False, "--speaker-column", help="metadata.csv rows are id|speaker|text"
    ),
    val_count: int = typer.Option(0, help="Held-out utterances for the VITS val filelist"),
    preserve_stress: bool = typer.Option(False, help="Keep espeak stress marks in phone labels"),
    flag_threshold: float = typer.Option(
        0.5, help="Flag utterances whose mean phone confidence is below this"
    ),
):
    """Align a corpus and write TTS-ready artifacts."""
    from phonalign import corpus as corpus_mod
    from phonalign import qa
    from phonalign.align import Aligner
    from phonalign.writers import ManifestWriter, VitsFilelistWriter, write_durations, write_textgrid

    fmts = [f.strip() for f in formats.split(",") if f.strip()]
    unknown = set(fmts) - set(FORMATS)
    if unknown:
        err_console.print(f"unknown formats: {', '.join(sorted(unknown))}")
        raise typer.Exit(2)

    try:
        utts = corpus_mod.discover(input, has_speaker=speaker_column)
    except Exception as exc:
        err_console.print(str(exc))
        raise typer.Exit(2)
    console.print(f"[green]Found {len(utts)} utterances[/green] in {input}")

    console.print(f"Loading G2P + acoustic model (lang={lang}, device={device}) ...")
    aligner = Aligner(lang=lang, device=device, preserve_stress=preserve_stress)

    if batch_size == 0:
        batch_size = 8 if aligner.acoustic.device.type == "cuda" else 1
    indexed = list(enumerate(utts))
    if batch_size > 1:
        # Batches are padded to their longest utterance; grouping similar
        # lengths keeps that padding (wasted compute) near zero.
        import soundfile as sf

        def _duration(u) -> float:
            try:
                return sf.info(str(u.wav_path)).duration
            except Exception:
                return 0.0

        indexed.sort(key=lambda iu: _duration(iu[1]))
        console.print(f"Batching {batch_size} utterances per forward pass (length-sorted)")

    out.mkdir(parents=True, exist_ok=True)
    manifest = ManifestWriter(out) if "json" in fmts else None
    vits = VitsFilelistWriter(out / "vits", val_count=val_count) if "vits" in fmts else None
    rows_by_idx: dict[int, qa.QARow] = {}
    n_err = 0

    with Progress(console=console) as progress:
        task = progress.add_task("Aligning", total=len(utts))
        for start in range(0, len(indexed), batch_size):
            chunk = indexed[start : start + batch_size]
            outcomes = aligner.align_batch(
                [(str(u.wav_path), u.text) for _, u in chunk], batch_size=batch_size
            )
            for (idx, utt), outcome in zip(chunk, outcomes):
                try:
                    if isinstance(outcome, Exception):
                        raise outcome
                    result = outcome
                    if "textgrid" in fmts:
                        write_textgrid(result, out / "textgrid" / f"{utt.id}.TextGrid")
                    if manifest is not None:
                        manifest.add(utt.id, str(utt.wav_path), result, order=idx)
                    if vits is not None:
                        vits.add(str(utt.wav_path), result, speaker=utt.speaker, order=idx)
                    if "durations" in fmts:
                        write_durations(
                            result, utt.id, out / "durations",
                            sample_rate=sample_rate, hop_length=hop_length,
                        )
                    rows_by_idx[idx] = qa.evaluate(utt.id, result, flag_threshold)
                except Exception as exc:  # skip the utterance, log it, keep going
                    n_err += 1
                    rows_by_idx[idx] = qa.error_row(utt.id, exc)
                    progress.console.print(f"[red]skip[/red] {utt.id}: {exc}")
                progress.advance(task)
    # report.csv keeps corpus order even when batching processed length-sorted
    qa_rows = [rows_by_idx[i] for i in sorted(rows_by_idx)]

    if manifest is not None:
        manifest.close()
    if vits is not None:
        vits.close()
    report = qa.write_report(qa_rows, out)

    n_flag = sum(1 for r in qa_rows if r.status == "flagged")
    n_ok = sum(1 for r in qa_rows if r.status == "ok")
    console.print(
        f"\n[bold]Done:[/bold] {n_ok} ok, {n_flag} flagged, {n_err} skipped. "
        f"Report: {report}"
    )
    if n_err:
        console.print(
            f"[yellow]{n_err} utterance(s) skipped — see the 'error' rows in {report}[/yellow]"
        )
    if utts and n_err == len(utts):
        # every utterance failed: that's a setup problem, not bad data
        err_console.print("all utterances failed — check language code, audio format, and transcripts")
        raise typer.Exit(1)


@app.command()
def doctor(
    download_model: bool = typer.Option(
        False, "--download-model", help="Pre-download the acoustic model (~1.3 GB)"
    ),
):
    """Check that all runtime pieces are in place and print fixes."""
    from phonalign.acoustic import MODEL_ID

    ok = True

    def check(label: str, fn):
        nonlocal ok
        try:
            detail = fn()
            console.print(f"  [green]OK[/green]  {label}" + (f" — {detail}" if detail else ""))
        except Exception as exc:
            ok = False
            console.print(f"  [red]FAIL[/red] {label} — {exc}")

    console.print(f"[bold]phonalign doctor[/bold] (python {sys.version.split()[0]})\n")

    def _torch():
        import torch
        dev = "cuda available" if torch.cuda.is_available() else "cpu only"
        return f"torch {torch.__version__}, {dev}"
    check("torch", _torch)

    def _espeak():
        from phonalign.g2p import EspeakG2P, ensure_espeak_library
        source = ensure_espeak_library()
        g2p = EspeakG2P("en-us")
        phones = g2p.word_phones("hello")[0].phones
        return f"library={source}, 'hello' -> {' '.join(phones)}"
    check("espeak-ng (English G2P)", _espeak)

    def _lao():
        from phonalign.g2p import EpitranLaoG2P
        g2p = EpitranLaoG2P()
        words = g2p.word_phones("ສະບາຍດີ")
        rendered = "; ".join(f"{w.word} -> {' '.join(w.phones)}" for w in words)
        return rendered
    check("epitran + laonlp (Lao G2P)", _lao)

    def _audio():
        import soundfile, soxr  # noqa: F401
        return None
    check("soundfile + soxr (audio I/O)", _audio)

    def _praatio():
        import praatio  # noqa: F401
        return None
    check("praatio (TextGrid output)", _praatio)

    def _model():
        from huggingface_hub import snapshot_download
        if download_model:
            path = snapshot_download(MODEL_ID)
            return f"downloaded to {path}"
        try:
            snapshot_download(MODEL_ID, local_files_only=True)
            return "cached locally"
        except Exception:
            raise RuntimeError(
                f"{MODEL_ID} not cached yet — first `phonalign align` run will download "
                f"~1.3 GB, or run `phonalign doctor --download-model` now"
            )
    check(f"acoustic model ({MODEL_ID})", _model)

    if not sys.flags.utf8_mode:
        console.print(
            "\n[yellow]hint:[/yellow] Python UTF-8 mode is off. phonalign handles this "
            "internally, but setting the env var PYTHONUTF8=1 avoids encoding surprises "
            "in your own scripts."
        )

    console.print()
    if ok:
        console.print("[bold green]All checks passed.[/bold green]")
    else:
        console.print("[bold red]Some checks failed — see above.[/bold red]")
        raise typer.Exit(1)


@app.command()
def version():
    """Print version."""
    from phonalign import __version__

    console.print(f"phonalign {__version__}")


def main():
    app()


if __name__ == "__main__":
    main()
