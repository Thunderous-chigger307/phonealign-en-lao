# phonalign

Audio-to-phoneme forced alignment for TTS dataset preprocessing.

Point it at a folder of audio + transcripts and it produces training-ready
artifacts for VITS/VITS2, FastSpeech2, and other TTS architectures:

- **Praat TextGrids** — word + phone tiers, inspectable in Praat
- **JSON manifests** — per-utterance phones with timestamps and confidence scores, plus a corpus-level `manifest.jsonl`
- **VITS filelists** — pipe-delimited `wav|phonemes` (or `wav|speaker|phonemes`) with train/val split
- **Duration arrays** — per-phoneme mel-frame counts (`.npy`) that sum *exactly* to the utterance frame count, for duration-supervised training
- **QA report** — `report.csv` flagging utterances whose alignment confidence suggests a transcript/audio mismatch

Everything installs with pip — no Montreal Forced Aligner, no conda, no system
espeak install. Works on Windows, CPU-only is fine (GPU optional via `--device cuda`).

## Languages

| Language | Code | G2P engine |
|---|---|---|
| English (US) | `en-us` | espeak-ng (bundled via `espeakng-loader`) |
| Lao | `lo` | epitran `lao-Laoo` + laonlp word segmentation |
| ~100 others | any espeak-ng code | espeak-ng |

Alignment uses `facebook/wav2vec2-lv-60-espeak-cv-ft`, a multilingual wav2vec2
CTC model over espeak IPA phones (~1.3 GB, downloaded once on first use), with
a pure-PyTorch CTC Viterbi forced aligner. For languages outside the model's
training set (like Lao) alignment is zero-shot: boundaries are usable but
confidence scores run lower — check the QA report and spot-check TextGrids in Praat.

## Install

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -e .
phonalign doctor --download-model   # verify setup + pre-download the model
```

## Usage

Corpus layouts accepted:

```
corpus/                      corpus/
├── utt1.wav                 ├── metadata.csv    # id|text  or  id|text|normalized
├── utt1.txt                 └── wavs/
├── utt2.wav                     ├── utt1.wav    # LJSpeech layout
└── utt2.lab                     └── utt2.wav
```

```bash
# English corpus -> all four output formats
phonalign align -i corpus/ -o out/ -l en-us --val-count 100

# Lao
phonalign align -i lao_corpus/ -o out_lo/ -l lo

# Duration frames matched to your TTS config (default 22050 Hz, hop 256)
phonalign align -i corpus/ -o out/ --sample-rate 24000 --hop-length 300 -f durations,json

# Multi-speaker metadata.csv (id|speaker|text)
phonalign align -i corpus/ -o out/ --speaker-column
```

Output tree:

```
out/
├── textgrid/<id>.TextGrid
├── json/<id>.json           # phones/words with start, end, score
├── manifest.jsonl
├── durations/<id>.npy       # int64 frame counts, gap-filled with "sil"
├── durations/<id>.json      # phone labels matching the .npy entries
├── vits/filelist_{all,train,val}.txt
└── report.csv
```

### Python API

```python
from phonalign import Aligner

aligner = Aligner(lang="en-us")           # or lang="lo", device="cuda"
result = aligner.align("utt1.wav", "Printing in the only sense.")
for p in result.phones:
    print(p.label, p.start, p.end, p.score)
for w in result.words:
    print(w.label, w.start, w.end)
timeline = result.full_timeline()          # gap-free, with "sil" phones
```

Writers are importable too: `phonalign.writers.write_textgrid`,
`write_durations`, `VitsFilelistWriter`, `ManifestWriter`.

### Feeding VITS

`vits/filelist_train.txt` lines look like:

```
wavs/utt1.wav|pɹɪntɪŋ ɪn ðə oʊnli sɛns
```

Point your VITS config's `training_files`/`validation_files` at the filelists
and use a pass-through text cleaner (the text is already IPA), e.g.
`"text_cleaners": []` with `"cleaned_text": true` in the config.

## How it works

1. **G2P**: text → IPA phones per word (espeak-ng or epitran; Lao text is
   word-segmented with laonlp first since Lao script has no spaces).
2. **Vocab mapping**: each phone is mapped onto one or more of the acoustic
   model's 392 espeak phone tokens (per-language `LANG_TOKEN_OVERRIDES` →
   exact match → stress/length-stripped → greedy longest-match segmentation →
   `FALLBACK_PHONE_MAP`). Unmappable phones raise `UnmappablePhoneError`
   naming the offending phone and word; the CLI catches it, skips that one
   utterance into `report.csv`, and continues the batch. The overrides exist because the model concentrates its
   probability mass on the token spellings its training transcripts used —
   e.g. for Lao it emits espeak's `ph`/`th`/`x` and tone-marked vowels
   (`i5`, `ɑ5`), so aligning against those instead of the exact-IPA tokens
   (`pʰ`, `iː`) gives markedly better boundaries and confidence. Output
   labels always keep the G2P phone.
3. **Alignment**: wav2vec2 CTC log-probs (20 ms frames) + Viterbi forced
   alignment over the blank-interleaved token sequence; token spans are merged
   back into phone and word intervals with per-phone confidence.
4. **Outputs**: timestamps are converted to your TTS sample rate; duration
   rounding is cumulative so frame counts always sum to the utterance total.

## Troubleshooting

- **`phonalign doctor`** diagnoses most setup issues (espeak library, model
  cache, audio I/O).
- Console shows `?` instead of IPA: the terminal font lacks IPA glyphs —
  files on disk are unaffected (always UTF-8).
- An utterance fails (bad audio, `UnmappablePhoneError`, ...): the CLI skips
  it, records an `error` row in `report.csv` with the reason, and keeps
  going — one weird transcript never kills a large batch run. Filter
  `report.csv` by status to find and fix the skipped utterances. For
  unmappable phones specifically, add a mapping to
  `phonalign.align.FALLBACK_PHONE_MAP` (map the phone to the closest tokens
  in the model vocab) — and please report it.
- A specific phone consistently scores ~0 in the JSON output even though the
  audio is clean: the model probably prefers a different vocab token for that
  sound. Greedy-decode the emissions to see what the model hears, then add a
  per-language entry to `phonalign.align.LANG_TOKEN_OVERRIDES`.
- Many `flagged` rows in `report.csv`: transcripts likely don't match the
  audio (wrong file pairing, heavy noise), or the language is far from the
  model's training data (expected for zero-shot languages like Lao).

## Development

```bash
pip install -e .[dev]
pytest
```
