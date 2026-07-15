from phonalign import qa
from phonalign.align import AlignmentResult, Phone
from phonalign.errors import UnmappablePhoneError


def make_result(scores):
    phones = [Phone(f"p{i}", i * 0.1, (i + 1) * 0.1, s) for i, s in enumerate(scores)]
    return AlignmentResult(
        phones=phones, words=[], audio_duration=len(scores) * 0.1, language="en-us", text=""
    )


def test_evaluate_ok():
    row = qa.evaluate("utt1", make_result([0.9, 0.8, 0.7]))
    assert row.status == "ok"
    assert row.n_phones == 3
    assert row.detail == ""


def test_evaluate_flagged_below_threshold():
    row = qa.evaluate("utt1", make_result([0.2, 0.3]))
    assert row.status == "flagged"
    assert "confidence" in row.detail


def test_error_row_unmappable_phone_is_structured():
    exc = UnmappablePhoneError(phone="ɸ", word="ຝັນ", language="lo")
    row = qa.error_row("utt7", exc)
    assert row.status == "error"
    assert "'ɸ'" in row.detail
    assert "'ຝັນ'" in row.detail
    assert "lo" in row.detail
    assert "FALLBACK_PHONE_MAP" in row.detail


def test_error_row_generic_exception_keeps_type():
    row = qa.error_row("utt8", ValueError("bad wav header"))
    assert row.status == "error"
    assert row.detail == "ValueError: bad wav header"


def test_error_rows_survive_report_roundtrip(tmp_path):
    rows = [
        qa.evaluate("good", make_result([0.9])),
        qa.error_row("bad", UnmappablePhoneError("ɸ", "x", "xx")),
    ]
    path = qa.write_report(rows, tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "good,ok" in text
    assert "bad,error" in text
