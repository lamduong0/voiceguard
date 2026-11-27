"""Regression coverage for the recording checker and its public entry points."""
import contextlib
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

import cli
import check
from voiceguard import assess as assessment, detect


@pytest.fixture(autouse=True)
def offline_intent(monkeypatch):
    monkeypatch.delenv("VG_LLM_KEY", raising=False)


@pytest.fixture
def recording(monkeypatch):
    waveform = np.full(32000, 0.1, dtype=np.float32)
    decoder = Mock(return_value=waveform)
    transcriber = Mock(return_value=[{"t": 0.0, "text": "gift card right now"}])
    scorer = Mock(score_windows=Mock(return_value=[(0, 0.9), (1, 0.9)]))
    monkeypatch.setattr(assessment, "load_audio", decoder)
    monkeypatch.setattr(assessment, "transcribe", transcriber)
    monkeypatch.setattr(assessment, "_get_scorer", lambda: scorer)
    return waveform, decoder, transcriber, scorer


@pytest.mark.parametrize("missing", ["speech", "acoustic", "both"])
def test_missing_evidence_never_produces_a_risk_level(recording, missing):
    _, _, transcriber, scorer = recording
    if missing in ("speech", "both"):
        transcriber.return_value = []
    if missing in ("acoustic", "both"):
        scorer.score_windows.return_value = []
    result = assessment.assess("call.wav")
    assert result["level"] == "INSUFFICIENT_EVIDENCE"
    assert result["risk"] is None
    assert result["reasons"]
    if missing == "speech":
        assert result["clone_score"] > 0.9
        assert result["intent"] is None
    if missing == "acoustic":
        assert result["intent"] == 1.0
        assert result["clone_score"] is None


def test_short_recording_preserves_duration_and_skips_inference(recording):
    _, decoder, transcriber, scorer = recording
    decoder.return_value = np.zeros(4000, dtype=np.float32)
    result = assessment.assess("short.wav")
    assert result["duration_s"] == 0.25
    assert result["risk"] is None
    transcriber.assert_not_called()
    scorer.score_windows.assert_not_called()


def test_both_models_receive_the_same_decoded_waveform(recording):
    waveform, decoder, transcriber, scorer = recording
    result = assessment.assess("call.m4a")
    decoder.assert_called_once_with("call.m4a")
    assert transcriber.call_args.args[0] is waveform
    assert scorer.score_windows.call_args.args[0] is waveform
    assert result["level"] == "HIGH"
    assert result["clone_score"] == result["p_clone"]


@pytest.mark.parametrize("stage", ["decode", "transcription", "acoustic"])
def test_pipeline_failure_is_explained(recording, stage):
    _, decoder, transcriber, scorer = recording
    target = {"decode": decoder, "transcription": transcriber, "acoustic": scorer.score_windows}[stage]
    target.side_effect = ValueError("fixture failure")
    result = assessment.assess("broken.wav")
    assert result["risk"] is None
    assert result["level"] == "INSUFFICIENT_EVIDENCE"
    assert "fixture failure" in " ".join(result["reasons"])


def test_container_without_an_audio_stream_is_inconclusive(monkeypatch):
    decoder = Mock(side_effect=IndexError("tuple index out of range"))
    monkeypatch.setitem(sys.modules, "faster_whisper.audio", SimpleNamespace(decode_audio=decoder))
    result = assessment.assess("empty-container.m4a")
    assert result["level"] == "INSUFFICIENT_EVIDENCE"
    assert result["risk"] is None
    assert "no decodable audio stream" in result["reasons"][0]


def test_cli_final_verdict_uses_the_last_llm_score(monkeypatch, capsys):
    monkeypatch.setattr(cli, "score_intent", Mock(return_value=(1.0, "llm")))
    cli.play(0.9, [{"text": "Hello"}], {"c": 0.4, "g": 7}, json_out=True, use_llm=True)
    result = json.loads(capsys.readouterr().out)
    assert result["final_risk"] == result["trace"][-1]["risk"]
    assert result["final_level"] == result["trace"][-1]["level"] == "HIGH"
    assert result["intent_source"] == "llm"
    cli.score_intent.assert_called_once()


@pytest.mark.parametrize("turns", [[], [{"text": "  "}]])
def test_cli_missing_transcript_is_inconclusive(turns, capsys):
    cli.play(0.9, turns, {"c": 0.4, "g": 7}, json_out=True)
    result = json.loads(capsys.readouterr().out)
    assert result["final_risk"] is None
    assert result["final_level"] == "INSUFFICIENT_EVIDENCE"


def test_check_prints_unavailable_scores(recording, monkeypatch, capsys):
    recording[2].return_value = []
    monkeypatch.setattr(sys, "argv", ["check.py", "silent.wav"])
    check.main()
    output = capsys.readouterr().out
    assert "INSUFFICIENT_EVIDENCE" in output
    assert "unavailable" in output
    assert "P(clone)" not in output


def test_web_result_handles_missing_scores(recording, monkeypatch):
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    pytest.importorskip("gradio")
    import app
    recording[2].return_value = []
    recording[3].score_windows.return_value = []
    verdict, transcript = app.run("silent.wav")
    assert "INSUFFICIENT EVIDENCE" in verdict
    assert "Voice-clone scam score **unavailable**" in verdict
    assert "No usable acoustic windows" in verdict
    assert transcript == "(no speech detected)"


@pytest.mark.parametrize("response, expected", [
    ("0.83", 0.83), (" 1.0\n", 1.0), ("0", 0.0),
    ("Score out of 1: 0.83", None), ("10", None), ("1.2", None),
    ("-0.2", None), ("nan", None), ("inf", None), ("", None),
])
def test_llm_requires_a_complete_finite_score(monkeypatch, response, expected):
    import openai
    reply = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=response))])
    client = Mock()
    client.chat.completions.create.return_value = reply
    monkeypatch.setattr(openai, "OpenAI", Mock(return_value=client))
    monkeypatch.setenv("VG_LLM_KEY", "test-placeholder")
    assert detect.llm_intent([{"text": "Hello"}]) == expected


def test_keyword_matching_uses_word_boundaries_and_lowercase():
    assert detect.keyword_intent(["wireless router"]) == 0.05
    assert detect.keyword_intent(["I called the IRS"]) == 0.65
    assert detect.keyword_intent(["wire the money"]) == 0.65


def test_calibration_survives_a_different_working_directory(monkeypatch, tmp_path):
    expected = detect.load_calibration(model=assessment.M2, source="itw")
    monkeypatch.chdir(tmp_path)
    assert detect.load_calibration(model=assessment.M2, source="itw") == expected
    assert expected["revision"] == detect.MODEL_REVISIONS[assessment.M2]
    assert expected["c"] == 0.4102
    with pytest.raises(FileNotFoundError):
        detect.load_calibration("missing.json")


def test_packaged_calibration_matches_its_source_data():
    import hashlib
    root = Path(__file__).resolve().parents[1]
    for model, source in [(detect.DEFAULT_MODEL, "asv"), (assessment.M2, "itw")]:
        packaged = detect.load_calibration(model=model, source=source)
        path = root / packaged["data_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == packaged["data_sha256"]
        fitted = detect.load_calibration(path, source=source)
        assert {key: packaged[key] for key in fitted} == fitted


def test_transcription_enables_speech_detection(monkeypatch):
    model = Mock()
    model.transcribe.return_value = ([], None)
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=Mock(return_value=model)))
    assert detect.transcribe(np.full(16000, 0.1, dtype=np.float32)) == []
    assert model.transcribe.call_args.kwargs["vad_filter"] is True


def test_long_recording_reports_the_peak_without_overrunning_duration(recording):
    _, decoder, _, scorer = recording
    decoder.return_value = np.ones(61 * 16000 + 8000, dtype=np.float32)
    scorer.score_windows.return_value = [(i, 0.1 if i < 45 else 0.9) for i in range(62)]
    result = assessment.assess("long.wav")
    assert result["peak_segment"] == {"start": 45, "end": 61.5}
    assert result["acoustic"] == 0.9
    assert result["duration_s"] == 61.5


@pytest.mark.parametrize("extension", ["wav", "mp3", "m4a"])
def test_audio_formats_reach_the_acoustic_scorer(tmp_path, extension):
    pytest.importorskip("faster_whisper.audio")
    original = tmp_path / "signal.wav"
    samples = (np.sin(2 * np.pi * 440 * np.arange(16000) / 16000) * 16000).astype("<i2")
    with wave.open(str(original), "wb") as handle:
        handle.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        handle.writeframes(samples.tobytes())
    audio = original
    if extension != "wav":
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            pytest.skip("ffmpeg is needed to create compressed test fixtures")
        audio = tmp_path / f"signal.{extension}"
        subprocess.run([ffmpeg, "-v", "error", "-i", str(original), str(audio)], check=True)
    waveform = detect.load_audio(audio)
    assert waveform.dtype == np.float32
    assert waveform.ndim == 1
    assert 15000 <= len(waveform) <= 18000
    scorer = object.__new__(detect.AcousticScorer)
    scorer.fe = lambda samples, **kwargs: {}
    scorer.m = lambda **kwargs: SimpleNamespace(logits=np.array([[0.2, 0.8]]))
    scorer.torch = SimpleNamespace(no_grad=contextlib.nullcontext, softmax=lambda logits, dim: logits)
    scorer.fake_idx = 1
    assert scorer.score_windows(audio)[0] == (0, 0.8)
