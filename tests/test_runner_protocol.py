# MIT License — DFAH-Bench
"""Regression tests for the benchmark runner replay protocol.

These tests pin the fixes from the 2026-06 hardening audit:

  C1: the Anthropic runner must pass temperature=0.0 explicitly
      (previously omitted -> provider default 1.0 while logs claimed 0.0).
  C2: logged sampling metadata must come from sampling_params_for(), the
      same source used to build the request, so logs cannot drift.
  C4: determinism-correction re-logs must not destroy tool outputs
      (previously overwrote both light and _full logs with empty outputs).
  H1: parse-failure fallback decisions must be flagged via decision_source.
  H5: decision extraction must be word-boundary based ("approve" must not
      match inside "disapprove").

They are pure-unit tests: no network, no API keys, no model downloads.
The runner module lives in econometrics/benchmarks/; tests locate it
relative to the repo root regardless of where pytest is invoked.
"""

import json
import sys
from pathlib import Path

import pytest

# Locate econometrics/benchmarks next to this tests/ directory's parent.
# Works both in the repo root layout and the staged artifact layout.
_HERE = Path(__file__).resolve().parent
for _cand in (_HERE.parent, _HERE.parent.parent):
    bench_dir = _cand / "econometrics" / "benchmarks"
    if bench_dir.is_dir():
        sys.path.insert(0, str(bench_dir))
        break
else:
    pytest.skip(
        "econometrics/benchmarks not found relative to tests/",
        allow_module_level=True,
    )

import run_logger as run_logger_module  # noqa: E402
import run_unified_benchmark as rub  # noqa: E402
from run_logger import RunLogger  # noqa: E402


# ---------------------------------------------------------------------------
# H5 — word-boundary decision extraction
# ---------------------------------------------------------------------------

class TestExtractDecision:
    DECISIONS = ["approve", "reject", "modify"]

    def test_marker_extraction(self):
        assert rub._extract_decision("DECISION: APPROVE", self.DECISIONS) == "approve"

    def test_substring_does_not_false_positive(self):
        # "disapprove" contains "approve" as a substring; word-boundary
        # matching must not match it.
        text = "I would disapprove of rushing here, but no decision yet."
        assert rub._extract_decision(text, self.DECISIONS) is None

    def test_last_marker_wins(self):
        text = "DECISION: APPROVE ... wait, on reflection. DECISION: REJECT"
        assert rub._extract_decision(text, self.DECISIONS) == "reject"

    def test_underscore_labels_with_punctuation(self):
        decisions = ["auto_fix", "escalate", "quarantine"]
        assert rub._extract_decision("My decision is AUTO_FIX.", decisions) == "auto_fix"

    def test_tail_window_scan(self):
        text = ("lots of analysis " * 50) + "therefore I reject this trade"
        assert rub._extract_decision(text, self.DECISIONS) == "reject"

    def test_empty_text(self):
        assert rub._extract_decision("", self.DECISIONS) is None
        assert rub._extract_decision(None, self.DECISIONS) is None


# ---------------------------------------------------------------------------
# C2 — single source of truth for sampling params
# ---------------------------------------------------------------------------

class TestSamplingParams:
    def test_ollama(self):
        p = rub.sampling_params_for("ollama")
        assert p["temperature"] == 0.0
        assert p["seed"] == 42

    def test_anthropic_temperature_zero_no_seed(self):
        p = rub.sampling_params_for("anthropic")
        assert p["temperature"] == 0.0
        assert p["seed"] is None  # Anthropic API exposes no seed parameter

    def test_gemini(self):
        p = rub.sampling_params_for("gemini")
        assert p["temperature"] == 0.0
        assert p["seed"] is None

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError):
            rub.sampling_params_for("watsonx-nonexistent")


# ---------------------------------------------------------------------------
# C1 + H1 — Anthropic runner passes temperature; fallback is flagged
# ---------------------------------------------------------------------------

class _Block:
    def __init__(self, type_, text=None):
        self.type = type_
        self.text = text


class _Resp:
    def __init__(self, text, stop_reason="end_turn"):
        self.content = [_Block("text", text)]
        self.stop_reason = stop_reason


class _RecordingMessages:
    def __init__(self, reply_text):
        self.calls = []
        self._reply_text = reply_text

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return _Resp(self._reply_text)


class _StubClient:
    def __init__(self, reply_text):
        self.messages = _RecordingMessages(reply_text)


def _benchmark_cfg():
    return {
        "system_prompt": "You are a compliance agent.",
        "case_formatter": lambda case: f"Alert: {case['alert_id']}",
        "execute_tool": lambda name, args: {"ok": True},
        "decisions": ["escalate", "dismiss", "investigate"],
        "tools_anthropic": [],
    }


class TestAnthropicRunnerProtocol:
    def test_temperature_passed_explicitly(self):
        client = _StubClient("DECISION: ESCALATE")
        rub.run_agent_anthropic(
            client, "claude-test", {"alert_id": "A-1"}, _benchmark_cfg()
        )
        assert len(client.messages.calls) >= 1
        for call in client.messages.calls:
            assert call.get("temperature") == 0.0, (
                "Anthropic call missing temperature=0.0 — replay protocol "
                "violation (audit finding C1)"
            )

    def test_parsed_decision_source(self):
        client = _StubClient("DECISION: ESCALATE")
        result = rub.run_agent_anthropic(
            client, "claude-test", {"alert_id": "A-1"}, _benchmark_cfg()
        )
        assert result["decision"] == "escalate"
        assert result["decision_source"] == "parsed"

    def test_fallback_decision_is_flagged(self):
        client = _StubClient("I cannot determine anything conclusive here.")
        result = rub.run_agent_anthropic(
            client, "claude-test", {"alert_id": "A-1"}, _benchmark_cfg()
        )
        # Fallback assigns the last ontology label, but MUST flag it.
        assert result["decision"] == "investigate"
        assert result["decision_source"] == "fallback_last_ontology_label"


# ---------------------------------------------------------------------------
# C4 — RunLogger must never destroy a richer _full log
# ---------------------------------------------------------------------------

class TestRunLoggerEvidencePreservation:
    def _logger(self, tmp_path, monkeypatch):
        # Redirect ALL logger filesystem activity (including the mkdir in
        # __init__) into the pytest tmp dir so tests never touch the repo.
        monkeypatch.setattr(run_logger_module, "_results_root", lambda: tmp_path)
        logger = RunLogger(benchmark="testbench", model="test-model")
        logger.log_dir = tmp_path
        return logger

    def _log(self, logger, outputs, deterministic, note=None):
        return logger.log_run(
            case_id="CASE-1", run_id=0,
            seed=None, temperature=0.0,
            tool_sequence=["tool_a", "tool_b"],
            tool_outputs=outputs,
            decision_output="escalate",
            deterministic=deterministic,
            faithfulness_score=0.0,
            runtime_seconds=1.5,
            extra={"note": note} if note else None,
        )

    def test_correction_relog_keeps_full_outputs(self, tmp_path, monkeypatch):
        logger = self._logger(tmp_path, monkeypatch)
        original_outputs = [{"risk": "high"}, {"sanctions": ["X"]}]

        # First pass: low faithfulness -> _full.json written with outputs.
        self._log(logger, original_outputs, deterministic=True)
        full_path = tmp_path / "case_CASE-1_run_0_full.json"
        assert full_path.exists()
        assert json.loads(full_path.read_text())["tool_outputs"] == original_outputs

        # Correction pass with empty outputs (legacy behavior) must NOT
        # clobber the richer existing full log.
        self._log(logger, [], deterministic=False, note="deterministic_correction")
        assert json.loads(full_path.read_text())["tool_outputs"] == original_outputs, (
            "deterministic_correction re-log destroyed the evidence channel "
            "(audit finding C4)"
        )

    def test_correction_with_real_outputs_updates_full(self, tmp_path, monkeypatch):
        logger = self._logger(tmp_path, monkeypatch)
        self._log(logger, [{"v": 1}], deterministic=True)
        # A re-log that carries real outputs may overwrite.
        self._log(logger, [{"v": 2}], deterministic=False)
        full_path = tmp_path / "case_CASE-1_run_0_full.json"
        assert json.loads(full_path.read_text())["tool_outputs"] == [{"v": 2}]

    def test_light_log_hashes_match_outputs(self, tmp_path, monkeypatch):
        logger = self._logger(tmp_path, monkeypatch)
        self._log(logger, [{"v": 1}], deterministic=True)
        light = json.loads((tmp_path / "case_CASE-1_run_0.json").read_text())
        assert len(light["tool_output_hashes"]) == 1
        assert light["tool_sequence"] == ["tool_a", "tool_b"]
