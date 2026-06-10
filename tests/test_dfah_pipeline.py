# MIT License — DFAH-Bench
"""Regression tests for the replay-analysis pipeline conventions.

Pins the denominator conventions and aggregation weighting that the
published Table 1 / kill-criterion numbers depend on (audit finding H2):

  - TAR is computed over runs WITH tool calls; zero-tool runs are treated
    as missing the trajectory channel, not as empty sequences.
  - A group where NO run made tool calls has TAR = None (channel absent),
    never TAR = 1.0.
  - DAR is computed over runs with a non-empty decision label.
  - Model-level aggregation is task-averaged (mean of per-benchmark means),
    not pooled across case groups.
  - Kill criterion eligibility: DAR >= 0.9 AND trajectory data present.

All fixtures are synthetic with hand-computed expected values.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_ROOT))           # bench/
sys.path.insert(0, str(_ROOT / "scripts"))  # compute_dfah_metrics

import compute_dfah_metrics as cdm  # noqa: E402
from bench.spec.schema import (  # noqa: E402
    Decision,
    ReplayEpisode,
    RunMetadata,
    ToolCall,
)


def _episode(case_id, run_id, decision, tools=(), benchmark="compliance",
             model="test-model"):
    return ReplayEpisode(
        case_id=case_id,
        benchmark=benchmark,
        run_id=run_id,
        decision=Decision(label=decision),
        runtime_seconds=1.0,
        metadata=RunMetadata(model_name=model, temperature=0.0, seed=42),
        tool_calls=[ToolCall(name=t, output_hash=f"h{i}") for i, t in enumerate(tools)]
        if tools else None,
        evidence_contacts=None,
        reasoning_trace=None,
    )


# ---------------------------------------------------------------------------
# TAR denominator conventions (H2)
# ---------------------------------------------------------------------------

class TestTarConventions:
    def test_tar_exact_match_hand_computed(self):
        eps = [
            _episode("C1", 0, "escalate", tools=("a", "b")),
            _episode("C1", 1, "escalate", tools=("a", "b")),
            _episode("C1", 2, "escalate", tools=("a", "c")),
        ]
        tar, n_unique, modal = cdm._compute_tar(eps)
        assert tar == pytest.approx(2 / 3)
        assert n_unique == 2
        assert modal == "a -> b"

    def test_zero_tool_runs_excluded_from_denominator(self):
        # 2 runs with identical sequences + 1 run with no tools:
        # published convention -> TAR = 2/2 = 1.0, not 2/3.
        eps = [
            _episode("C1", 0, "escalate", tools=("a",)),
            _episode("C1", 1, "escalate", tools=("a",)),
            _episode("C1", 2, "escalate", tools=()),
        ]
        tar, n_unique, _ = cdm._compute_tar(eps)
        assert tar == 1.0
        assert n_unique == 1

    def test_all_zero_tool_group_has_no_trajectory_channel(self):
        eps = [_episode("C1", i, "escalate", tools=()) for i in range(3)]
        tar, n_unique, modal = cdm._compute_tar(eps)
        assert tar is None
        assert n_unique == 0
        assert modal is None

    def test_modal_tie_is_deterministic(self):
        # Two sequences each appearing twice: Counter.most_common resolves
        # ties by insertion (first-seen) order — pin that behavior.
        eps = [
            _episode("C1", 0, "escalate", tools=("a",)),
            _episode("C1", 1, "escalate", tools=("b",)),
            _episode("C1", 2, "escalate", tools=("a",)),
            _episode("C1", 3, "escalate", tools=("b",)),
        ]
        tar, n_unique, modal = cdm._compute_tar(eps)
        assert tar == 0.5
        assert n_unique == 2
        assert modal == "a"  # first-seen wins the tie


# ---------------------------------------------------------------------------
# DAR / case-metric conventions
# ---------------------------------------------------------------------------

class TestCaseMetrics:
    KEY = ("C1", "compliance", "test-model")

    def test_dar_hand_computed(self):
        eps = [
            _episode("C1", 0, "escalate", tools=("a",)),
            _episode("C1", 1, "escalate", tools=("a",)),
            _episode("C1", 2, "dismiss", tools=("a",)),
        ]
        row = cdm.compute_case_metrics(self.KEY, eps)
        assert row["dar"] == pytest.approx(2 / 3)
        assert row["modal_decision"] == "escalate"
        assert row["n_unique_decisions"] == 2
        assert row["n_runs"] == 3

    def test_empty_decisions_excluded_from_dar(self):
        eps = [
            _episode("C1", 0, "escalate", tools=("a",)),
            _episode("C1", 1, "escalate", tools=("a",)),
            _episode("C1", 2, "", tools=("a",)),  # missing decision channel
        ]
        row = cdm.compute_case_metrics(self.KEY, eps)
        # Convention: DAR over non-empty labels -> 2/2, not 2/3.
        assert row["dar"] == 1.0

    def test_gap_requires_both_channels(self):
        eps = [_episode("C1", i, "escalate", tools=()) for i in range(3)]
        row = cdm.compute_case_metrics(self.KEY, eps)
        assert row["tar"] is None
        assert row["dar_tar_gap"] is None
        assert row["has_trajectory"] is False

    def test_dar_tar_gap_hand_computed(self):
        # DAR = 3/3 = 1.0; TAR = 2/3 -> gap = 1/3. The canonical
        # "same conclusion, different trajectory" shape.
        eps = [
            _episode("C1", 0, "escalate", tools=("a", "b")),
            _episode("C1", 1, "escalate", tools=("a", "b")),
            _episode("C1", 2, "escalate", tools=("b", "a")),
        ]
        row = cdm.compute_case_metrics(self.KEY, eps)
        assert row["dar"] == 1.0
        assert row["tar"] == pytest.approx(2 / 3)
        assert row["dar_tar_gap"] == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# Kill criterion (overall + per-model CSV)
# ---------------------------------------------------------------------------

def _case_row(model, dar, tar, n_unique_traj, has_traj=True):
    return {
        "case_id": "X", "benchmark": "compliance", "model": model,
        "n_runs": 3, "dar": dar, "modal_decision": "escalate",
        "n_unique_decisions": 1, "dcb": 1.0,
        "has_trajectory": has_traj, "tar": tar,
        "n_unique_trajectories": n_unique_traj,
        "dar_tar_gap": (dar - tar) if tar is not None else None,
        "has_evidence": False, "ecd": None, "scde": None,
    }


class TestKillCriterion:
    def test_eligibility_and_thresholds(self):
        rows = [
            _case_row("m1", 1.0, 1.0, 1),     # eligible, no divergence
            _case_row("m1", 1.0, 0.8, 2),     # eligible, moderate (0.7<=TAR<0.9)
            _case_row("m1", 0.95, 0.6, 3),    # eligible, strong (TAR<0.7)
            _case_row("m1", 0.85, 0.5, 3),    # NOT eligible (DAR < 0.9)
            _case_row("m1", 1.0, None, 0, has_traj=False),  # no trajectory
        ]
        kill_df, by_model = cdm.evaluate_kill_criterion(pd.DataFrame(rows))

        overall = {r["criterion"]: r for _, r in kill_df.iterrows()}
        assert overall["any_variation"]["n_eligible"] == 3
        assert overall["any_variation"]["n_divergent"] == 2
        assert overall["moderate_tar_lt_0.9"]["n_divergent"] == 2
        assert overall["strong_tar_lt_0.7"]["n_divergent"] == 1

        m1 = by_model[by_model["model"] == "m1"].iloc[0]
        assert m1["n_eligible"] == 3
        assert m1["n_any_variation"] == 2
        assert m1["pct_any_variation"] == pytest.approx(200 / 3)
        assert m1["n_strong_tar_lt_0.7"] == 1

    def test_empty_input(self):
        kill_df, by_model = cdm.evaluate_kill_criterion(
            pd.DataFrame([_case_row("m1", 0.5, 0.5, 2)])  # nothing eligible
        )
        assert len(kill_df) == 0
        assert len(by_model) == 0


# ---------------------------------------------------------------------------
# Aggregation weighting: task-averaged, not pooled
# ---------------------------------------------------------------------------

class TestAggregationWeighting:
    def test_model_mean_is_benchmark_mean_not_pooled(self):
        # Benchmark A: 3 cases at DAR 1.0; benchmark B: 1 case at DAR 0.5.
        # Pooled mean = (3*1.0 + 0.5)/4 = 0.875.
        # Task-averaged   = (1.0 + 0.5)/2 = 0.75  <- published convention.
        rows = (
            [_case_row("m1", 1.0, 1.0, 1) for _ in range(3)]
            + [{**_case_row("m1", 0.5, 0.5, 2), "benchmark": "portfolio"}]
        )
        case_df = pd.DataFrame(rows)
        task_df = cdm.aggregate_task_level(case_df)
        model_df = cdm.aggregate_model_level(task_df)
        assert model_df.iloc[0]["mean_dar"] == pytest.approx(0.75)
        assert model_df.iloc[0]["n_benchmarks"] == 2
        assert int(model_df.iloc[0]["total_episodes"]) == 12
