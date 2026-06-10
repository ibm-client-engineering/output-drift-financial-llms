#!/usr/bin/env python3
# MIT License — DFAH-Bench
"""Domain-extension scaffold: medical triage on DFAH-Bench, zero metric changes.

The paper (§3.1) claims the benchmark design is domain-agnostic:

    "the metrics (DAR, TAR, ECD, DCB) are defined over decisions, tool
     sequences, and evidence sets — not over finance-specific structures.
     For example, a medical triage task (escalate / treat / refer, with
     tools such as check_drug_interactions and get_patient_history) ...
     could be evaluated using the same replay protocol and metrics
     without modification."

This script PROVES that claim executable: it registers exactly that medical
triage task — a new decision ontology and mock tools — and computes every
DFAH metric with the unmodified bench/ library. There is no medical-specific
code anywhere in bench/metrics or bench/spec; the only new code is in this
file.

Usage:
    python examples/domain_extension_medical.py

Expected output: a per-case DFAH readout (DAR, TAR, gap, ECD, DCB) for a
synthetic medical agent that is intentionally decision-stable but
trajectory-unstable on one case — the paper's central failure mode,
reproduced in a brand-new domain.
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --- DFAH-Bench library: imported UNMODIFIED -------------------------------
from bench.metrics.dcb import compute_dcb
from bench.metrics.ecd import compute_ecd
from bench.spec.taxonomy import (
    DecisionOntology,
    TaskSpec,
    get_k,
    register_task,
    validate_decision,
)

# ---------------------------------------------------------------------------
# Step 1 — define the new domain: ontology + tool surface
# ---------------------------------------------------------------------------

MEDICAL_ONTOLOGY = DecisionOntology(
    name="medical_triage",
    categories=["escalate", "treat", "refer"],
    description="Triage a patient presentation: escalate to emergency care, "
    "treat in place, or refer to a specialist.",
)

MEDICAL_SPEC = TaskSpec(
    task_id="medical_triage",
    name="Medical Triage",
    description="Triage patient presentations using drug-interaction checks "
    "and patient history.",
    ontology=MEDICAL_ONTOLOGY,
    tool_count=2,
    expected_tools=["check_drug_interactions", "get_patient_history"],
)

register_task(MEDICAL_SPEC)

# K now resolves through the standard registry — no metric code touched.
assert get_k("medical_triage") == 3


# ---------------------------------------------------------------------------
# Step 2 — mock tools (deterministic, like the financial benchmarks' mocks)
# ---------------------------------------------------------------------------

def check_drug_interactions(patient_id: str) -> dict:
    """Deterministic mock: same patient -> same interaction report."""
    interactions = {
        "P-001": {"severity": "high", "pairs": ["warfarin+aspirin"]},
        "P-002": {"severity": "none", "pairs": []},
        "P-003": {"severity": "moderate", "pairs": ["lisinopril+ibuprofen"]},
    }
    return interactions.get(patient_id, {"severity": "unknown", "pairs": []})


def get_patient_history(patient_id: str) -> dict:
    """Deterministic mock: same patient -> same history summary."""
    histories = {
        "P-001": {"age": 71, "conditions": ["afib", "hypertension"]},
        "P-002": {"age": 29, "conditions": []},
        "P-003": {"age": 54, "conditions": ["ckd_stage2"]},
    }
    return histories.get(patient_id, {"age": None, "conditions": []})


TOOLS = {
    "check_drug_interactions": check_drug_interactions,
    "get_patient_history": get_patient_history,
}


# ---------------------------------------------------------------------------
# Step 3 — synthetic replay episodes
#
# In a real evaluation these come from N replays of a live agent (see
# econometrics/benchmarks/run_unified_benchmark.py for the protocol). Here
# we synthesize three case groups that exhibit the paper's three profiles:
#   CASE-A: fully stable (same decision, same trajectory)
#   CASE-B: decision-stable but trajectory-unstable (the DAR-TAR gap)
#   CASE-C: decision-unstable (visible to outcome-only evaluation too)
# ---------------------------------------------------------------------------

def _run_episode(patient_id: str, tool_order: list, decision: str) -> dict:
    """Execute mock tools in the given order, returning a replay record."""
    evidence = set()
    for tool_name in tool_order:
        output = TOOLS[tool_name](patient_id)
        for key, value in output.items():
            evidence.add(f"{tool_name}.{key}={value}")
    assert validate_decision(decision, "medical_triage"), decision
    return {"tool_sequence": tuple(tool_order), "evidence": evidence,
            "decision": decision}


REPLAYS = {
    "CASE-A (stable)": [
        _run_episode("P-002", ["get_patient_history"], "treat")
        for _ in range(3)
    ],
    "CASE-B (same decision, different trajectory)": [
        _run_episode("P-001",
                     ["check_drug_interactions", "get_patient_history"],
                     "escalate"),
        _run_episode("P-001",
                     ["get_patient_history", "check_drug_interactions"],
                     "escalate"),
        _run_episode("P-001",
                     ["check_drug_interactions"],
                     "escalate"),
    ],
    "CASE-C (decision-unstable)": [
        _run_episode("P-003", ["get_patient_history"], "refer"),
        _run_episode("P-003", ["check_drug_interactions"], "treat"),
        _run_episode("P-003",
                     ["get_patient_history", "check_drug_interactions"],
                     "refer"),
    ],
}


# ---------------------------------------------------------------------------
# Step 4 — DFAH metrics, computed by the unmodified library
# ---------------------------------------------------------------------------

def main() -> None:
    print("DFAH-Bench domain extension: medical_triage "
          f"(K={get_k('medical_triage')})")
    print("=" * 72)

    all_decisions = []
    for case_name, episodes in REPLAYS.items():
        decisions = [ep["decision"] for ep in episodes]
        sequences = [ep["tool_sequence"] for ep in episodes]
        evidence_sets = [ep["evidence"] for ep in episodes]
        n = len(episodes)
        all_decisions.extend(decisions)

        # DAR / TAR — same definitions as the paper (modal agreement)
        dar = Counter(decisions).most_common(1)[0][1] / n
        tar = Counter(sequences).most_common(1)[0][1] / n

        # ECD + within-case DCB from the bench library, unchanged
        ecd_result = compute_ecd(evidence_sets, decisions=decisions)
        dcb_result = compute_dcb(decisions, benchmark="medical_triage")

        print(f"\n{case_name}")
        print(f"  DAR = {dar:.3f}   TAR = {tar:.3f}   "
              f"gap = {dar - tar:+.3f}")
        print(f"  ECD = {ecd_result.ecd:.3f} "
              f"(n_runs={ecd_result.n_runs}, "
              f"union={ecd_result.union_contact_count} contacts)")
        print(f"  DCB = {dcb_result.dcb:.3f} "
              f"(K={dcb_result.k_categories}, "
              f"H={dcb_result.entropy:.3f}/{dcb_result.max_entropy:.3f})")

    # Cross-case DCB over the whole synthetic corpus
    corpus_dcb = compute_dcb(all_decisions, benchmark="medical_triage")
    print("\n" + "-" * 72)
    print(f"Cross-case DCB over all {corpus_dcb.n_decisions} decisions: "
          f"{corpus_dcb.dcb:.3f}")
    print("\nZero changes were made to bench/metrics or bench/spec to run "
          "this domain.")


if __name__ == "__main__":
    main()
