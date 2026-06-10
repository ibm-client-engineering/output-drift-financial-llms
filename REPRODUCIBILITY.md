# REPRODUCIBILITY — DFAH-Bench

Every number in the DFAH-Bench paper regenerates from the raw replay logs in
this repository with one command. This document records the exact
environment, commands, and known caveats.

## TL;DR

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make reproduce-paper          # full verification incl. B=10,000 bootstrap
make test-bench               # unit/regression test suite
```

`make reproduce-paper` exits non-zero and prints a mismatch table if ANY
published number fails to regenerate. `make reproduce-paper-fast` skips the
bootstrap and subsampling stages for quick iteration.

## Environment

| Component | Version |
|---|---|
| Python | 3.10+ (development and paper numbers produced on CPython 3.11) |
| numpy | 1.26.4 (pinned) |
| pandas | 2.2.3 (pinned) |
| scipy | 1.14.1 (pinned) |
| cryptography | >= 42.0 (provenance layer only) |
| sentence-transformers | >= 2.2.0 (SCDR rationale mode only — NOT needed for any paper number; lazily imported) |
| OS | macOS 14+ / Linux x86_64 (float results identical to numeric tolerance 1e-9) |

No network access, API keys, or model downloads are required to reproduce
the paper numbers — the raw replay corpus is checked in.

## Data: the raw replay corpus

```
econometrics/benchmarks/results/run_logs/{compliance,dataops,portfolio}/<model>/
    case_<id>_run_<n>.json        # one replay episode (8,129 files)
    case_<id>_run_<n>_full.json   # full tool outputs where captured (4,697 files)
```

Episode accounting (asserted by `reproduce_paper.py`):

- 8,129 raw episodes → **8,127 analyzed** (2 single-run case groups excluded;
  they are listed in `results/dfah_skipped_case_groups.csv`)
- **1,338 case groups**, **30 benchmark–model configurations**
- N ∈ {3, 8} replays per case group (3 for API models, 8 for local);
  a handful of groups have intermediate N due to crashed runs — these are
  explicit in the case-level CSV (`n_runs` column), never silently padded
- DeepSeek-R1 8B (14 episodes, compliance only) is excluded from main
  results per the paper footnote; Qwen 3.5 portfolio is partial (3/50 cases)

## Pipeline

`make reproduce-paper` runs, in order:

| Stage | Script | Paper artifact |
|---|---|---|
| Case/task/model metrics + kill criterion | `scripts/compute_dfah_metrics.py` | Table 1 DAR/TAR/Gap/ECD; §4.4 kill criterion (912 / 21.8% / 19.4%; per-model 55.6% Sonnet, 56.6% Gemini Pro) |
| Cross-case DCB | `scripts/compute_dcb_across_case.py` | Table 1 DCB column |
| Task-weighted accuracy | `scripts/compute_dfah_accuracy.py` | Table 1 Acc column |
| Chance-corrected agreement | `scripts/compute_kappa.py` | Table 1 κ column |
| Ground-truth baselines | `scripts/compute_gt_baselines.py` | Table 1 GT reference row |
| Tool-call counts | `scripts/compute_tool_call_counts.py` | Appendix B channel matrix |
| Task-level gap CIs | `scripts/compute_task_gap_cis.py` | Table 5 (B=10,000, seed=42) |
| Model-level bootstrap CIs | `scripts/compute_bootstrap_cis.py` | Appendix CI table (B=10,000, seed=42) |
| Metric–accuracy correlations | `scripts/compute_accuracy_metric_correlations.py` | §5.2 orthogonality stats |
| N=3 subsampling robustness | `scripts/n3_subsampling_sensitivity.py` | §5 robustness (all C(8,3)=56 subsets, 568 case groups, Spearman ρ=1.0) |

All stochastic stages use `numpy.random.default_rng(seed=42)` and are
byte-reproducible given the pinned numpy version. `PYTHONHASHSEED=0` is set
by the harness for belt-and-braces determinism (no result depends on hash
ordering; all iteration is over sorted keys).

## Verification semantics

1. Reference CSVs under `results/` are snapshotted to `build/repro/reference/`.
2. The full pipeline regenerates every CSV from raw logs.
3. Regenerated CSVs are diffed against the reference (exact for counts and
   strings, |a−b| ≤ 1e-12 + 1e-9·|b| for floats).
4. The paper's headline numbers are asserted **directly against the
   regenerated CSVs** — an edited reference file cannot mask drift.
5. On failure, the reference is restored and divergent outputs are preserved
   in `build/repro/regenerated/` for inspection.

## Provenance layer

```bash
python3 -m bench.provenance.verify --help
```

Audit bundles are hash-chained (SHA-256) and signed (Ed25519 via the
`cryptography` package — the provenance layer has no other non-stdlib
dependency). Canonical JSON serialization is deterministic for the value
domain used; it does **not** claim RFC 8785 compliance.

## Known caveats (disclosed, not hidden)

1. **Anthropic temperature (corpus episodes logged before 2026-06-09).**
   The runner omitted `temperature` in Anthropic API calls, so Claude
   episodes in the corpus were sampled at the provider default (1.0) while
   their log metadata records 0.0. Ollama and Gemini runs correctly used
   temperature 0.0. The runner is fixed (explicit `temperature=0.0`,
   metadata threaded from the same constants as the request — see
   `tests/test_runner_protocol.py`). Implications for interpretation are
   discussed in the paper's limitations section: Gemini 2.5 Pro exhibits the
   same trajectory-divergence phenomenon (56.6% diverger rate) at
   temperature 0.0, so the paper's central claim does not rest on the
   Claude rows.
2. **Evidence channel for decision-divergent groups.** A legacy logging bug
   overwrote tool outputs with empty lists when a case group was re-logged
   as non-deterministic (1,499 episodes). ECD is therefore computed on the
   case groups where evidence survived — predominantly decision-stable
   groups. The logger is fixed (re-logs preserve outputs; an overwrite guard
   refuses to clobber richer full logs). The destroyed outputs are not
   recoverable, so published ECD values are unchanged and the
   channel-availability matrix (Appendix B) reflects actual coverage.
3. **Replay vs. inference-stack determinism.** API models (Claude, Gemini)
   expose no seed parameter; API-layer nondeterminism is part of the
   deployment behavior under measurement (paper §3.2).

## Test suite

```bash
make test-bench   # = python3 -m pytest tests/ -q
```

- Metric math: `test_dcb.py`, `test_ecd.py`, `test_scdr.py` (embeddings mocked — no network)
- Schema/round-trip: `test_schema.py`
- Provenance: `test_canonicalize.py`, `test_chain.py`, `test_certificate.py`, `test_verify.py`
- Stats/bootstrap determinism: `test_stats.py`
- Pipeline conventions (TAR/DAR denominators, kill criterion, aggregation weighting): `test_dfah_pipeline.py`
- Runner replay protocol (temperature, logged-metadata honesty, evidence preservation, decision parsing): `test_runner_protocol.py`
- Domain extension (§3.1, zero-metric-change): `test_domain_extension.py`

## Domain-agnostic claim (§3.1)

```bash
python3 examples/domain_extension_medical.py
```

Registers a medical-triage ontology (escalate / treat / refer) with two mock
tools and computes DAR/TAR/ECD/DCB with the unmodified `bench/` library.
