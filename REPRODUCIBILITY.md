# Reproducibility — DFAH-Bench

This repository supports two local forms of reproduction and documents the
published v3 study:

| Scope | Public materials | What can be checked |
| --- | --- | --- |
| Corrected v2 research | `bench/`, the sanitized replay fixture, `scripts/`, `results/v2/` | Retrospective CSVs and checks of the aggregate-only prospective extensions |
| Prospective package | `src/dfah/`, `examples/`, `tests/dfah/` | Local capture, replay, policy and execution-evidence behavior on synthetic examples |
| Published v3 study | [arXiv:2607.20491](https://arxiv.org/abs/2607.20491) | Methods, figures, findings and stated access limits; no local manuscript build |

The historical analysis, prospective API diagnostic, local systems check,
native banking study and fixed-state probes retain separate tasks,
denominators and capture contracts. Package smoke tests exercise implementation
behavior; they are not additional study observations.

The package environment uses `pyproject.toml`. The corrected v2 research
reproduction is independent and runs from `requirements.txt`.

## Published v3 study

The [paper on arXiv](https://arxiv.org/abs/2607.20491) contains the methods,
figures and findings. The [v3 study guide](docs/dfah/v3-study.md) explains the
native outcomes, gate probes and cost accounting. Manuscript build inputs are
maintained outside this public checkout.

The hosted banking and gate-probe runners, frozen collection plans and raw
provider captures are retained separately from this public checkout.
`make reproduce-paper` continues to reproduce the preserved **v2** artifacts;
it does not execute or verify the hosted v3 collection.

The manuscript reports 1,080 terminal native episodes, with 1,033 known
outcomes and 47 unknown outcomes, plus 1,944 scheduled fixed-state queries.
Scheduled, evaluable and paired denominators remain explicit. Missing outcomes
and unavailable gate decisions leave the prespecified confirmatory families
unavailable; the reported descriptive intervals and logical missing-outcome
bounds retain their separate meanings. The hosted runtime remains open:
retained request identities and settings cannot refreeze a provider's model.

## Prospective package

```bash
python3 -m venv .venv-dfah
source .venv-dfah/bin/activate
python -m pip install -e ".[dev,otel]"

make test-dfah
dfah check-agent --agent dfah.demo:toy_agent
dfah run \
  --agent dfah.demo:toy_agent \
  --replays 3 \
  --out .dfah/runs/quickstart
dfah analyze .dfah/runs/quickstart
```

The demo is local and makes no provider calls. Perfect agreement establishes
that bounded adapter/tool/replay contract only.

### Execution evidence in the source checkout

[Lab 10](docs/lab-10/README.md) runs a synthetic example of
[`execution_summary`](docs/dfah/execution-evidence.md): completed invocations,
rejected calls, unresolved proposals, errors and unavailable versus
observed-empty trajectories. After the editable installation above, its
focused implementation checks are:

```bash
python -m pytest tests/dfah/test_execution_evidence.py tests/dfah/test_expected_tool_capture.py -q
```

These offline checks use constructed records and adapters. They do not import
the hosted banking traces or supply a generic native-task or policy-correctness
evaluator. The helper leaves the historical agreement metrics unchanged.

## Historical public pipeline

Use a separate environment:

```bash
python3 -m venv .venv-research
source .venv-research/bin/activate
python -m pip install -r requirements.txt

make test-bench
make reproduce-paper
make verify-v2-manifest
make reproduce-paper-v1  # archived lineage, when specifically needed
```

The Makefile honors `PYTHON=/path/to/python` when the research environment is
not activated in the current shell.

The default `reproduce-paper` target regenerates every corrected retrospective
CSV from the sanitized public fixture in a temporary directory, compares it
with the committed v2 artifact, and verifies the aggregate-only prospective
extensions and manifest. `reproduce-paper-v1` is preserved only for version
lineage.

## Corrected evidence lineage

The corrected primary analysis is:

```text
8,129 raw episode records
−     2 singleton episodes
= 8,127 records in the archived v1 analysis
− 2,612 portfolio episodes / 449 groups
−    14 DeepSeek episodes / 2 groups
= 5,501 retained-task episodes / 887 groups
− 1,344 episodes in two configurations with zero observed tool calls / 168 groups
= 4,157 episodes from configurations with observed tool use / 719 groups
```

The primary slice contains eight retained configurations and two synthetic
tasks (compliance triage and financial DataOps). Its case-level analysis is
task-weighted and uses three or eight replays per group, shown explicitly in
every table. Twenty-five retained episodes in nine groups have observed empty
tool sequences; the denominator therefore describes configurations with
observed tool use, not per-episode tool calls.

The portfolio fixture and its dependent aggregates are excluded. Historical
Evidence Contact Divergence is also excluded because a legacy overwrite made
that channel missing not at random, and the retained hashes represented output
integrity rather than source contacts.

## Eligibility semantics

A replay group contributes to agreement only when:

- its suite, model, provider, prompt, tools, decoding settings, and required
  contract fields are comparable;
- every required channel is present and valid; and
- the required replay count is met.

An observed empty tool path is a valid path. A missing or malformed channel
makes the group ineligible and is never scored as agreement or zero
divergence.

## Corrected primary measures

| Measure | Definition |
|---|---|
| DAR | modal decision share within an eligible replay group |
| TARseq | modal exact ordered tool-name path share on the same denominator |
| Gap | paired per-case DAR − TARseq, aggregated with equal task weight |

Historical multiset and set projections are retained as sensitivity analyses.
Historical argument and result channels were not captured and cannot be
reconstructed.

## Prospective extensions

### API diagnostic

- 600 terminal episodes;
- 570 eligible episodes across 190 exact three-replay groups;
- 288 eligible Terra episodes / 96 groups;
- 282 eligible Sonnet 5 episodes / 94 groups.

One Sonnet/DataOps stratum retained 44 groups against a predeclared minimum of
45, so the global publication gate did not pass. The aggregate remains a
diagnostic extension, not a provider ranking.

The public component projection reports decision agreement, ordered tool-name
agreement, name-plus-canonical-argument agreement, and result-only agreement.
The API captures needed to recompute those aggregates remain approval-gated;
the public target verifies their safety-projected files by hash, schema,
denominator, gate, and published values.

### Local systems check

- 800 completed episodes;
- 792 eligible episodes across 99 eight-replay groups;
- Gemma 4 E4B: 400/400 eligible;
- Qwen 3.5: 392/400 eligible after eight parse failures made one group
  ineligible.

All eligible local groups repeated the fixed required path. This validates
capture and replay mechanics in that synthetic harness, not general model
determinism or financial accuracy.

## Provenance

The research API contains deterministic JSON canonicalization, SHA-256 hash
chains, and Ed25519 certificate utilities. The prospective package records
manifests, versioned suites, episode keys, parse provenance, and resumable
commits.

These capabilities do not imply that raw provider logs or signed provider
bundles are publicly releasable. Public releases should contain only approved,
sanitized artifacts.

## Test targets

```bash
make test-bench  # frozen research tests, excludes tests/dfah
make test-dfah   # prospective package tests
make test-all    # both layers
```

All tests are offline unless an explicitly opt-in integration environment is
configured.

## Version 2 correction note

Version 2:

- corrects `claude-opus-4-20250514` from “Claude Opus 4.5” to **Claude Opus
  4**;
- excludes the inconsistent portfolio fixture and its dependent results;
- removes historical evidence-contact analysis affected by nonrandom
  missingness;
- formalizes fail-closed replay eligibility; and
- adds separate prospective argument/result-aware evaluations.

The central conclusion remains unchanged: stable decisions can conceal
unstable observable tool paths.
