# Reproducibility — DFAH-Bench

This repository contains two related but separate reproducibility surfaces:

1. the frozen historical research pipeline under `bench/`, `scripts/`, and
   `results/`; and
2. the prospective, pip-installable package under `src/dfah/`.

Do not treat the package smoke test, historical analysis, prospective API
diagnostic, and local systems check as a single model comparison. Their tasks,
replay counts, and capture contracts differ.

Package commands and package-document links below require the prospective
package surface (`pyproject.toml`, `src/dfah/`, and `docs/dfah/`) from the
first-stage package change. The corrected v2 research reproduction is
independent of that package surface and runs from `requirements.txt`.

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
