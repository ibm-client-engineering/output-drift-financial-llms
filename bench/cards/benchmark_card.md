# DFAH-Bench Benchmark Card

## What DFAH-Bench measures

DFAH-Bench measures repeatability in the observable execution of a
tool-using agent. Under a declared replay contract, it compares:

- the final decision (DAR);
- the ordered tool-name path (TARseq); and
- when captured prospectively, ordered tool names, canonical arguments, and
  deterministic result identities (TARstrong).

“Faithfulness” means fidelity to the observable replay record. The benchmark
does not inspect hidden reasoning and does not establish correctness,
truthfulness, safety, materiality, or regulatory compliance.

## Corrected primary evidence

The authoritative historical slice contains:

- **4,157 episodes from configurations with observed tool use**;
- **719 comparable replay groups**;
- **eight retained configurations**;
- **two synthetic tasks**: compliance triage and financial DataOps; and
- three or eight replays per group, reported explicitly by row.

Among 627 unanimous-decision groups, 122 (19.5%) changed tool sequence. Of
those, 17 changed order only, 58 changed call multiplicity while retaining the
same tool-name set, and 47 changed the tool-name set.

The raw ledger contains 8,129 records. The former 8,127-record analysis is
retained only as lineage; it included an inconsistent portfolio fixture and a
historical evidence-contact channel affected by nonrandom missingness.

## Separate prospective checks

- **API diagnostic**: 600 terminal episodes, 570 eligible episodes, 190
  groups. Decision agreement was 94.2–95.1%, ordered name-path agreement was
  66.9–69.4%, name-plus-canonical-argument agreement was 45.0–51.5%, and
  result-only agreement was 54.3–56.9%. One predeclared coverage gate missed
  by one group, so this is diagnostic rather than a provider ranking.
- **Local systems check**: 800 completed episodes, 792 eligible episodes, 99
  groups. The fixed required four-tool path repeated exactly in eligible
  groups. This validates the bounded harness and capture path, not general
  model determinism or accuracy.

These studies use different tasks, replay counts, and contracts. Do not combine
them into a head-to-head leaderboard.

## Replay eligibility

A group is eligible only when:

1. each replay satisfies the declared configuration-equivalence contract;
2. every required channel is present and valid; and
3. the group has the required replay count.

An observed empty tool path is data. A missing or malformed required channel
makes the group ineligible and is never scored as agreement or zero
divergence.

## Intended use

- diagnosing decision/path disagreement in synthetic or shadow replay;
- estimating review load before a blocking gate;
- regression testing a pinned agent configuration;
- validating adapter, capture, manifest, and resumability behavior; and
- supporting, but not replacing, broader model-risk evidence.

## Out-of-scope use

- model capability or accuracy ranking;
- certification of compliance, safety, or deployment readiness;
- claims about hidden reasoning;
- extrapolation from one synthetic suite to production behavior; and
- comparison across suite versions or incompatible replay contracts.

## Decision ontologies

| Task | Labels |
|---|---|
| Compliance triage | escalate, dismiss, investigate |
| Financial DataOps | auto_fix, escalate, quarantine |

The portfolio fixture remains in repository history but is excluded from the
corrected empirical analysis.

## Implementations

- `bench/` contains the frozen historical research API and lineage pipeline.
- `src/dfah/` contains the prospective, pip-installable package with
  fail-closed eligibility, versioned suites, resumability, review-load
  reporting, OpenTelemetry support, and a pytest plugin.

Start with `dfah check-agent --agent dfah.demo:toy_agent`, then use a synthetic
suite that matches the tool schema and risk boundary of the intended
integration.

## Known limitations

- synthetic English-language tasks;
- no expert-adjudicated accuracy or materiality claim;
- historical paths are tool-name only;
- historical evidence contacts are not recoverable and are omitted;
- two historical rows are limited contiguous prefixes;
- API providers do not expose identical seed/decoding controls; and
- prospective API aggregates missed one predeclared coverage gate.

## Release note

The corrected v2 analysis changes the historical model display name from
Claude Opus 4.5 to **Claude Opus 4** for exact ID
`claude-opus-4-20250514`, excludes the inconsistent portfolio fixture,
withdraws historical evidence-contact results, formalizes fail-closed replay
eligibility, and adds separate prospective argument/result-aware checks. The
central conclusion remains unchanged: stable decisions can conceal unstable
observable tool paths.
