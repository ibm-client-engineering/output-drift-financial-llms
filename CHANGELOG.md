# Changelog

All notable changes to the prospective `dfah-bench` package are recorded here.
Historical research artifacts and paper computations have their own frozen
provenance and are not versioned by this changelog.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases use semantic versioning.

## [0.1.3] - Unreleased

Hardening release candidate. No artifact schema change: run plans, episode
stores, and reports written by 0.1.2 load and verify unchanged under 0.1.3,
and run directories written by 0.1.3, which add a `gates/` directory, load and
verify under 0.1.2 because neither the store nor the report loader reads
`gates/`. Both directions were checked against runs produced by the
published 0.1.2 wheel.

### Added

- Record every `Replay` policy evaluation as `RUN/gates/<report_id>.json`
  (`GateRecord`: mode, policy, policy SHA-256, and every check) in shadow and
  blocking mode, and expose it as `Replay.last_gate_result`.
- Add `check_agent(expected_tools=...)` and `dfah check-agent --expect-tools
  CASE_ID=tool[,tool]`. The `expected_tool_capture` check fails when an
  expected call was not captured through the injected session in every
  conformance replay, which is how an adapter that invokes a tool
  implementation directly becomes visible. Observed-empty paths remain valid
  when no expectation is declared; the report lists `selected_case_ids`.

### Fixed

- Print the policy outcome (`policy=PASS|FAIL` and failed check names) from
  `dfah run --policy` for shadow runs and passing blocking runs; previously a
  shadow evaluation was discarded. A failing blocking run prints an error
  naming the failed checks and the gate record instead.
- Name the persisted gate record in the blocking-mode `GateViolationError`.
- Report a run directory that has no persisted report clearly, naming the
  missing `RUN/reports`. Version 0.1.2 already rejected such a directory, but
  by attempting to parse the run plan as a report and failing schema
  validation; the loader no longer falls back to other JSON files in the run
  root at all.

### Changed

- State in the README and production guide that artifact verification is a
  consistency check bound to the run directory, not a signature, and how to
  anchor the run-plan, episode-root, and policy commitments externally.
- Document that tool return values must be JSON-serializable and that the
  `gates/` directory sits beside the store.

## [0.1.2] - 2026-09-04

### Added

- Add a local exporter for the Every Eval Ever v0.2.2
  interchange schema. It preserves DFAH suite, replay, eligibility, and
  artifact commitments without uploading results.
- Add `dfah run --policy` for explicit shadow and blocking policy checks.
- Add a bounded, no-network replay-and-review example: a candidate with stable
  decisions and varying tool paths fails, then a corrected version passes the
  same policy with separate manifests and preserved evidence.

### Fixed

- Reject blocking mode without an explicit policy before importing or calling
  an agent. Library callers receive the same configuration check.
- Enforce an explicitly supplied pytest `--dfah-policy` before collection,
  including sessions with no DFAH fixture users. Failed gates exit 1; invalid
  policy/report configuration exits 4.
- Align EEE episode scores with the report's eligible replay groups and
  distinguish pooled episode/group rates from task-weighted agreement metrics.
- Bind exported episode records to the verified report's exact artifact root;
  a concurrent resume cannot mix a newer episode set with an older aggregate.
- Pin the available upstream EEE validator, `0.2.3rc1`, separately from schema
  `0.2.2`, and require its runtime validation in the Python 3.12 CI job.

### Changed

- PyPI publication now requires the full Python 3.10–3.13 package validation
  workflow on the tagged commit and publishes its verified distributions
  without rebuilding them.

### Security

- Reject aggregate interchange exports when DFAH metrics are unavailable.
- Hash arbitrary request-parameter maps instead of copying their values into
  exported metadata, while retaining explicit standard decoding controls.
- Document the deployment-identification risk of model, provider, adapter,
  decision-label, tool-name, and equality-hash metadata.

## [0.1.1] - 2026-07-24

### Fixed

- Point first-time users to the published `dfah-bench` package instead of an
  editable source checkout.
- Add live PyPI and supported-Python badges, a direct package link, and a
  no-clone, no-API-key quickstart.

## [0.1.0] - 2026-07-23

### Added

- Typed, provider-neutral replay API with versioned suites and manifests.
- Fail-closed decision parsing and required-channel eligibility.
- DAR, trajectory agreement, paired gap, replay subsampling, permutation,
  leave-one-case-out, and adversarial parser-fallback sensitivity utilities.
- Privacy-safe argument-aware tool recorder and optional GenAI telemetry.
- Append-only episode store with pre-/post-dispatch recovery boundaries.
- Shadow sampling, cost admission, review-load metrics, declarative gates,
  one-line agent conformance, a CLI, and a pytest plugin.
- Draft 2020-12 tool-argument validation before execution, durable conservative
  reservation accounting for unknown post-dispatch outcomes, and sanitized
  OpenTelemetry error spans.
- Strict report invariants plus an episode-artifact commitment that is
  regenerated before default CLI/pytest gates may pass.
- Immutable run-plan commitments, task-specific gate policies, review-load
  breakdowns, privacy-safe case inspection, and optional artifact case
  pseudonyms.
- Wire-payload parameter attestation (including nested provider fields),
  cryptographic adapter implementation provenance, and complete/verified
  population checks before report comparison.
- Persistent inode-bound writer guards with explicit stale-lease recovery and
  a two-process recovery-race regression test.
- Design-bound per-episode timeouts that preserve conservative cost, record
  `unknown_after_dispatch`, and never resend the ambiguous episode.
- A distribution allowlist/build guard, wheel-and-sdist CI smoke checks, and
  selective design notes covering the external patterns adopted and rejected.

### Changed

- Reports no longer substitute numeric zeros when no replay group is eligible:
  unavailable DAR, TAR, gap, and flag rates serialize as `null` and render as
  `—`; reports retain privacy-safe affected-group eligibility reason counts.
- Default run directories are derived from the manifest and replay design,
  the CLI infers an agent-bound suite, and expected CLI errors are concise
  without tracebacks or local-variable dumps.
- Conformance checks treat decision/path variation as an observational warning,
  keep contract and deterministic-tool violations as failures, include terminal
  status/error-kind counts, and can retain an explicit diagnostic directory.
- Executed tool calls are eligible only when their result channel was observed
  and committed with an output hash; missing results are never treated as
  agreement.
- OpenTelemetry tool spans export standard operation/tool identity and
  non-content state, but no argument or result fingerprints.
- `Report.compare()` defaults to complete, artifact-verified reports with the
  same selected case population and replay denominator. Every relaxation is
  explicit.
