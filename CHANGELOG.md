# Changelog

All notable changes to the prospective `dfah-bench` package are recorded here.
Historical research artifacts and paper computations have their own frozen
provenance and are not versioned by this changelog.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases use semantic versioning.

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
