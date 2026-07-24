# Operating DFAH in production

Use DFAH as a prospective replay-stability control around an adapter you own.
It is provider-neutral: it does not send provider requests, manage
credentials, or retry a provider on your behalf. Your adapter must capture the
normalized request and observable tool path it actually used.

## Pin the experiment contract

Create a versioned `Suite` for each behavior contract. `suite_version` is a
SemVer identifier and the human-readable compatibility contract; the suite
also derives fixture and tool-schema hashes as exact integrity checks.
`Replay` rejects an agent whose manifest does not exactly match all four of
those values. `Report.compare()` also rejects cross-manifest and cross-
`suite_version` comparisons unless its explicit override arguments are used.
An override does not make changed case populations, incomplete runs, or
unverified artifacts suitable for a release comparison. Do not use one to
make changed fixtures look like a trend.

Top-level report DAR/TAR values are task-weighted: cases are averaged within
task, then every represented task receives equal weight. Gate thresholds use
those task-weighted aggregates, while `case_reports` retains the individual
groups for review.

Build the manifest from exact post-normalization request settings. In
particular, `temperature`, `top_p`, and `seed` must agree with the recorded
`WireRequest.parameters` when the episode completes. Store provider model,
adapter name/version, library version, and either an approved Git revision or
a reviewed implementation hash through the manifest rather than unstructured
log text. An adapter version string alone is not source provenance.

Before dispatch, DFAH writes an immutable `run-plan.json` commitment over the
manifest, selected artifact case IDs/tasks, replay count, seed, sample rate,
and shuffled schedule. Reusing the directory with a different run design
fails rather than appending a second experiment to the first.

## Capture tools through a registry

Declare `ToolSpec` values in the suite and register the corresponding
functions in a `ToolRegistry`. Bind it once with
`@agent(manifest=manifest, tools=tool_registry, suite=suite)`. DFAH creates a per-episode
`ToolSession`, puts it on `RunContext.tools`, and verifies that the adapter
returns the session's exact trajectory. An explicit `Replay(tools=...)`
override remains available for non-decorator adapters. This prevents a
separate, hand-recorded path from silently diverging from observed calls.

The registry validates each call against its declared Draft 2020-12 JSON
Schema before invoking the implementation. A missing field, wrong type, or
disallowed extra field is recorded as a rejected call and cannot be coerced
silently by Python or a provider adapter.

The session returns tool values to the adapter but, by default, persists only
canonical argument and output hashes. Raw arguments are redacted from the
record while `arguments_hash` remains available for strong agreement. Set
`capture_tool_arguments=True` only under an approved data-retention policy.
Unknown, rejected, or failed calls become `tool_error`; do not convert them to
an empty trajectory.

## Start in shadow mode

`Replay` defaults to `ReplayMode.SHADOW`: it always writes the report and, if
a policy is supplied, evaluates the policy without raising for a failed gate.
Use it first to measure data quality, review volume, provider latency, and
cost without blocking the surrounding workflow.

```python
from dfah import GatePolicy, Replay, ReplayMode, TaskGatePolicy

policy = GatePolicy(
    required_replays=3,
    min_dar=0.95,
    min_tar_seq=0.90,
    max_gap=0.05,
    max_flags_per_100_cases=5.0,
    max_cost_per_case_usd=0.02,
    min_eligible_fraction=1.0,
    min_observed_groups=20,
    require_complete=True,
    by_task=(
        TaskGatePolicy(
            task="compliance",
            min_tar_seq=0.95,
            max_unanimous_path_change_rate=0.03,
            min_observed_groups=20,
        ),
    ),
)

shadow_report = Replay(
    suite="my-suite.json",
    replays=3,
    seed=42,
    out=".dfah/runs/my-suite-shadow",
    mode=ReplayMode.SHADOW,
    gate=policy,
    capture_tool_arguments=False,
    budget_usd=10.00,
    estimated_max_episode_cost_usd=0.02,
    episode_timeout_s=90.0,
    concurrency=1,
).run(my_agent)
```

The optional per-task checks prevent a stable task from masking a weak one in
the task-weighted aggregate. Select thresholds from a reviewed shadow
baseline; the example values are illustrative, not universal control limits.

`budget_usd` is a hard admission ceiling only when
`estimated_max_episode_cost_usd` is also positive. DFAH reserves that amount
before each dispatch and records `budget_stop` episodes when the next
reservation would exceed the cap. `total_cost_usd` and `total_latency_ms` are
sums supplied/measured per episode. If a response becomes unknown after
dispatch, DFAH conservatively consumes and persists the full reservation as
that episode's cost; reconcile this upper bound with provider billing later.
Set `episode_timeout_s` to an explicit provider-aware ceiling for unattended
runs. A timeout occurs after the durable dispatch boundary, so DFAH records
`unknown_after_dispatch`, retains the conservative reservation, and never
resends that episode automatically. The timeout is part of the immutable run
design; changing it requires a new output directory.
`max_cost_per_case_usd` divides total cost by observed eligible case groups and
fails when that denominator is zero. Start serially: higher `concurrency` can
increase outstanding budget reservations and external provider load.

Promote only a reviewed policy to blocking mode:

```python
blocking_report = Replay(
    suite="my-suite.json",
    replays=3,
    seed=42,
    out=".dfah/runs/my-suite-blocking",
    mode=ReplayMode.BLOCKING,
    gate=policy,
    budget_usd=10.00,
    estimated_max_episode_cost_usd=0.02,
).run(my_agent)
```

In blocking mode a failed gate raises `GateViolationError` **after** the
report is persisted, so preserve the run directory for review. Without a
`gate` policy, blocking mode has no threshold to enforce.

## Interpret and review

Require a `complete` report for a release gate unless an explicit exception is
approved. A partial report can arise from a budget stop, parse failure,
provider/contract failure, interruption, missing required channels, or an
incomplete replay group. Do not fill missing values with empty tool paths.

| Signal | Meaning | Typical action |
| --- | --- | --- |
| High DAR, low TAR.seq / positive gap | Stable final decision with varying observed tool order/path | Inspect tool routing, retrieval, and policy controls. |
| `flags_per_100_cases` | Rate of eligible groups with unanimous decisions but strong path variation | Triage `argument_or_result`, `order_only`, `multiplicity`, and `tool_set`; none is automatically material. |
| Cost and latency totals | Values recorded per episode and summed in the report | Tune sampling, replay count, concurrency, or provider budget. |
| Ineligible/partial groups | Capture or terminal-state failure, not evidence of agreement | Repair adapter telemetry before using stability metrics. |

Use at least two replays; DFAH rejects fewer. `sample_rate` chooses a seeded,
deterministic subset of suite cases, but it does not convert that subset into a
population sample or a release-quality test suite.

Before a larger or paid run, bound the integration smoke test explicitly:

```python
from dfah.testing import check_agent

preflight = check_agent(
    my_agent,
    max_cases=2,
    budget_usd=0.20,
    estimated_max_episode_cost_usd=0.05,
    episode_timeout_s=30.0,
    raise_on_error=True,
)
```

The preflight always uses two replays per selected case, so this example can
make at most four first-pass adapter calls. It then repeats the run against
the same store to check resumability without another adapter call. A
non-applicable observation is reported as `SKIP`; two stable repetitions do
not prove that every wall-clock or ambient-state dependency is absent.

## Persistence, security, and recovery

Keep prospective stores outside the historical corpus and under access control.
The current local `FileStore` supports macOS and POSIX systems; it requires
advisory file locks and directory-descriptor operations. Windows support needs
a reviewed storage backend and is not claimed by this alpha.
The store never overwrites a committed episode. On a second invocation with
the same manifest, artifact case ID, replay index, and output location it
reads the existing committed record instead of invoking the agent.

Set `Case.artifact_case_id` to a stable pseudonym for production cases. That
pseudonym becomes the durable run-plan, episode, report, and inspection
identifier while the adapter can still receive the original `Case`. The
original ID may still reach systems called by the adapter; review their
logging and retention separately.

One output directory permits one active writer. A kernel-backed local lock
protects the live process, and a metadata lease remains after abnormal exit.
Do not delete the lock by hand. `recover_stale_lease=True` is an explicit
operator action after confirming the recorded local process is dead; DFAH
archives the stale lease before proceeding.

DFAH writes a planned record before reservation and writes a durable
dispatching intent immediately before calling the adapter. A planned-only
record is safely resumed because no external call was eligible to occur. A
dispatching record without a terminal commit becomes `interrupted` with
`unknown_after_dispatch` and is never automatically re-dispatched. Manually
creating a fresh manifest or deleting artifacts changes the experiment and
must follow the local control process.

Raw output is not persisted by `Replay`. The artifact serializer redacts a
small set of credential-shaped strings, but callers must minimize sensitive
values before placing them in tool arguments, request parameters, metadata,
or exceptions. Artifact wire-payload, argument, and result hashes are equality fingerprints,
not anonymization: low-entropy values may be recovered by enumeration.
Pseudonymize or tokenize sensitive values before hashing.

With `otel=True`, OpenTelemetry uses the GenAI semantic conventions for agent
and tool operations and records model, usage, status, parse provenance, and
non-content tool state. It excludes raw prompts, arguments, results, and their
fingerprints from span attributes.

The built-in suites are synthetic integration checks. A production suite needs
its own approved ontology, fixtures, channel requirements, registered tool
schemas, and versioning policy. It must not point at frozen paper data or
private exports.

## Put the gate in the test suite

The pytest entry point exposes two fixtures when `dfah-bench` is installed. A
release test consumes a previously generated run directory without making a
live provider call. By default the report must match the committed episode
root, counts, costs, latency, and regenerated case metrics; a standalone JSON
report is inspection-only and cannot satisfy the default gate policy.

```python
def test_no_hidden_path_variation(dfah_report):
    assert dfah_report.metrics_available
    assert dfah_report.unanimous_with_path_change_rate is not None
    assert dfah_report.unanimous_with_path_change_rate < 0.05
```

```bash
pytest --dfah-report .dfah/runs/release-candidate \
  --dfah-policy examples/dfah_gate.yaml
```

For an explicit policy assertion, use the second fixture:

```python
def test_dfah_release_gate(dfah_gate):
    dfah_gate().raise_for_failures()
```

Generate reports in a separate, credentialed shadow job; run the test job
against the sanitized artifact. CI should never contain a live-provider test.

For a flagged case, inspect the verified run without exposing captured
content:

```bash
dfah inspect .dfah/runs/release-candidate --case CASE-PSEUDONYM
```

The command shows decisions, tool-name order, and local equality identities.
It does not print prompts, tool arguments, or tool results.
