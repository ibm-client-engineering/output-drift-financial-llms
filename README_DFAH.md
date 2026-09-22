# DFAH-Bench

DFAH is a replay harness for tool-using AI agents. It asks a practical
question: when an agent reaches the same decision more than once, did it also
take the same observable path?

The package records versioned replay groups, qualifies whether their required
channels are comparable, and reports decision agreement (DAR), tool-path
agreement (TAR), the paired DAR–TAR gap, and expected review load. It measures
repeatability and observable execution fidelity. It does not establish that a
decision is correct, that latent reasoning is faithful, or that a system is
safe.

The package is alpha software. Start with synthetic cases and shadow replays.

## Install

From PyPI:

```bash
python -m venv .venv-dfah
source .venv-dfah/bin/activate
python -m pip install dfah-bench
```

Add OpenTelemetry support when you need it:

```bash
python -m pip install "dfah-bench[otel]"
```

For package development:

```bash
git clone https://github.com/ibm-client-engineering/output-drift-financial-llms
cd output-drift-financial-llms
python -m venv .venv-dfah
source .venv-dfah/bin/activate
python -m pip install -e ".[dev,otel]"
```

Python 3.10 or newer is required.

## A two-minute, no-network run

The demo agent ships inside the wheel, so these commands work outside a source
checkout and do not require an API key:

```bash
dfah check-agent \
  --agent dfah.demo:toy_agent \
  --episode-timeout-s 5

dfah run \
  --agent dfah.demo:toy_agent \
  --replays 3 \
  --episode-timeout-s 5 \
  --out .dfah/runs/toy-local-01

dfah analyze .dfah/runs/toy-local-01 \
  --report .dfah/runs/toy-local-01/report.html

dfah inspect .dfah/runs/toy-local-01 --case CASE-001
```

The stable demo should finish with `DAR=1.000`, `TARseq=1.000`, `gap=0.000`,
and `flags/100=0.0`. That is an integration check, not a claim that every agent
or task should score 1.0.

## Python API

```python
from pathlib import Path

from dfah import Replay
from dfah.demo import toy_agent, toy_suite

report = Replay(
    suite=toy_suite,
    replays=3,
    seed=42,
    out=Path(".dfah/runs/python-quickstart"),
).run(toy_agent)

if report.metrics_available:
    assert report.tar is not None
    print(report.dar, report.tar.seq, report.gap)
else:
    print("metrics unavailable", report.ineligibility_reasons)
```

`check_agent()` is the first call to use with a real integration:

```python
from dfah.testing import check_agent

conformance = check_agent(
    my_agent,
    max_cases=2,
    expected_tools={"CASE-001": ["read_risk_tier"]},
    budget_usd=0.20,
    estimated_max_episode_cost_usd=0.05,
    episode_timeout_s=30.0,
    raise_on_error=True,
)
```

`expected_tools` names, per selected artifact case ID, the declared tools whose
calls must be captured through the injected session in every replay. Without
it an observed-empty path is accepted as valid, so an adapter that invokes a
tool implementation directly stays invisible; the report's `selected_case_ids`
lists the cases the preflight ran.

An integration implements the small `Agent` protocol and returns a typed
`AgentResult` containing an observed trajectory, parse provenance, and a
sanitized echo of the request that was actually sent. The package does not
read ambient API keys, retry provider calls, or normalize requests behind the
manifest.

## What DFAH keeps explicit

- Missing or malformed required channels make a replay group ineligible.
  Unavailable aggregates render as `—` and serialize as `null`; they never
  become zero agreement or zero divergence.
- `suite_version`, fixture hashes, tool-schema hashes, request settings, and
  implementation provenance are part of the comparison contract.
- Empty observed paths remain different from missing paths.
- Strong trajectory identity can include canonical argument and result hashes
  without placing raw values in reports.
- Run plans are immutable, episode commits are append-only, and resumability
  does not resend an already committed episode.
- Artifact verification regenerates a report from its committed episode store
  and binds the two by commitment. It is tamper-evident within a run
  directory, not a signature: protect the directory, and anchor
  `run_plan_sha256`, `episode_artifact_root_sha256`, and each gate record's
  `policy_sha256` outside it when authenticity matters.
- Cost admission is conservative after dispatch, and shadow sampling reports
  both estimated cost and expected flags per 100 cases.
- The optional OpenTelemetry integration emits GenAI spans without prompts,
  arguments, or tool results. Normalized decision labels and tool identities
  remain observable metadata and should be reviewed before export.
- The pytest plugin lets an existing test suite load a verified report and
  enforce project-specific replay gates.
- Every policy evaluation, shadow or blocking, is recorded as
  `RUN/gates/<report_id>.json` with the policy and its SHA-256. The CLI prints
  `policy=PASS` or `policy=FAIL` with the failed check names for shadow runs
  and passing blocking runs, and an error naming the record for a failing
  blocking run.

## Research artifact versus package

The repository contains complementary research and package layers:

- `bench/` and the checked-in replay corpus reproduce the corrected v2
  retrospective analysis of [DFAH-Bench](https://arxiv.org/abs/2607.20491).
- `paper/arxiv_dfah_bench_v3/` contains the v3 manuscript and its figures.
  Building that paper and reproducing the new hosted studies have different
  requirements; see the [v3 study guide](docs/dfah/v3-study.md).
- `src/dfah/` is the prospective package for new integrations and new replay
  captures.

The package does not rewrite historical logs or silently mix old and new
studies. Its built-in suites validate integration plumbing; they are not
financial-accuracy benchmarks.

The 0.1.3 source adds `dfah.metrics.execution_summary`: it separates captured
returns, rejected invocations, unresolved proposals and errors. An unavailable
channel keeps unavailable counts, while an observed-empty channel has zero
counts. See the [execution evidence guide](docs/dfah/execution-evidence.md)
and the [offline Lab 10](docs/lab-10/README.md). This addition leaves the replay
agreement metrics and their eligibility rules unchanged.

## Replay, review, and retest

Supply a policy when a run should enforce thresholds:

```bash
dfah run --agent package.module:agent \
  --mode blocking --policy gate.yaml --episode-timeout-s 30
```

Blocking mode without a policy is a configuration error. A failed gate keeps
the report, records the evaluation under `RUN/gates/`, names that record in
its error, and exits unsuccessfully. In shadow mode the same record is written
and the outcome is printed without stopping the run. In pytest, an explicit
`--dfah-policy` is enforced before collection, even if no test uses a DFAH
fixture.

The [bounded replay-and-review example](docs/dfah/replay-review-loop.md)
evaluates two explicitly versioned local candidates under one fixed policy:
the first changes tool paths and fails; the corrected candidate passes. It
keeps both evidence sets so the change can be reviewed and retested.

## Export replay evidence

The package includes a local exporter for the Every Eval Ever v0.2.2
interchange schema:

```bash
dfah export .dfah/runs/MY-RUN \
  --format every-eval-ever \
  --out .dfah/exports/MY-RUN
```

The exporter writes local files only. It requires an artifact-verified run
with at least one eligible replay group, hashes arbitrary request settings,
and excludes prompts, raw tool arguments, raw results, and reasoning traces.
Model/provider/adapter identifiers, normalized decision labels, tool names,
and equality hashes remain metadata; review them before sharing an export.

The exporter works on all supported Python versions. Optional upstream
validation requires Python 3.12 or newer and `dfah-bench[eee]`; the extra pins
the upstream 0.2.3rc1 validator for schema 0.2.2. See the
[export guide](docs/dfah/every-eval-ever.md).

## Guides

- [Quickstart](https://ibm-client-engineering.github.io/output-drift-financial-llms/dfah/quickstart/)
- [Bring your own agent](https://ibm-client-engineering.github.io/output-drift-financial-llms/dfah/bring-your-own-agent/)
- [Production rollout](https://ibm-client-engineering.github.io/output-drift-financial-llms/dfah/production/)
- [Design decisions](https://ibm-client-engineering.github.io/output-drift-financial-llms/dfah/design/)
- [Maintainer release process](https://github.com/ibm-client-engineering/output-drift-financial-llms/blob/main/docs/dfah/releasing.md)
- [Research paper](https://arxiv.org/abs/2607.20491)

The recommended rollout is simple: qualify the adapter, run sampled shadow
replays, inspect the review queue, and only then decide whether a gate should
block promotion.
