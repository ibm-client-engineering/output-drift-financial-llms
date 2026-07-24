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

From this repository:

```bash
git clone https://github.com/ibm-client-engineering/output-drift-financial-llms
cd output-drift-financial-llms
python -m venv .venv-dfah
source .venv-dfah/bin/activate
python -m pip install -e ".[otel]"
```

The package-index command will be:

```bash
python -m pip install "dfah-bench[otel]"
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
    budget_usd=0.20,
    estimated_max_episode_cost_usd=0.05,
    episode_timeout_s=30.0,
    raise_on_error=True,
)
```

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
- Cost admission is conservative after dispatch, and shadow sampling reports
  both estimated cost and expected flags per 100 cases.
- The optional OpenTelemetry integration emits GenAI spans without prompts,
  arguments, or tool results.
- The pytest plugin lets an existing test suite load a verified report and
  enforce project-specific replay gates.

## Research artifact versus package

The repository contains two complementary layers:

- `bench/` and the checked-in replay corpus reproduce the published
  [DFAH-Bench preprint](https://arxiv.org/abs/2607.20491).
- `src/dfah/` is the prospective package for new integrations and new replay
  captures.

The package does not rewrite historical logs or silently mix old and new
studies. Its built-in suites validate integration plumbing; they are not
financial-accuracy benchmarks.

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
