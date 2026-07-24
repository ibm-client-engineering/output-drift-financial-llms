# DFAH quickstart

DFAH measures whether repeated executions of the same case reach the same
decision and observable tool path. It does not measure correctness. The
no-network example uses a deterministic toy agent and a registered local
tool. Install the published package; no repository clone or API key is needed.

```bash
python -m pip install dfah-bench
```

Then run:

```bash
dfah check-agent --agent dfah.demo:toy_agent --episode-timeout-s 5
dfah run \
  --agent dfah.demo:toy_agent \
  --replays 3 \
  --episode-timeout-s 5 \
  --out .dfah/runs/quickstart
dfah analyze .dfah/runs/quickstart \
  --report .dfah/runs/quickstart/report.html
```

The installed demo builds the public contract: a versioned `Suite`, a
`Manifest`, a `ToolRegistry`, an `@agent` adapter, `AgentResult`,
`WireRequest`, and `Replay`. Its implementation is also available as
`examples/dfah_quickstart.py`.

Before a replay begins, DFAH also writes an immutable run plan covering the
manifest, selected cases/tasks, replay count, seed, sample rate, and shuffled
schedule hash. Reusing the directory with a different design fails before an
agent call.

## The adapter and tool contract

An adapter exposes a manifest and returns `AgentResult` from one async call
per case. The manifest must match the selected suite's ID, version, fixture
hash, and tool-schema hash. `Replay` also checks that provider, model, adapter,
temperature, top-p, and seed in the recorded `WireRequest` match the manifest.

Register each tool before running. Its specs must exactly match
`Suite.tools` after stable name ordering; otherwise
`Replay(..., tools=registry)` fails closed. DFAH
creates one `ToolSession` per episode and injects it as `context.tools`.
Call the session, then return `context.tools.trajectory()`; a hand-built
trajectory that differs from the session record is a contract error.

```python
import hashlib
from pathlib import Path

from dfah import (
    AgentResult, Case, Replay, Suite, ToolRegistry, ToolSpec, Usage,
    WireRequest, agent, build_manifest,
)

risk_tool = ToolSpec(
    name="read_risk_tier",
    input_schema={
        "type": "object",
        "properties": {"risk_tier": {"type": "string"}},
        "required": ["risk_tier"],
        "additionalProperties": False,
    },
)
suite = Suite(
    suite_id="toy-risk",
    suite_version="1.0.0",
    decisions=("escalate", "dismiss"),
    cases=(
        Case(
            case_id="RISK-001",
            artifact_case_id="CASE-001",
            input={"risk_tier": "high"},
        ),
        Case(
            case_id="RISK-002",
            artifact_case_id="CASE-002",
            input={"risk_tier": "low"},
        ),
    ),
    tools=(risk_tool,),
)
tools = ToolRegistry()

@tools.tool(risk_tool)
def read_risk_tier(*, risk_tier: str) -> str:
    return risk_tier

parameters = {"temperature": 0.0, "top_p": 1.0, "seed": 42}
implementation_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
manifest = build_manifest(
    suite,
    provider="my-provider",
    model="my-model",
    adapter="my_project.adapter",
    adapter_version="1.0.0",
    implementation_hash=implementation_hash,
    request_parameters=parameters,
)

@agent(manifest=manifest, tools=tools, suite=suite)
async def my_agent(case, context):
    assert context.tools is not None
    risk = await context.tools.call("read_risk_tier", risk_tier=case.input["risk_tier"])
    label = "ESCALATE" if risk == "high" else "DISMISS"
    payload = {"model": "my-model", **parameters, "case_id": case.case_id}
    return AgentResult(
        output_text=f"DECISION: {label}",
        trajectory=context.tools.trajectory(),
        wire_request=WireRequest.from_payload(
            provider="my-provider",
            model="my-model",
            payload=payload,
            parameters=parameters,
            adapter="my_project.adapter",
            adapter_version="1.0.0",
        ),
        usage=Usage(input_tokens=0, output_tokens=0),
        cost_usd=0.0,
    )

report = Replay(suite=suite, replays=3, out=".dfah/runs/toy-risk").run(my_agent)
```

A session returns raw tool values to the adapter but records only canonical
argument and output hashes by default. Its recorded `ToolCall.arguments` is
empty and `arguments_redacted=True`; `arguments_hash` remains available for
strong path agreement. Pass `capture_tool_arguments=True` to `Replay` only
when approved artifact policy allows retaining arguments. A requested,
rejected, or failed tool makes the episode a `tool_error`. Tool arguments are
validated against the declared Draft 2020-12 JSON Schema before the function
can execute; schema errors are recorded without retaining the rejected value.

The decision parser accepts exactly one line `DECISION: LABEL`, where `LABEL`
is in the suite ontology. It does not guess, use a modal fallback, or accept a
bare label. A malformed output becomes `parse_failure` and is ineligible.

A zero-tool session is `ChannelState.OBSERVED_EMPTY`, not `UNAVAILABLE`.
Use a registry whenever DFAH should be the source of trajectory capture; a
manually supplied trajectory remains useful only for integrations that do not
use injected tools. Manual calls are still checked against the suite's tool
names and argument schemas, then redacted before persistence.

## Read a report

Case-level `DAR` is modal decision agreement. `TAR.seq` is modal exact ordered
tool-path agreement. Report-level DAR/TAR first average cases within each task,
then weight represented tasks equally; gates use those task-weighted values.
`gap` is `DAR - TAR.seq`, using the same retained replay groups.
`flags_per_100_cases` is 100 times the fraction of eligible case groups where
the decision is unanimous (`DAR == 1.0`) but the strong path identity changes.
That catches tool-set, multiplicity, order, argument, result-state, and hashed
result changes. `sequence_flags_per_100_cases` exposes the narrower tool-name
sequence trigger used in the historical study, and each `CaseReport` provides
`path_variation_kind` for queue triage. These are review-load signals, not a
percentage of all input cases or a correctness score.

If no replay group is eligible, aggregate metrics are unavailable:
`Report.dar`, `Report.tar`, `Report.gap`, and the flag-rate properties are
`None`; JSON contains `null`; and the CLI/HTML/Markdown render `—`. This is
different from a measured zero gap. Inspect `ineligible_groups` and the
privacy-safe `ineligibility_reasons` affected-group counts before changing an
adapter or suite. Gates continue to fail closed when a requested metric has no
eligible denominator.

## Privacy and persistence

Use a new output directory such as `.dfah/runs/...`; never target historical
`results/`, `traces/`, or paper-artifact paths. The file store is
append-only and writes mode-700 directories, mode-600 artifacts, a SHA-256
sidecar, and a commit marker. Episode records contain the parsed decision,
trajectory, request metadata, usage, cost, latency, and sanitized errors—not
the full `AgentResult.output_text`.

Reports carry a canonical commitment to the verified episode set. Loading a
run directory rechecks its immutable run plan, start/dispatch boundaries,
episode metadata, wire echoes, ontology, commitment, case metrics, cost, and
latency before a gate can pass. A detached report JSON may be opened only with
`allow_unverified=True` and is never release-gate evidence by default.

Serialization redacts several common credential shapes. That is a safety net,
not a data-classification system: do not put prompts, customer data, raw tool
results, or secrets into manifest parameters, tool arguments, metadata, or
error text unless an approved artifact policy permits it. The current
`capture_raw_content` manifest field is recorded as provenance; `Replay` does
not make it capture raw output.

## Resuming safely

Episode identity is `(manifest_hash, artifact_case_id, replay_index)`, where
`artifact_case_id` is the case's explicit pseudonym or, if omitted, its raw
`case_id`. Set an artifact pseudonym before using production cases. Re-running
the same configuration reuses committed episodes byte-for-byte and does not
call the agent again.

One output directory also permits only one writer process. A second runner
fails before dispatch. A lease left by a dead local process is not displaced
unless the caller explicitly sets `recover_stale_lease=True` after confirming
that no runner remains active; the stale lease is archived for review.

A durable planned-only record proves no external dispatch boundary was reached,
so DFAH safely resumes it. Immediately before invoking the adapter, DFAH writes
a durable dispatching intent. A dispatching record without a terminal commit is
converted to an `interrupted` episode with `unknown_after_dispatch` and is not
reissued, because the provider may already have received the request. Change
the suite version, fixture/tool contract, manifest settings, or output location
deliberately when a new experiment is intended.

Before a paid run, use:

```python
from dfah.testing import check_agent

check_agent(
    my_agent,
    max_cases=2,  # at most four agent calls: two cases x two replays
    budget_usd=0.20,
    estimated_max_episode_cost_usd=0.05,
    episode_timeout_s=30.0,
    raise_on_error=True,
)
```

Pass `out=".dfah/conformance/my-agent"` when a failed preflight needs durable
local diagnostics. Without `out`, DFAH returns terminal status/error-kind
counts in the report and removes the temporary episode store.

The conformance report records the selected case count, maximum planned calls,
and estimated cost ceiling. Its `replay_stability_smoke` reports
replay-visible decision/path variation as an observational warning rather than
an adapter-contract failure; the deterministic-tool check is `SKIP`, not
`PASS`, when the agent makes no tool calls. Passing two repetitions does not
prove the absence of all wall-clock or ambient-state dependencies.
