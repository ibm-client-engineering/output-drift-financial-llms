# Bring your own agent

DFAH wraps one execution; it does not replace the agent or its provider
client. An integration implements the small `Agent` protocol (or uses
`@dfah.agent`) and returns a typed `AgentResult`.

The adapter owns five things:

1. construct the final provider request;
2. make exactly one episode-level dispatch;
3. capture the observable tool path;
4. return the provider's output, usage, and cost; and
5. attest the settings inside the final wire payload.

API keys stay in the application. They must never be placed in a `Manifest`,
`Case`, `WireRequest`, tool argument, error message, or report.

## Bind source and wire truth

Use a reviewed Git revision for a clean repository, or hash the exact adapter
artifact that will execute:

```python
from hashlib import sha256
from pathlib import Path

from dfah import WireRequest, build_manifest

adapter_path = Path("my_project/provider_adapter.py")
implementation_hash = sha256(adapter_path.read_bytes()).hexdigest()
parameters = {"temperature": 0.0, "top_p": 1.0, "seed": 42}

manifest = build_manifest(
    suite,
    provider="my-provider",
    model="pinned-model-identifier",
    endpoint="https://provider.example/v1/responses",
    adapter="my_project.provider_adapter",
    adapter_version="1.3.0",
    implementation_hash=implementation_hash,
    request_parameters=parameters,
)
```

Build the `WireRequest` only after the provider payload has reached its final
shape. Top-level settings need no mapping:

```python
wire = WireRequest.from_payload(
    provider="my-provider",
    model="pinned-model-identifier",
    endpoint="https://provider.example/v1/responses",
    adapter="my_project.provider_adapter",
    adapter_version="1.3.0",
    payload=payload_sent_to_provider,
    parameters=parameters,
)
```

If a provider nests settings, declare their exact paths:

```python
wire = WireRequest.from_payload(
    provider="my-provider",
    model="pinned-model-identifier",
    endpoint="https://provider.example/v1/responses",
    adapter="my_project.provider_adapter",
    adapter_version="1.3.0",
    payload={"generation": {"temperature": 0.0, "top_p": 1.0, "seed": 42}},
    parameters=parameters,
    parameter_paths={
        "temperature": ("generation", "temperature"),
        "top_p": ("generation", "top_p"),
        "seed": ("generation", "seed"),
    },
)
```

A declared setting that is absent from—or differs from—the hashed payload is a
contract error. An adapter version string alone is not source provenance.

## Capture tools

Prefer a `ToolRegistry` bound to the agent. DFAH validates arguments against
the suite's JSON Schemas, injects a per-episode `ToolSession` as
`context.tools`, and records results before returning them to the adapter.
Return `context.tools.trajectory()` in `AgentResult`.

Manual trajectory capture is available when a framework owns tool execution,
but every call must still use a declared tool, schema-valid arguments, an
observed result state, and an output hash. If the framework cannot expose
those channels, record the episode as ineligible; do not manufacture an empty
path.

## Framework and router integrations

Pydantic AI, LangGraph, LiteLLM, provider SDKs, and custom loops can all sit
behind the same protocol. The boundary is evidentiary, not brand-specific:

- the adapter must see the final normalized provider payload;
- the manifest and wire echo must identify the endpoint and pinned model;
- hidden provider retries must not masquerade as one replay;
- parse failures and tool errors are terminal episode outcomes, not retry
  triggers; and
- framework traces do not replace the typed `Trajectory` contract.

If a router cannot expose what was actually sent, use a lower-level adapter
for a manifest-authoritative run. Convenience integrations can be added as
optional packages later without weakening the core contract.

Run `check_agent()` with explicit call and cost bounds before a larger replay,
then start with sampled shadow mode as described in `production.md`.

If a run reports `metrics_available == False`, do not tune thresholds around
the missing result. Read `ineligibility_reasons`: each entry counts replay
groups containing one normalized, content-free reason. For example,
`wire_payload_mismatch` means the final hashed payload changed across repeats
of the same case; terminal-status and missing-channel codes identify capture
or execution failures. The CLI prints the same affected-group counts without
prompts, payloads, arguments, results, or provider exception text.
