# DFAH design: a small replay control, not another agent framework

DFAH has one job: determine whether repeated executions are comparable,
sufficiently observed, and stable in both decision and observable tool path.
It is designed to sit around an agent that already exists. The package does
not need to own prompting, orchestration, provider selection, or production
tool implementations to make that measurement useful.

The public surface stays deliberately small:

1. define a semantically versioned `Suite`;
2. expose an `Agent` with a pinned `Manifest`;
3. run a bounded `check_agent()` preflight, then a `Replay`;
4. inspect the typed `Report`; and
5. enforce a `GatePolicy` directly or through pytest.

## Patterns adopted selectively

These projects informed the shape of DFAH. The mapping is conceptual; DFAH
does not claim API compatibility with them.

| Reference | Pattern adopted in DFAH | Boundary |
| --- | --- | --- |
| [Inspect AI](https://inspect.aisi.org.uk/) | Treat structured execution records as first-class evaluation artifacts, with a clear separation between the case contract, the agent execution, and analysis. | DFAH ships typed episode records, a verified report, static rendering, and case inspection—not a general evaluation catalog, model layer, scorer framework, or transcript viewer. |
| [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) | Make the evaluation contract visibly versioned instead of relying on a content hash alone. | `suite_version` is the human-readable compatibility contract; fixture and tool-schema hashes remain the exact integrity checks. |
| [scikit-learn `check_estimator`](https://scikit-learn.org/stable/modules/generated/sklearn.utils.estimator_checks.check_estimator.html) | Give integrators a one-call conformance check before a larger run. | `dfah.testing.check_agent()` is a bounded smoke test. It reports pass/fail/skip outcomes; it is not a proof that an adapter is free from every ambient dependency. |
| [Pandera](https://pandera.readthedocs.io/) | Prefer declarative constraints and typed validation results that fit an existing Python workflow. | `GatePolicy`, `GateResult`, and the pytest fixtures provide this pattern without adding Pandera or taking over the user's data pipeline. |
| [LangGraph persistence](https://docs.langchain.com/oss/python/langgraph/persistence) | Give resumable work an explicit, stable checkpoint identity and distinguish safe reuse from a possibly dispatched operation. | DFAH keys an episode by manifest, artifact case ID, and replay index, and records planned and dispatching boundaries. It is not a graph runtime or general state store. |
| [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) | Emit agent and tool operations into an observability system teams may already run. | The optional `[otel]` integration uses `gen_ai.*` attributes where the conventions define them and adds a small set of `dfah.*` status/provenance fields. It emits no prompts, tool arguments, tool results, or their fingerprints. |

The common idea is workflow fit: the replay control should produce durable
evidence, but it should not require a team to replace its agent framework,
provider client, CI system, or telemetry backend.

## Contract identity and comparability

No single identifier answers every comparability question. DFAH keeps the
human contract, exact content, execution configuration, and run design
separate:

| Identity | What it commits to |
| --- | --- |
| `suite_id` + `suite_version` | The named behavioral contract and its SemVer revision. |
| `fixture_hash` | Exact suite cases, ontology, channel requirements, and suite semantics. |
| `tool_schema_hash` | Exact declared tool interface. |
| `manifest_hash` | Provider/model settings, adapter/source identity, suite hashes, and capture contract. |
| `run-plan.json` hash | Selected artifact case IDs/tasks, replay count, seed, sample rate, and schedule. |
| episode key | `(manifest_hash, artifact_case_id, replay_index)`, the unit of idempotent reuse. |
| episode-root commitment | The exact committed episode set from which the report was regenerated. |

Changing a suite's meaning requires a new `suite_version`; silently editing
fixtures while retaining a version is still detected by the fixture hash.
`Report.compare()` rejects incompatible contracts by default. Its explicit
overrides are for an intentional analysis, not a way to turn changed cases,
partial runs, or unverified artifacts into release evidence.

The immutable run plan closes a different gap. A manifest alone does not say
which cases were selected or in what replay design. The plan is written before
dispatch and binds those choices to the artifact directory. Reopening the same
directory with a different population, seed, sample rate, or replay count
fails instead of silently mixing experiments.

## Artifact-first resumability

DFAH persists four operational boundaries: planned episode, dispatch intent,
terminal episode, and commit marker. A planned-only episode is safe to resume
because it did not cross the durable dispatch boundary. A dispatching episode
without a terminal commit is classified as unknown after dispatch and is not
automatically called again.

Each run directory also has a single-writer lease. A kernel lock protects live
local writers; a metadata lease remains after an abnormal process exit.
Recovery is explicit, limited to a confirmed-dead local process, and archives
the stale lease for review. This favors duplicate-call prevention over
unattended retry.

This persistence model is intentionally smaller than a workflow checkpointer.
It does not save arbitrary agent state, resume inside a model turn, or
coordinate distributed workers.

## Integration and test ergonomics

`check_agent()` exercises the integration using two replays per selected case,
then reruns against the same store to verify idempotent reuse. By default it
selects at most two cases, so the maximum first-pass adapter-call count is
four. Paid adapters should always set both `budget_usd` and
`estimated_max_episode_cost_usd`.

The checks cover:

- suite/manifest compatibility;
- parse provenance on every terminal episode;
- the post-normalization wire echo;
- deterministic observed tool results for identical inputs, when tools run;
- replay-visible decision and strong-path stability; and
- resumability without a second adapter call.

A check that cannot be observed is `SKIP`, not a reassuring `PASS`. For
example, an agent that makes no tool calls cannot demonstrate deterministic
tool output in that conformance run.

The pytest plugin makes the same design usable in release workflows:

```python
def test_no_hidden_path_variation(dfah_report):
    assert dfah_report.metrics_available
    assert dfah_report.unanimous_with_path_change_rate is not None
    assert dfah_report.unanimous_with_path_change_rate < 0.05
```

The `dfah_gate` fixture evaluates a typed policy, including optional
task-specific thresholds. Both fixtures consume an existing verified run; CI
does not need provider credentials or a remembered live-evaluation command.

For a flagged case, `dfah inspect RUN --case ARTIFACT_CASE_ID` prints replay
decisions, tool-name sequences, and local equality identities. It does not
print prompts, raw arguments, or raw results.

## Privacy boundary

Content minimization happens before persistence. Production suites should set
`Case.artifact_case_id` to a stable pseudonym. DFAH then uses that value in
the run plan, episode keys, reports, and inspection commands while the adapter
can still receive the original `Case`. The adapter and provider may have their
own logging behavior; DFAH cannot make those external systems private.

SHA-256 values in local artifacts support equality and integrity checks. They
are **not anonymization**: a low-entropy account number, status value, or
small-domain argument may be recovered by enumeration. Tokenize or
pseudonymize sensitive values before they reach the artifact boundary, and
apply access controls and retention policy to the run directory.

The OpenTelemetry integration does not export local argument/result
fingerprints. It emits operation, model, usage, terminal status, decision
parse provenance, and non-content tool state only.

## Deliberate non-goals

The following are outside the core package:

- **No provider SDK or model router.** The user-owned adapter sends the
  request and returns a sanitized post-normalization wire echo.
- **No LiteLLM normalization in core.** A team may wrap LiteLLM in its
  adapter, but that adapter must attest to the final request it sent. DFAH
  does not treat a pre-normalization configuration as wire truth.
- **No agent-framework lock-in.** LangGraph, Pydantic AI, custom loops, and
  other runtimes can implement the same small `Agent` protocol.
- **No automatic provider retry.** An ambiguous post-dispatch failure cannot
  be made scientifically equivalent by retrying it under the same episode
  identity.
- **No LLM judge in core.** DFAH measures agreement over a declared closed
  decision ontology; it does not outsource label validity or materiality to
  another model.
- **No raw prompts, tool arguments, or tool results by default.** Strong path
  identity uses local canonical hashes. Raw tool-argument capture is an
  explicit, policy-controlled exception.
- **No production truth claims from built-in fixtures.** The bundled
  synthetic suites test integration plumbing, not financial accuracy,
  fairness, policy compliance, or business materiality.
- **No full trace viewer or general eval platform.** Static reports and
  content-safe case inspection cover the first operational workflow. Richer
  visualization can remain a separate layer over the typed artifacts.

These boundaries keep the package useful as a control: small enough to add to
an existing system, strict enough that a passing report has a clear meaning,
and explicit about what still requires domain review.
