# Lab 9: Replay, Review, and Retest

Run a bounded review cycle with **DFAH-Bench 0.1.2**: capture a path change,
inspect the failed policy checks, and evaluate a supplied correction against
the same policy. Then export the verified evidence for other evaluation tools.

**Duration:** about 25 minutes. **Requirements:** Python 3.10+, Git, and an
internet connection for installation. The experiment uses local synthetic
tools and needs no model service or API key. Optional upstream export validation
requires Python 3.12+.

Preview the [recorded example in the explorer](../explorer/index.html#review-loop).
That page displays saved results; the steps below execute the Python example.

## 1. Install the released package and example

Use a fresh directory and the release tag so the example matches the package:

```bash
git clone --branch dfah-v0.1.2 --depth 1 \
  https://github.com/ibm-client-engineering/output-drift-financial-llms.git dfah-012
cd dfah-012
python -m venv .venv-dfah
source .venv-dfah/bin/activate
python -m pip install 'dfah-bench==0.1.2'
python -m dfah --version
```

The version should be `0.1.2`. On Windows, activate with
`.venv-dfah\Scripts\activate` instead. The example is in the tagged repository;
the package itself comes from PyPI.

## 2. Run the two-candidate loop

```bash
python examples/dfah_gate_loop.py --out .dfah/lab9-review
```

Each candidate runs two cases twice, calling `read_status` and `read_rule`
in every episode: **eight episodes and 16 recorded tool calls** in total.
The first candidate reverses the order on the second replay of each case.
The second uses a fixed order. Both implementations are prewritten; the loop
evaluates exactly two candidates and does not generate or edit code.

| Candidate | DAR | TARseq | Flagged groups | Gate |
|---|---:|---:|---:|---|
| `01-varying-path` | 1.0 | 0.5 | 2 of 2 | Fail |
| `02-fixed-path` | 1.0 | 1.0 | 0 of 2 | Pass |

The command exits successfully because the supplied correction passes. The
earlier failure remains recorded. Reusing this output directory is rejected
before existing evidence changes; use a fresh `--out` for another run.

## 3. Inspect what changed and what stayed fixed

Open the two local reports in your browser:

```text
.dfah/lab9-review/01-varying-path/report.html
.dfah/lab9-review/02-fixed-path/report.html
```

Or inspect the recorded episodes in your terminal:

```bash
python -m dfah inspect .dfah/lab9-review/01-varying-path --case CASE-001
python -m dfah inspect .dfah/lab9-review/02-fixed-path --case CASE-001
```

For `CASE-001`, both decisions are `proceed`. Only the first candidate changes
the order of its two tool calls. The second case keeps its `review` decision
and shows the same path pattern.

The candidates have separate output directories, adapter versions,
implementation hashes, manifests, and episode commitments. The cases, tool
schemas, and gate policy stay fixed. `summary.json` records both outcomes and
their commitments; each candidate's `gate.json` records individual checks.
Each gate uses a report reloaded and verified against its committed episode store.

The policy in `.dfah/lab9-review/policy.json` requires complete,
artifact-verified evidence, two eligible groups, two replays per case, full
eligibility, decision and ordered-path agreement of 1.0, and zero gap or flags.
Run the checks separately:

```bash
python -m dfah gate \
  .dfah/lab9-review/01-varying-path \
  .dfah/lab9-review/policy.json
```

**Expected exit code: 1.** The failed checks are `tar_seq`, `gap`, and
`flags_per_100_cases`. This is the intended failure. Then run:

```bash
python -m dfah gate \
  .dfah/lab9-review/02-fixed-path \
  .dfah/lab9-review/policy.json
```

**Expected exit code: 0.** Do not join these commands with `&&`: the first
failure would prevent the second check from running.

The two read tools in this example are independent. A path change therefore
provides a review signal without establishing a wrong decision. The correction
meets the declared policy on these cases; that is bounded repeatability evidence.

## 4. Require a policy in blocking mode

In 0.1.2, a blocking run requires an explicit policy before the agent is loaded.
This separate CLI demonstration uses the packaged toy suite, with the same
thresholds. It is not a third candidate in the paired comparison above.

```bash
python -m dfah run \
  --agent dfah.demo:toy_agent \
  --mode blocking \
  --policy .dfah/lab9-review/policy.json \
  --replays 2 --episode-timeout-s 5 \
  --out .dfah/lab9-review/toy-blocking
```

The deterministic toy run passes. Omitting `--policy` is a configuration error;
a completed replay that fails its policy exits with code 1 and retains its
evidence for review. For pytest integrations, an explicit `--dfah-policy` is
also checked before test collection, even if no test requests a DFAH fixture.
See the [package quickstart](../dfah/quickstart.md) for pytest setup.

## 5. Export verified results

```bash
python -m dfah export \
  .dfah/lab9-review/02-fixed-path \
  --format every-eval-ever \
  --out .dfah/lab9-review/eee-fixed
```

The command prints paths to a UUID-named `.every-eval-ever.json` aggregate
and a `.jsonl` file with **four committed episode records**. It writes local
files; it does not upload them. Add `--aggregate-only` when you need only the
summary. The exporter checks that the report and episode snapshot match.

For optional validation against the upstream schema, use Python 3.12+:

```bash
python -m pip install 'dfah-bench[eee]==0.1.2'
python -m dfah export \
  .dfah/lab9-review/02-fixed-path \
  --format every-eval-ever \
  --out .dfah/lab9-review/eee-validated \
  --validate
```

The extra pins upstream validator `0.2.3rc1`; the exported schema is `0.2.2`.
EEE's required `is_correct` field means **membership in an eligible replay
group** here. It means neither that the gate passed nor that the financial
decision was correct. The export includes the explicit semantic marker
`replay_group_retention_not_decision_correctness`.

Raw prompts, arguments, and results are excluded. Case IDs, decision labels,
tool names, deployment identifiers, and equality hashes remain; inspect those
before sharing a real deployment's export. Read the
[export guide](../dfah/every-eval-ever.md) for the exact mapping and privacy boundary.

## Apply this to your own change

Choose a small synthetic suite representative of your tool contract. Fix the
policy before evaluating the change, give each implementation a distinct version
and run directory, and review the failed checks and paths before accepting it.
Decide separately whether the observed differences matter to the task.

- [Bring your own agent](../dfah/bring-your-own-agent.md)
- [Full example guide](../dfah/replay-review-loop.md)
- [Lab 8: Replay Measurement](../lab-8/README.md)
- [Return to the explorer](../explorer/index.html#review-loop)
