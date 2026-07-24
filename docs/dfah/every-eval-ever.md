# Every Eval Ever export

DFAH can export an artifact-verified replay run to the Every Eval Ever (EEE)
v0.2.2 interchange schema. EEE answers “how can evaluation results travel
between tools?” DFAH answers a different question: “which repeated executions
are comparable and sufficiently observed, and how far do decision and path
agreement separate?” The exporter preserves that distinction.

## Export a verified run

```bash
python -m dfah export .dfah/runs/MY-RUN \
  --format every-eval-ever \
  --out .dfah/exports/MY-RUN
```

The exporter refuses aggregate output when no replay group is eligible.
Unavailable DAR, TAR, gap, and flag-rate values are never converted into
numeric scores.

This produces:

- one aggregate JSON record with DAR, TARseq, TARbag, TARset, TARstrong,
  decision–path gap, eligible fraction, and flags per 100 groups; and
- one JSONL record per committed replay episode, including tool names,
  equality hashes, channel states, parse provenance, and replay identity.

Use `--aggregate-only` when episode-level interoperability is unnecessary.
For upstream model validation (Python 3.12 or newer, matching the upstream
package requirement):

```bash
python -m pip install -e '.[eee]'
python -m dfah export .dfah/runs/MY-RUN \
  --format every-eval-ever \
  --out .dfah/exports/MY-RUN \
  --validate
```

The Python API is equivalent:

```python
from dfah import export_every_eval_ever

result = export_every_eval_ever(
    ".dfah/runs/MY-RUN",
    ".dfah/exports/MY-RUN",
    validate=True,
)
print(result.aggregate_path, result.instances_sha256)
```

## Privacy boundary

The default profile deliberately excludes:

- prompts and raw case inputs;
- raw tool arguments and results;
- reasoning traces;
- provider endpoints and provider-native usage payloads; and
- organization identity unless the caller supplies it explicitly.

Artifact case identifiers, normalized decision labels, model/provider/adapter
identifiers, tool names, and SHA-256 equality fingerprints remain. Those
identifiers can reveal a deployment or business context even when captured
content is absent. Arbitrary request settings are committed by hash rather than
copied into the record. Hashes are integrity commitments, not anonymization:
low-entropy sensitive values can still be guessed. Use pseudonymous artifact
case IDs, generic deployment identifiers, and tokenize sensitive values before
capture.

The command only writes local mode-600 files. It does not upload, publish, or
register results with EEE or any other service.

## Semantic mapping

EEE’s aggregate score records carry DFAH’s metrics. Namespaced metadata retains
the suite version, fixture and schema commitments, manifest hash, replay
design, eligible denominator, and artifact root.

EEE instance records require an `is_correct` field. In a DFAH export that field
means only that the replay capture was eligible for the declared metric. It
does **not** mean the agent’s financial decision was correct. Each record states
this as `capture_eligibility_not_decision_correctness`.

EEE is therefore an optional reporting layer. It does not relax DFAH’s
manifest, suite-version, required-channel, or artifact-verification rules, and
records from different replay contracts must not be silently compared.

Upstream project and schema: <https://evalevalai.com/projects/every-eval-ever/>
