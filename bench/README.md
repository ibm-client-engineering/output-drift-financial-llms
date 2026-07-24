# DFAH-Bench research modules

`bench/` is the frozen research API and lineage pipeline for DFAH-Bench. The
supported prospective package lives in `src/dfah/`; start with
[`README_DFAH.md`](../README_DFAH.md) if you want to test an agent integration.

## Authoritative measurement

The corrected primary analysis pairs:

- **DAR**: modal decision agreement within an eligible replay group; and
- **TARseq**: modal exact ordered tool-name agreement on the same denominator.

Prospective captures can add **TARstrong**, which compares ordered tool names,
canonical arguments, and deterministic result identities.

A group is eligible only when its replay contract matches and every required
channel is present and valid. Missingness is not agreement. An observed empty
path remains a valid observed path.

See the [benchmark card](cards/benchmark_card.md) for the corrected counts,
study boundaries, non-claims, and version note.

## Historical modules

The directory retains earlier experimental modules for reproducibility and
lineage:

| Module | Purpose | Current status |
|---|---|---|
| `spec/` | Episode schemas and task ontologies | Retained |
| `metrics/dcb.py` | Cross-case decision concentration | Exploratory |
| `metrics/ecd.py` | Evidence-set distance implementation | Implementation retained; historical ECD result withdrawn |
| `metrics/scdr.py` | Optional text/embedding sensitivity | Exploratory; no hidden-reasoning claim |
| `stats/` | Bootstrap and permutation utilities | Retained |
| `provenance/` | Canonicalization, hash chains, signatures | Capability retained; no claim that provider bundles are publicly released |

The portfolio fixture and its dependent aggregates are excluded from the
corrected empirical analysis. Historical evidence-contact hashes were affected
by nonrandom legacy missingness and represented output-integrity fingerprints,
not source contacts; they are not used as v2 evidence.

## Package quickstart

```bash
python -m pip install -e ".[dev]"

dfah check-agent --agent dfah.demo:toy_agent
dfah run \
  --agent dfah.demo:toy_agent \
  --replays 3 \
  --out .dfah/runs/quickstart
dfah analyze .dfah/runs/quickstart
```

The built-in demo makes no network calls. A perfect score confirms its bounded
integration contract, not model quality or general determinism.

## Structure

```text
bench/
├── analysis/       # historical aggregation helpers
├── cards/          # corrected benchmark card
├── comparison/     # conventional-evaluation comparisons
├── export/         # dataset-export scaffolds
├── metrics/        # historical metric implementations
├── provenance/     # canonicalization, chains, certificates
├── spec/           # schemas and ontologies
└── stats/          # statistical utilities
```
