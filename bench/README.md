# DFAH-Bench

**Behavioral consistency evaluation for LLM agents in financial decision-making.**

DFAH-Bench measures observable agent instability through divergence channels — revealing when agents reach the same conclusion through different trajectories, different evidence contacts, or different rationales.

## Quick Start

```python
from bench.metrics.dcb import compute_dcb
from bench.metrics.ecd import compute_ecd
from bench.spec.schema import load_episodes

# Load existing replay traces
episodes = load_episodes("econometrics/benchmarks/results/run_logs/compliance/qwen2.5_7b-instruct")

# Compute Decision Concentration Bias
decisions = [ep.decision.label for ep in episodes]
dcb_result = compute_dcb(decisions, benchmark="compliance")
print(f"DCB: {dcb_result.dcb:.3f}")  # 1.0 = all mass on single decision

# Compute Evidence Contact Divergence (when evidence data is available)
evidence_sets = [
    {ec.source_id for ec in ep.evidence_contacts}
    for ep in episodes if ep.evidence_contacts
]
if evidence_sets:
    ecd_result = compute_ecd(evidence_sets)
    print(f"ECD: {ecd_result.ecd:.3f}")
```

## Divergence Channels

| Channel | What it measures | Metric | Data requirement |
|---------|-----------------|--------|-----------------|
| Decision concentration | Does the model collapse to one action across cases? | DCB | `decision_output` aggregated over cases |
| Trajectory divergence | Do runs use different tool-call sequences? | Exact-sequence agreement / DAR-TAR gap (paper-level); optional SCDR | `tool_sequence` (partial coverage) |
| Evidence-contact divergence | Do runs consult different evidence? | ECD | Tool outputs (partial coverage) |
| Rationale divergence | Do runs produce different reasoning? | SCDR (rationale mode) | Full reasoning text (unavailable in current data) |

## Positioning vs. Related Work

| Work | What it measures | Domain | Multi-run | Reasoning paths | Provenance |
|------|-----------------|--------|-----------|-----------------|------------|
| HELM (Liang et al.) | Output correctness, calibration | General | No | No | No |
| FinBen (Xie et al.) | Financial task accuracy | Finance | No | No | No |
| Wang & Wang 2025 | Output agreement across runs | Finance | Yes (50 runs) | No | No |
| ReasonBENCH (Potamitis et al.) | Reasoning stability | General | Yes | Partial | No |
| Self-Consistency (Wang et al.) | Majority vote over CoT | General | Yes | Yes (exploits it) | No |
| AEGIS (Li 2026) | Execution provenance | Agent-general | No | No | Yes |
| **DFAH-Bench (ours)** | **Behavioral consistency** | **Finance** | **Yes (8,129 raw replay episodes)** | **Yes (observable path/evidence channels)** | **Yes** |

## Module Structure

```
bench/
├── spec/           # Schema, taxonomy, channel definitions
│   ├── schema.py   # ReplayEpisode, DivergenceChannel, loaders
│   └── taxonomy.py # Decision ontologies from benchmark task truth
├── metrics/        # Behavioral consistency metrics
│   ├── dcb.py      # Decision Concentration Bias
│   ├── ecd.py      # Evidence Contact Divergence
│   └── scdr.py     # Same Conclusion Different Reasoning
├── stats/          # Statistical utilities
│   ├── bootstrap.py    # Percentile bootstrap CIs
│   └── significance.py # Permutation tests, Spearman correlation
├── analysis/       # Aggregation helpers
│   └── aggregate.py    # Case/task/model aggregation for paper tables
├── provenance/     # Cryptographic audit bundles
│   ├── canonicalize.py # Deterministic JSON serialization
│   ├── chain.py        # SHA-256 hash chain (AEGIS-inspired)
│   ├── certificate.py  # Ed25519 signing
│   └── verify.py       # Bundle verification
├── comparison/     # Comparison with conventional evaluation
├── export/         # Dataset export (HuggingFace, Croissant) — stubs
└── README.md
```

## Provenance

Every batch of replay runs can be packaged into a verifiable audit bundle:

```python
from bench.provenance.chain import Chain
from bench.provenance.certificate import generate_keypair, issue_certificate
from bench.provenance.verify import export_bundle, verify_bundle

# Build chain
chain = Chain("experiment-001", "researcher-alice")
chain.append("compliance.qwen3_8b.replay", {"case_id": "TXN-001", "decision": "escalate"})

# Sign and export
private_key, public_key = generate_keypair()
cert = issue_certificate(chain, private_key)
bundle = export_bundle(chain, cert)

# Verify
result = verify_bundle(bundle)
assert result.valid
```
