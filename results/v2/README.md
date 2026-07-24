# Corrected arXiv v2 artifacts

The corrected retrospective result covers **4,157 episodes from
configurations with observed tool use across 719 repeated case groups** in the
synthetic compliance and financial-DataOps tasks.

This phrasing is intentional: 25 retained episodes have an observed empty tool
sequence. They belong to nine groups in configurations that did use tools
elsewhere; observed tool use is a configuration-level inclusion rule.

## Layout

- `fixtures/retrospective_episodes.jsonl` is the public reproduction input.
  Each synthetic episode contains only model, task, synthetic case ID, replay
  index, normalized decision, and ordered tool names. It contains no prompt,
  tool arguments, tool results, credentials, or local paths.
- `retrospective/*.csv` contains the corrected v2 machine outputs. Run
  `make reproduce-paper` to regenerate them in a temporary directory and
  compare them with the committed files.
- `extensions/*.csv` contains safety-projected aggregate results for the
  prospective diagnostic, prospective component analysis, and local
  reconciliation extension. Their raw provider captures are approval-gated
  and are not published. The public target therefore checks their hashes,
  schemas, denominators, gates, and published values; it cannot regenerate
  them from raw captures.
- `manifest.json` pins the full public inventory, corpus lineage, scientific
  denominators, model identity, and SHA-256 digests.

The CSV files directly under the parent `results/` directory remain archived
v1 outputs. They are intentionally separate from this corrected v2 tree.

## Commands

```bash
make reproduce-paper       # corrected v2: regenerate retrospective + validate extensions
make reproduce-paper-v2    # explicit alias for the same corrected target
make verify-v2-manifest     # hashes, schemas, denominators, and release claims
make reproduce-paper-v1    # archived v1 lineage only
```

The v2 retrospective builder uses seed 42 for its declared bootstrap,
subsampling, and permutation procedures. It reproduces 719 case-level rows,
4,157 episodes, 627 unanimous-decision groups, and the 122-group sequence
variation decomposition (17 reorder-only, 58 multiplicity-only, 47 tool-set
changes).
