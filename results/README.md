# Machine-output versions

This directory deliberately preserves two analysis generations.

- The CSV files directly under `results/` are **archived arXiv v1 machine
  outputs**. They describe the historical 8,127-episode, three-task analysis
  and are retained for lineage. They are not the corrected paper result.
- [`v2/`](v2/) is the **corrected arXiv v2 release**. Its retrospective
  artifacts regenerate offline from a sanitized synthetic replay fixture, and
  its prospective extensions are aggregate-only files pinned by checksums.

Use `make reproduce-paper` for the corrected v2 release. Use
`make reproduce-paper-v1` only when reconstructing the archived v1 pipeline.
No v1 artifact was silently rewritten as v2.
