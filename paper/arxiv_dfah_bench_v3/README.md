# DFAH-Bench v3 manuscript

These sources connect replay observability to evidence, authorization, task
completion and cost. The revision retains the earlier replay evidence and adds
an interactive banking study with 1,080 scheduled episodes and 1,944 fixed-state
gate probes. Replay agreement, captured execution, native task success and
constructed policy-label agreement retain their own units and denominators.
Missing outcomes leave the prespecified confirmatory families unavailable; the
paper reports descriptive comparisons and finite-schedule bounds with their
coverage and limitations.

## Build

Use a TeX Live or MacTeX installation with `latexmk` and BibTeX. From the
repository root:

```sh
cd paper/arxiv_dfah_bench_v3
latexmk -pdf -interaction=nonstopmode -halt-on-error -no-shell-escape main.tex
```

The output is `main.pdf`. The build uses the included bibliography, style and
eight figure PDFs. It requires no model API calls. The source directory includes
only the active manuscript inputs; generated TeX files are ignored.

## Reproduction scope

Compiling the paper reproduces its presentation from the included figures.
The manuscript explains the separate evidence and access requirements for
regenerating analyses from retained traces. The historical v2 reproduction
commands remain documented in [REPRODUCIBILITY.md](../../REPRODUCIBILITY.md),
with their original fixtures and manifests preserved. The package has its own
[guide](../../README_DFAH.md) and tests.
