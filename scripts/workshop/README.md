# Workshop utilities

This package contains the implementations of the historical output-drift
workshop commands:

| Stable command | Implementation |
| --- | --- |
| `python run_evaluation.py` | `scripts/workshop/run_evaluation.py` |
| `python run_dfah_demo.py` | `scripts/workshop/run_dfah_demo.py` |
| `python plot_results.py` | `scripts/workshop/plot_results.py` |
| `python make_tables.py` | `scripts/workshop/make_tables.py` |

Keep the root launchers. They are compatibility entry points for published
labs, examples, and external links. New implementation work belongs in this
package; the root files should remain thin launchers.

Run these commands from the repository root unless a command documents a
different working directory. Their established output locations remain at the
repository root, including `results/`, `traces/`, `figs/`, `tables/`, and
`dfah_results/`.

The corrected v2 reproduction under `bench/` and `results/v2/` is a separate,
frozen publication surface. Do not move or rename its manifest-listed files as
part of workshop maintenance.

## Capture-loss example (Lab 11)

```bash
python3 scripts/workshop/capture_loss_demo.py
```

This standard-library example constructs three replays and shows how repeated
omissions preserve agreement. It compares each record with its supplied
execution reference and keeps missing records distinct from observed empty
paths. It makes no model or network calls and contains no workshop study data.
See [Lab 11](../../docs/lab-11/README.md) and the
[workshop studies](../../docs/dfah/workshop-studies.md).
