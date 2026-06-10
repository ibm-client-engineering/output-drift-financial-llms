# DFAH-Bench — paper reproduction and test targets
#
#   make reproduce-paper        # regenerate every paper number from raw replay
#                               # logs; fails loudly on any mismatch
#                               # (B=10,000 bootstrap, seed=42)
#   make reproduce-paper-fast   # skip bootstrap + subsampling stages
#   make test-bench             # unit/regression test suite (offline, no keys)
#
# See REPRODUCIBILITY.md for environment details.

PYTHON := python3

.PHONY: reproduce-paper reproduce-paper-fast test-bench

reproduce-paper:
	$(PYTHON) scripts/reproduce_paper.py

reproduce-paper-fast:
	$(PYTHON) scripts/reproduce_paper.py --skip-bootstrap --skip-subsampling

test-bench:
	$(PYTHON) -m pytest tests/ -q
