# DFAH-Bench — paper reproduction and test targets
#
#   make reproduce-paper        # corrected arXiv v2 offline reproduction
#   make reproduce-paper-v1     # archived v1 lineage reproduction
#   make verify-v2-manifest     # verify v2 hashes, schemas, and denominators
#   make test-bench             # frozen research tests (offline, no keys)
#   make test-dfah              # installable package tests
#   make test-all               # both layers
#
# See REPRODUCIBILITY.md for environment details.

PYTHON ?= python3

.PHONY: reproduce-paper reproduce-paper-v2 verify-v2-manifest \
	reproduce-paper-v1 reproduce-paper-v1-fast test-bench test-dfah test-all

reproduce-paper: reproduce-paper-v2

reproduce-paper-v2:
	$(PYTHON) scripts/reproduce_paper_v2.py

verify-v2-manifest:
	$(PYTHON) scripts/make_v2_manifest.py --check

reproduce-paper-v1:
	$(PYTHON) scripts/reproduce_paper.py

reproduce-paper-v1-fast:
	$(PYTHON) scripts/reproduce_paper.py --skip-bootstrap --skip-subsampling

test-bench:
	$(PYTHON) -m pytest tests/ --ignore=tests/dfah -q

test-dfah:
	$(PYTHON) -m pytest tests/dfah -q

test-all:
	$(PYTHON) -m pytest tests/ -q
