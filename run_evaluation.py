#!/usr/bin/env python3
"""Compatibility launcher for the historical output-drift runner."""

import asyncio
import sys

from scripts.workshop import run_evaluation as _implementation

if __name__ == "__main__":
    raise SystemExit(asyncio.run(_implementation.main()))

sys.modules[__name__] = _implementation
