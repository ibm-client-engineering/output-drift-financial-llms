#!/usr/bin/env python3
"""Compatibility launcher for the historical DFAH replay demo."""

from scripts.workshop.run_dfah_demo import (
    OUTPUT_DIR,
    main,
    print_footer,
    print_header,
    save_results,
)

__all__ = ["OUTPUT_DIR", "main", "print_footer", "print_header", "save_results"]


if __name__ == "__main__":
    raise SystemExit(main())
