"""Strict decision extraction with explicit failure provenance."""

from __future__ import annotations

import re
from collections.abc import Sequence

from ..models import Decision, ParseProvenance

_MARKER = re.compile(r"(?im)^\s*DECISION\s*:\s*([A-Z][A-Z0-9_]*)\s*[.!]?\s*$")


def parse_decision(
    text: str, allowed_labels: Sequence[str]
) -> tuple[Decision | None, ParseProvenance]:
    """Extract exactly one allowed ``DECISION: LABEL`` marker.

    Malformed, absent, repeated, or out-of-ontology markers return no decision.
    No modal label, substring guess, or ontology fallback is ever substituted.
    """

    allowed = {label.strip().lower() for label in allowed_labels if label.strip()}
    if len(allowed) < 2:
        raise ValueError("allowed_labels must contain at least two distinct labels")
    matches = [match.lower() for match in _MARKER.findall(text)]
    if len(matches) != 1 or matches[0] not in allowed:
        return None, ParseProvenance(
            strategy="none",
            raw_span=None,
            confidence=None,
            fallback=True,
            accepted=False,
        )
    label = matches[0]
    return (
        Decision(label=label),
        ParseProvenance(
            strategy="strict_marker",
            raw_span=f"DECISION: {label.upper()}",
            confidence=1.0,
            fallback=False,
            accepted=True,
        ),
    )
