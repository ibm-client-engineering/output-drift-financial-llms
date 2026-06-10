"""Bundle verification: chain integrity + certificate + cross-layer consistency.

Provides a single verify_bundle() function that checks all layers, plus
export_bundle() for packaging chain + certificate for third-party audit.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .canonicalize import canonicalize
from .certificate import import_public_key, verify_certificate
from .chain import Chain, verify_chain


BUNDLE_FORMAT_VERSION = "1.0.0"


@dataclass
class VerificationResult:
    """Result of full bundle verification."""
    valid: bool
    chain_valid: bool
    certificate_valid: bool
    cross_layer_valid: bool
    errors: List[str] = field(default_factory=list)


def verify_bundle(bundle: Dict[str, Any]) -> VerificationResult:
    """Verify a complete provenance bundle.

    Checks:
        1. Chain integrity (I1, I2, I3, hash recomputation)
        2. Certificate signature (Ed25519)
        3. Cross-layer consistency (cert.head_hash == chain head,
           cert.genesis_hash == chain genesis, cert.event_count == chain length)

    Args:
        bundle: Dict with "chain", "certificate", "signature", "public_key".

    Returns:
        VerificationResult with per-layer status and error details.
    """
    errors = []

    # 1. Verify chain
    chain_data = bundle.get("chain", {})
    chain_valid, chain_error = verify_chain(chain_data)
    if not chain_valid:
        errors.append(f"Chain: {chain_error}")

    # 2. Verify certificate
    cert_data = bundle.get("certificate", {})
    sig_b64 = bundle.get("signature", "")
    pubkey_pem = bundle.get("public_key", "")

    certificate_valid = False
    try:
        pubkey = import_public_key(pubkey_pem)
        certificate_valid = verify_certificate(cert_data, sig_b64, pubkey)
        if not certificate_valid:
            errors.append("Certificate: signature verification failed")
    except Exception as e:
        errors.append(f"Certificate: {e}")

    # 3. Cross-layer consistency
    cross_layer_valid = True
    if chain_valid and certificate_valid:
        # Head hash
        events = chain_data.get("events", [])
        chain_head = events[-1]["event_hash"] if events else chain_data.get("genesis_hash", "")
        cert_head = cert_data.get("head_hash", "")
        if chain_head != cert_head:
            cross_layer_valid = False
            errors.append(
                f"Cross-layer: head_hash mismatch "
                f"(chain={chain_head[:16]}..., cert={cert_head[:16]}...)"
            )

        # Genesis hash
        if chain_data.get("genesis_hash") != cert_data.get("genesis_hash"):
            cross_layer_valid = False
            errors.append("Cross-layer: genesis_hash mismatch")

        # Event count
        if len(events) != cert_data.get("event_count", -1):
            cross_layer_valid = False
            errors.append(
                f"Cross-layer: event_count mismatch "
                f"(chain={len(events)}, cert={cert_data.get('event_count')})"
            )
    else:
        cross_layer_valid = False
        if not chain_valid:
            errors.append("Cross-layer: skipped (chain invalid)")
        if not certificate_valid:
            errors.append("Cross-layer: skipped (certificate invalid)")

    valid = chain_valid and certificate_valid and cross_layer_valid

    return VerificationResult(
        valid=valid,
        chain_valid=chain_valid,
        certificate_valid=certificate_valid,
        cross_layer_valid=cross_layer_valid,
        errors=errors,
    )


def export_bundle(
    chain: Chain,
    cert_result: Dict[str, Any],
) -> Dict[str, Any]:
    """Package a chain and certificate into a verifiable bundle.

    Args:
        chain: The hash chain.
        cert_result: Output from issue_certificate().

    Returns:
        Dict suitable for JSON serialization and third-party verification.
    """
    return {
        "version": BUNDLE_FORMAT_VERSION,
        "chain": chain.to_dict(),
        "certificate": cert_result["certificate"],
        "signature": cert_result["signature"],
        "public_key": cert_result["public_key"],
        "algorithm": cert_result.get("algorithm", "Ed25519"),
    }
