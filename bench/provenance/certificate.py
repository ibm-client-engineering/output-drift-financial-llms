"""L3 Ed25519 signing for chain certificates.

Inspired by the AEGIS protocol (Li, 2026), Apache-2.0 license.
See https://github.com/crabsatellite/aegis-protocol

Signs a canonicalized chain summary with Ed25519, enabling independent
third-party verification of chain integrity.

Dependencies: cryptography (Ed25519).
"""

import base64
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives import serialization

from .canonicalize import canonicalize
from .chain import Chain


def generate_keypair() -> Tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    """Generate an Ed25519 key pair for chain signing."""
    private_key = Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    return private_key, public_key


def export_public_key(key: Ed25519PublicKey) -> str:
    """Export a public key as PEM string."""
    return key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")


def import_public_key(pem: str) -> Ed25519PublicKey:
    """Import a public key from PEM string."""
    key = serialization.load_pem_public_key(pem.encode("utf-8"))
    if not isinstance(key, Ed25519PublicKey):
        raise TypeError(f"Expected Ed25519PublicKey, got {type(key).__name__}")
    return key


def _build_certificate_payload(
    chain: Chain,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the certificate payload dict from chain state."""
    return {
        "version": "1.0.0",
        "chain_id": chain.chain_id,
        "agent_id": chain.agent_id,
        "created_at": chain.created_at,
        "genesis_hash": chain.genesis_hash,
        "event_count": chain.length,
        "head_hash": chain.head,
        "issued_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }


def issue_certificate(
    chain: Chain,
    private_key: Ed25519PrivateKey,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Issue a signed certificate for a chain.

    The certificate payload is canonicalized and signed with Ed25519.

    Args:
        chain: The chain to certify.
        private_key: Ed25519 private key for signing.
        metadata: Optional metadata to include in the certificate.

    Returns:
        Dict with "certificate", "signature" (base64), and "public_key" (PEM).
    """
    payload = _build_certificate_payload(chain, metadata)
    canonical_bytes = canonicalize(payload)
    signature = private_key.sign(canonical_bytes)

    return {
        "certificate": payload,
        "signature": base64.b64encode(signature).decode("ascii"),
        "public_key": export_public_key(private_key.public_key()),
        "algorithm": "Ed25519",
    }


def verify_certificate(
    certificate: Dict[str, Any],
    signature_b64: str,
    public_key: Ed25519PublicKey,
) -> bool:
    """Verify a certificate's Ed25519 signature.

    Args:
        certificate: The certificate payload dict.
        signature_b64: Base64-encoded signature string.
        public_key: Ed25519 public key for verification.

    Returns:
        True if the signature is valid.
    """
    try:
        canonical_bytes = canonicalize(certificate)
        signature = base64.b64decode(signature_b64)
        public_key.verify(signature, canonical_bytes)
        return True
    except Exception:
        return False
