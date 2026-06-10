"""Tests for Ed25519 certificate signing and verification."""

import pytest
from bench.provenance.chain import Chain
from bench.provenance.certificate import (
    generate_keypair,
    issue_certificate,
    verify_certificate,
    export_public_key,
    import_public_key,
)


def _make_signed_chain():
    """Helper: create chain, sign it, return (chain, cert_result, keys)."""
    chain = Chain("test-chain", "test-agent")
    chain.append("test.event", {"data": "payload"}, timestamp="2026-03-22T10:00:00Z")
    chain.append("test.event", {"data": "payload2"}, timestamp="2026-03-22T10:01:00Z")

    private_key, public_key = generate_keypair()
    cert_result = issue_certificate(chain, private_key, metadata={"issuer": "test"})

    return chain, cert_result, private_key, public_key


class TestKeypairGeneration:

    def test_generates_keypair(self):
        private, public = generate_keypair()
        assert private is not None
        assert public is not None

    def test_different_each_time(self):
        _, pub1 = generate_keypair()
        _, pub2 = generate_keypair()
        assert export_public_key(pub1) != export_public_key(pub2)


class TestSignVerifyRoundtrip:

    def test_valid_certificate_verifies(self):
        _, cert_result, _, public_key = _make_signed_chain()
        assert verify_certificate(
            cert_result["certificate"],
            cert_result["signature"],
            public_key,
        )

    def test_certificate_has_expected_fields(self):
        _, cert_result, _, _ = _make_signed_chain()
        cert = cert_result["certificate"]
        assert cert["version"] == "1.0.0"
        assert cert["chain_id"] == "test-chain"
        assert cert["agent_id"] == "test-agent"
        assert cert["event_count"] == 2
        assert "head_hash" in cert
        assert "genesis_hash" in cert
        assert "issued_at" in cert


class TestTamperDetection:

    def test_tampered_certificate_rejected(self):
        """Modify certificate payload after signing → verification fails."""
        _, cert_result, _, public_key = _make_signed_chain()
        cert_result["certificate"]["event_count"] = 999
        assert not verify_certificate(
            cert_result["certificate"],
            cert_result["signature"],
            public_key,
        )

    def test_wrong_key_rejected(self):
        """Verify with a different key → fails."""
        _, cert_result, _, _ = _make_signed_chain()
        _, wrong_public = generate_keypair()
        assert not verify_certificate(
            cert_result["certificate"],
            cert_result["signature"],
            wrong_public,
        )

    def test_corrupted_signature_rejected(self):
        """Corrupt the signature → fails."""
        _, cert_result, _, public_key = _make_signed_chain()
        import base64
        sig = base64.b64decode(cert_result["signature"])
        corrupted = base64.b64encode(bytes([b ^ 0xFF for b in sig])).decode()
        assert not verify_certificate(
            cert_result["certificate"],
            corrupted,
            public_key,
        )


class TestKeyExportImport:

    def test_roundtrip(self):
        _, public = generate_keypair()
        pem = export_public_key(public)
        restored = import_public_key(pem)
        assert export_public_key(restored) == pem

    def test_pem_format(self):
        _, public = generate_keypair()
        pem = export_public_key(public)
        assert pem.startswith("-----BEGIN PUBLIC KEY-----")
        assert pem.strip().endswith("-----END PUBLIC KEY-----")
