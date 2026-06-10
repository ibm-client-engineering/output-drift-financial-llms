"""Tests for full bundle verification."""

import json
import tempfile
from pathlib import Path

import pytest
from bench.provenance.chain import Chain
from bench.provenance.certificate import generate_keypair, issue_certificate
from bench.provenance.verify import verify_bundle, export_bundle, VerificationResult


def _make_bundle():
    """Helper: create a complete verifiable bundle."""
    chain = Chain("bundle-chain", "bundle-agent")
    chain.append("test.event", {"data": "a"}, timestamp="2026-03-22T10:00:00Z")
    chain.append("test.event", {"data": "b"}, timestamp="2026-03-22T10:01:00Z")

    private_key, public_key = generate_keypair()
    cert_result = issue_certificate(chain, private_key)
    bundle = export_bundle(chain, cert_result)
    return bundle, chain, cert_result


class TestBundleRoundtrip:

    def test_valid_bundle_verifies(self):
        bundle, _, _ = _make_bundle()
        result = verify_bundle(bundle)
        assert result.valid
        assert result.chain_valid
        assert result.certificate_valid
        assert result.cross_layer_valid
        assert result.errors == []

    def test_json_roundtrip(self):
        """Bundle survives JSON serialization."""
        bundle, _, _ = _make_bundle()
        json_str = json.dumps(bundle)
        restored = json.loads(json_str)
        result = verify_bundle(restored)
        assert result.valid

    def test_bundle_has_version(self):
        bundle, _, _ = _make_bundle()
        assert "version" in bundle
        assert bundle["version"] == "1.0.0"


class TestCrossLayerConsistency:

    def test_head_hash_mismatch_detected(self):
        """Modify chain after signing → cross-layer fails."""
        bundle, chain, _ = _make_bundle()
        # Tamper with chain's last event hash
        bundle["chain"]["events"][-1]["event_hash"] = "0" * 64
        result = verify_bundle(bundle)
        assert not result.valid
        # Chain verification itself should fail too
        assert not result.chain_valid

    def test_event_count_mismatch_detected(self):
        """Add extra event to chain after signing → cross-layer fails."""
        bundle, chain, _ = _make_bundle()
        # Add a well-formed but unsigned event
        last = bundle["chain"]["events"][-1]
        bundle["chain"]["events"].append({
            "seq": last["seq"] + 1,
            "event_type": "sneaky.event",
            "timestamp": "2026-03-22T10:05:00Z",
            "payload_hash": "a" * 64,
            "prev_hash": last["event_hash"],
            "event_hash": "b" * 64,
        })
        result = verify_bundle(bundle)
        assert not result.valid


class TestMalformedBundles:

    def test_missing_chain(self):
        bundle, _, _ = _make_bundle()
        del bundle["chain"]
        result = verify_bundle(bundle)
        assert not result.valid

    def test_missing_certificate(self):
        bundle, _, _ = _make_bundle()
        del bundle["certificate"]
        result = verify_bundle(bundle)
        assert not result.valid

    def test_empty_bundle(self):
        result = verify_bundle({})
        assert not result.valid
