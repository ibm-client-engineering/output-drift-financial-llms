"""Tests for deterministic JSON serialization.

Tests cover deterministic serialization behaviors used by the provenance
hash chain. Does not claim full RFC 8785 compliance.
"""

from bench.provenance.canonicalize import canonicalize, canonicalize_str


class TestCanonicalize:

    def test_sorted_keys(self):
        assert canonicalize_str({"z": 1, "a": 2}) == '{"a":2,"z":1}'

    def test_nested_sorted_keys(self):
        obj = {"b": {"d": 1, "c": 2}, "a": 3}
        assert canonicalize_str(obj) == '{"a":3,"b":{"c":2,"d":1}}'

    def test_no_whitespace(self):
        result = canonicalize_str({"key": [1, 2, 3]})
        assert " " not in result
        assert "\n" not in result

    def test_null_boolean(self):
        assert canonicalize_str(None) == "null"
        assert canonicalize_str(True) == "true"
        assert canonicalize_str(False) == "false"

    def test_empty_structures(self):
        assert canonicalize_str({}) == "{}"
        assert canonicalize_str([]) == "[]"

    def test_array_order_preserved(self):
        assert canonicalize_str([3, 1, 2]) == "[3,1,2]"

    def test_string_escaping(self):
        result = canonicalize_str({"a": 'hello "world"'})
        assert '"hello \\"world\\""' in result

    def test_deterministic(self):
        """Same input always produces same output."""
        obj = {"b": 2, "a": 1, "c": [3, 2, 1]}
        assert canonicalize(obj) == canonicalize(obj)

    def test_returns_bytes(self):
        result = canonicalize({"key": "value"})
        assert isinstance(result, bytes)

    def test_utf8_encoding(self):
        result = canonicalize({"key": "value"})
        assert result == b'{"key":"value"}'

    def test_integer_serialization(self):
        assert canonicalize_str(42) == "42"
        assert canonicalize_str(0) == "0"

    def test_float_serialization(self):
        assert canonicalize_str(1.5) == "1.5"
