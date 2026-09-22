"""External schemas are data; argument checks must not retrieve remote resources."""

from __future__ import annotations

import urllib.request

import anyio
import pytest

from dfah import ChannelState, ToolExecutionError, ToolExecutionState, ToolRegistry, ToolSpec


@pytest.mark.parametrize("keyword", ["$ref", "$dynamicRef"])
@pytest.mark.parametrize(
    "reference",
    [
        "https://schema.invalid/private-context/schema.json",
        "http://127.0.0.1:9/schema.json",
        "file:///synthetic-private-context/schema.json",
    ],
)
def test_tool_schema_external_references_never_retrieve(monkeypatch, keyword, reference):
    retrievals = []

    def deny_transport(request, *args, **kwargs):
        retrievals.append(request)
        raise AssertionError("argument validation attempted resource retrieval")

    monkeypatch.setattr(urllib.request, "urlopen", deny_transport)
    spec = ToolSpec(
        name="synthetic",
        input_schema={"type": "object", "properties": {"value": {keyword: reference}}},
    )
    with pytest.raises(ValueError, match="cannot be resolved locally") as direct:
        spec.validate_arguments({"value": "example"})
    assert reference not in str(direct.value)

    executed = []
    registry = ToolRegistry()
    registry.register(spec, lambda **kwargs: executed.append(kwargs))
    session = registry.session()

    async def invoke():
        with pytest.raises(ToolExecutionError, match="cannot be resolved locally") as result:
            await session.call("synthetic", value="example")
        assert reference not in str(result.value)

    anyio.run(invoke)
    assert retrievals == []
    assert executed == []
    assert len(session.calls) == 1
    assert session.calls[0].execution_state is ToolExecutionState.REJECTED
    assert session.calls[0].result_state is ChannelState.MALFORMED
    assert session.calls[0].output_hash is None


@pytest.mark.parametrize("reference", ["#/$defs/positive", "#positive"])
def test_in_document_tool_schema_references_still_validate(monkeypatch, reference):
    def deny_transport(*args, **kwargs):
        raise AssertionError("local schema validation must not retrieve anything")

    monkeypatch.setattr(urllib.request, "urlopen", deny_transport)
    spec = ToolSpec(
        name="double",
        input_schema={
            "$id": "https://schema.invalid/local-root.json",
            "type": "object",
            "$defs": {"positive": {"$anchor": "positive", "type": "integer", "minimum": 1}},
            "properties": {"value": {"$ref": reference}},
            "required": ["value"],
        },
    )
    spec.validate_arguments({"value": 2})
    with pytest.raises(ValueError, match="violate the declared JSON Schema"):
        spec.validate_arguments({"value": 0})
    registry = ToolRegistry()
    registry.register(spec, lambda value: value * 2)
    session = registry.session()

    async def invoke():
        assert await session.call("double", value=2) == 4
        with pytest.raises(ToolExecutionError, match="violate its JSON Schema"):
            await session.call("double", value=0)

    anyio.run(invoke)
    assert [call.execution_state for call in session.calls] == [
        ToolExecutionState.EXECUTED,
        ToolExecutionState.REJECTED,
    ]
