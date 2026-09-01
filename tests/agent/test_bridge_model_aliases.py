"""Tests for model_aliases in resolve_inspect_model."""

import pytest

from inspect_ai.agent._bridge.util import resolve_inspect_model
from inspect_ai.model._model import Model, get_model


def test_resolve_inspect_model_bare_inspect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INSPECT_EVAL_MODEL", "mockllm/default")
    model = resolve_inspect_model("inspect")
    assert str(model) == "mockllm/default"


def test_resolve_inspect_model_prefixed() -> None:
    model = resolve_inspect_model("inspect/mockllm/model")
    assert str(model) == "mockllm/model"


def test_resolve_inspect_model_alias_takes_priority() -> None:
    target = get_model("mockllm/alias-target")
    aliases: dict[str, str | Model] = {"my-alias": target}
    result = resolve_inspect_model("my-alias", model_aliases=aliases)
    assert result is target


def test_resolve_inspect_model_alias_string() -> None:
    aliases: dict[str, str | Model] = {"my-alias": "mockllm/alias-target"}
    result = resolve_inspect_model("my-alias", model_aliases=aliases)
    assert str(result) == "mockllm/alias-target"


def test_resolve_inspect_model_fallback_used_for_non_inspect() -> None:
    result = resolve_inspect_model(
        "some-random-model", fallback_model="inspect/mockllm/fallback"
    )
    assert str(result) == "mockllm/fallback"


def test_resolve_inspect_model_alias_over_fallback() -> None:
    """Aliases are checked before fallback."""
    target = get_model("mockllm/alias-target")
    aliases: dict[str, str | Model] = {"my-name": target}
    result = resolve_inspect_model(
        "my-name", model_aliases=aliases, fallback_model="inspect/mockllm/other"
    )
    assert result is target


def test_resolve_inspect_model_default_fallback_to_eval_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The documented default: foreign names fall back to the eval model."""
    monkeypatch.setenv("INSPECT_EVAL_MODEL", "mockllm/default")
    result = resolve_inspect_model("some-random-model", fallback_model="inspect")
    assert str(result) == "mockllm/default"


def test_resolve_inspect_model_fallback_never_hijacks_prefixed() -> None:
    """An inspect/-prefixed request names its model and passes through."""
    result = resolve_inspect_model("inspect/mockllm/named", fallback_model="inspect")
    assert str(result) == "mockllm/named"
    # Same under an inspect/-prefixed force: the explicit request wins.
    result = resolve_inspect_model(
        "inspect/mockllm/named", fallback_model="inspect/mockllm/fallback"
    )
    assert str(result) == "mockllm/named"


def test_resolve_inspect_model_bare_inspect_under_prefixed_force(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bare "inspect" keeps the eval model even under an inspect/ force."""
    monkeypatch.setenv("INSPECT_EVAL_MODEL", "mockllm/default")
    result = resolve_inspect_model("inspect", fallback_model="inspect/mockllm/fallback")
    assert str(result) == "mockllm/default"


def test_sandbox_agent_bridge_model_defaults_to_inspect() -> None:
    """The documented fallback exists by default (#5155)."""
    import inspect

    from inspect_ai.agent._bridge.sandbox.bridge import sandbox_agent_bridge

    assert (
        inspect.signature(sandbox_agent_bridge).parameters["model"].default == "inspect"
    )
