"""Tests for factor registration / translation / versioned snapshot (DORA-145 / F-02).

Covers the three deliverables without needing py-alpha-lib installed:

* **translator** — delegates to ``alpha.lang.to_python`` with the ExecContext
  naming convention (``str.upper``), rejects blank input, wraps parse failures;
* **snapshot store** — register returns ``{factor_id, version}``, is idempotent,
  versions increment on new expressions, and a loaded snapshot computes offline
  (no re-translation);
* **security** — the path gate (traversal / bad version) and the content
  whitelist gate (import / eval / attribute / extra-def injection) both reject
  before any import.

No network; deterministic.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import src.factor_runtime.availability as availability
import src.factor_runtime.snapshot as snapshot
from src.factor_runtime import (
    FactorTranslationError,
    SnapshotNotFoundError,
    SnapshotStore,
    SnapshotValidationError,
    reset_probe,
    render_snapshot,
    translate_expression,
    validate_snapshot_source,
)

# A minimal, valid translated body the whitelist accepts.
_BODY = "def compute(ctx):\n    return ctx('CLOSE')\n"


@pytest.fixture(autouse=True)
def _fresh_probe_and_store():
    reset_probe()
    snapshot.reset_snapshot_store()
    yield
    reset_probe()
    snapshot.reset_snapshot_store()


# --------------------------------------------------------------------------- #
# Fake alpha module (availability gate + translator)                          #
# --------------------------------------------------------------------------- #


def _fake_alpha(to_python: Any | None = None) -> types.ModuleType:
    """A fake ``alpha`` package with the sanity marker + a ``lang`` translator."""
    mod = types.ModuleType("alpha")
    mod.MA = lambda data, periods: data  # sanity marker for the availability probe

    lang = types.ModuleType("alpha.lang")

    def _default_to_python(name: str, code: str, **kwargs: Any) -> str:
        assert kwargs.get("as_function") is True
        assert kwargs.get("name_convertor") is str.upper
        return _BODY

    lang.to_python = to_python or _default_to_python
    mod.lang = lang
    return mod


def _install_fake_alpha(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    fake = _fake_alpha()
    monkeypatch.setitem(sys.modules, "alpha", fake)
    reset_probe()
    return fake


def _force_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(name: str, *args: Any, **kwargs: Any) -> Any:
        raise ImportError(f"No module named {name!r} (test)")

    monkeypatch.setattr(availability.importlib, "import_module", _raise)
    reset_probe()


def _store(tmp_path: Path) -> SnapshotStore:
    return SnapshotStore(root=tmp_path / "factors")


# --------------------------------------------------------------------------- #
# Translator                                                                   #
# --------------------------------------------------------------------------- #


def test_translate_expression_delegates_to_alpha_lang(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def _to_python(name: str, code: str, **kwargs: Any) -> str:
        captured["name"] = name
        captured["code"] = code
        captured.update(kwargs)
        return _BODY

    _install_fake_alpha(monkeypatch).lang.to_python = _to_python
    body = translate_expression("close/ref(close,1)-1")

    assert body == _BODY
    assert captured["name"] == "compute"
    assert captured["code"] == "close/ref(close,1)-1"
    assert captured["as_function"] is True
    assert captured["name_convertor"] is str.upper


def test_translate_expression_rejects_blank(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_alpha(monkeypatch)
    for blank in ("", "   ", "\n"):
        with pytest.raises(FactorTranslationError):
            translate_expression(blank)


def test_translate_expression_wraps_parse_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(name: str, code: str, **kwargs: Any) -> str:
        raise ValueError("parse exploded")

    _install_fake_alpha(monkeypatch).lang.to_python = _boom
    with pytest.raises(FactorTranslationError) as exc_info:
        translate_expression("close +")
    assert "parse exploded" in str(exc_info.value)


# --------------------------------------------------------------------------- #
# Snapshot store                                                               #
# --------------------------------------------------------------------------- #


def test_register_returns_id_and_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(snapshot, "translate_expression", lambda expr: _BODY)
    store = _store(tmp_path)

    result = store.register("close/ref(close,1)-1", name="momentum")

    assert result["factor_id"] == "momentum"
    assert result["version"] == 1
    assert result["expression"] == "close/ref(close,1)-1"
    written = tmp_path / "factors" / "momentum" / "1" / "snapshot.py"
    assert written.is_file()
    # The written snapshot must pass our own content whitelist.
    validate_snapshot_source(written.read_text(encoding="utf-8"))


def test_register_is_idempotent_for_same_expression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(snapshot, "translate_expression", lambda expr: _BODY)
    store = _store(tmp_path)

    first = store.register("close", name="m")
    second = store.register("close", name="m")

    assert first["version"] == second["version"] == 1
    assert store.list_versions("m") == [1]


def test_register_increments_version_for_new_expression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(snapshot, "translate_expression", lambda expr: _BODY)
    store = _store(tmp_path)

    store.register("close", name="m")
    second = store.register("close/ref(close,1)-1", name="m")

    assert second["version"] == 2
    assert store.list_versions("m") == [1, 2]


def test_register_derives_deterministic_factor_id_without_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(snapshot, "translate_expression", lambda expr: _BODY)
    store = _store(tmp_path)

    a = store.register("close/ref(close,1)-1")
    b = store.register("close/ref(close,1)-1")

    assert a["factor_id"].startswith("f")
    assert a["factor_id"] == b["factor_id"]
    assert a["version"] == b["version"] == 1


def test_register_rejects_empty_expression(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _store(tmp_path).register("   ")


def test_load_snapshot_computes_offline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Loading + computing must not touch the online translator again."""
    calls: list[str] = []

    def _translator(expr: str) -> str:
        calls.append(expr)
        return _BODY

    monkeypatch.setattr(snapshot, "translate_expression", _translator)
    store = _store(tmp_path)
    store.register("close", name="m")

    # Drop the translator entirely, then load and compute.
    loaded = store.load("m", 1)

    class _Ctx:
        def __call__(self, name: str) -> Any:
            assert name == "CLOSE"
            return [1.0, 2.0, 3.0]

    assert loaded.compute(_Ctx()) == [1.0, 2.0, 3.0]
    assert calls == ["close"]  # translated exactly once, at register time


def test_load_unknown_snapshot_raises_not_found(tmp_path: Path) -> None:
    with pytest.raises(SnapshotNotFoundError):
        _store(tmp_path).load("m", 1)


# --------------------------------------------------------------------------- #
# Security: path gate                                                          #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad_id", ["../etc", "a/b", ".hidden", "A", "a" * 33, "foo.bar"])
def test_path_gate_rejects_bad_factor_id(tmp_path: Path, bad_id: str) -> None:
    with pytest.raises(SnapshotValidationError):
        _store(tmp_path).load(bad_id, 1)


@pytest.mark.parametrize("bad_version", [0, -1, True, "1"])
def test_path_gate_rejects_bad_version(tmp_path: Path, bad_version: Any) -> None:
    with pytest.raises(SnapshotValidationError):
        _store(tmp_path).load("m", bad_version)


# --------------------------------------------------------------------------- #
# Security: content whitelist gate                                             #
# --------------------------------------------------------------------------- #


def _wrapped(body_expr: str, imports: str = "import numpy as np") -> str:
    return (
        f"{imports}\n\n\ndef compute(ctx):\n    return {body_expr}\n\n\n"
        "__snapshot_meta__ = {'factor_id': 'm', 'version': 1, 'expression': 'x'}\n"
    )


def test_whitelist_accepts_generated_snapshot() -> None:
    source = render_snapshot(
        "momentum", 1, "close/ref(close,1)-1", _BODY, "0.3.0", "2026-08-24T00:00:00+00:00"
    )
    meta = validate_snapshot_source(source)
    assert meta["factor_id"] == "momentum"
    assert meta["version"] == 1


@pytest.mark.parametrize(
    "source",
    [
        _wrapped("1", imports="import os"),
        _wrapped("1", imports="from os import system"),
        _wrapped("__import__('os')"),
        _wrapped("eval('1+1')"),
        _wrapped("open('/etc/passwd')"),
        _wrapped("(1).__class__"),
        _wrapped("ctx.__class__"),
        _wrapped("ctx('CLOSE')", imports="import numpy as np\nimport subprocess"),
    ],
)
def test_whitelist_rejects_injection(source: str) -> None:
    with pytest.raises(SnapshotValidationError):
        validate_snapshot_source(source)


def test_whitelist_rejects_extra_function_definition() -> None:
    source = (
        "import numpy as np\n\n\ndef compute(ctx):\n    return ctx('CLOSE')\n\n\n"
        "def sneaky():\n    import os\n    os.system('x')\n"
    )
    with pytest.raises(SnapshotValidationError):
        validate_snapshot_source(source)


def test_whitelist_rejects_meta_mismatch() -> None:
    source = _wrapped("ctx('CLOSE')")
    with pytest.raises(SnapshotValidationError):
        validate_snapshot_source(source, factor_id="other", version=1)
    with pytest.raises(SnapshotValidationError):
        validate_snapshot_source(source, factor_id="m", version=99)


def test_load_rejects_tampered_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(snapshot, "translate_expression", lambda expr: _BODY)
    store = _store(tmp_path)
    store.register("close", name="m")

    target = tmp_path / "factors" / "m" / "1" / "snapshot.py"
    target.chmod(0o644)  # undo the read-only immutability hint to simulate tampering
    target.write_text(
        "import os\nos.system('echo pwn')\n\n\ndef compute(ctx):\n    return 1\n",
        encoding="utf-8",
    )

    with pytest.raises(SnapshotValidationError):
        store.load("m", 1)


# --------------------------------------------------------------------------- #
# API route: POST /alpha/custom                                               #
# --------------------------------------------------------------------------- #


def _allow() -> None:
    """Stub auth dependency — always allows."""


@pytest.fixture
def _client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("VIBE_TRADING_HOME", str(tmp_path))
    snapshot.reset_snapshot_store()

    from fastapi import FastAPI

    from src.api.alpha_routes import register_alpha_routes

    app = FastAPI()
    register_alpha_routes(app, require_auth=_allow, require_event_stream_auth=_allow)
    return TestClient(app)


def test_register_custom_factor_ok(_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_alpha(monkeypatch)
    response = _client.post("/alpha/custom", json={"expression": "close", "name": "m"})

    assert response.status_code == 201
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["factor_id"] == "m"
    assert payload["version"] == 1


def test_register_custom_factor_degrades_without_alpha(
    _client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _force_unavailable(monkeypatch)
    response = _client.post("/alpha/custom", json={"expression": "close"})

    assert response.status_code == 503
    assert "Docker" in response.json()["detail"]


def test_register_custom_factor_rejects_bad_expression(
    _client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(name: str, code: str, **kwargs: Any) -> str:
        raise ValueError("bad expression")

    _install_fake_alpha(monkeypatch).lang.to_python = _boom
    response = _client.post("/alpha/custom", json={"expression": "close +"})

    assert response.status_code == 400
    assert "invalid factor expression" in response.json()["detail"]


def test_register_custom_factor_rejects_bad_name(
    _client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_fake_alpha(monkeypatch)
    response = _client.post("/alpha/custom", json={"expression": "close", "name": "../etc"})

    assert response.status_code == 422


# --------------------------------------------------------------------------- #
# Operator-surface validation (DORA-188 — F-03 review P1)                     #
# --------------------------------------------------------------------------- #


class _MinimalExecContext:
    """Minimal ExecContext surface: MA only (REF/HHV/... absent, like 0.3.0)."""

    MA = staticmethod(lambda data, periods: data)


def _install_fake_alpha_with_context(
    monkeypatch: pytest.MonkeyPatch,
    *,
    to_python: Any,
    exec_context_cls: type,
) -> types.ModuleType:
    """Install a fake ``alpha`` with a ``lang`` translator AND an ExecContext."""
    fake = _fake_alpha(to_python=to_python)
    ctx_mod = types.ModuleType("alpha.context")
    ctx_mod.ExecContext = exec_context_cls
    fake.context = ctx_mod
    fake.ExecContext = exec_context_cls
    monkeypatch.setitem(sys.modules, "alpha", fake)
    monkeypatch.setitem(sys.modules, "alpha.context", ctx_mod)
    reset_probe()
    return fake


_REF_BODY = "def compute(ctx):\n    return ctx.REF(ctx('CLOSE'), 1)\n"
_MA_BODY = "def compute(ctx):\n    return ctx.MA(ctx('CLOSE'), 5)\n"


def test_register_custom_factor_rejects_unsupported_operator(
    _client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``ref(close,1)`` → 400 with a clear "operator not supported" message
    (doc-claimed REF alias absent from the 0.3.0 ExecContext), never a 500."""
    _install_fake_alpha_with_context(
        monkeypatch, to_python=lambda *a, **k: _REF_BODY, exec_context_cls=_MinimalExecContext
    )
    response = _client.post("/alpha/custom", json={"expression": "ref(close,1)", "name": "m"})

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "unsupported factor operator" in detail
    assert "REF" in detail
    assert "not supported" in detail


def test_register_custom_factor_accepts_supported_operator(
    _client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A supported operator (MA) still registers — the surface check is not a
    blanket rejection."""
    _install_fake_alpha_with_context(
        monkeypatch, to_python=lambda *a, **k: _MA_BODY, exec_context_cls=_MinimalExecContext
    )
    response = _client.post("/alpha/custom", json={"expression": "ma(close,5)", "name": "m"})

    assert response.status_code == 201
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["factor_id"] == "m"
    assert payload["version"] == 1
