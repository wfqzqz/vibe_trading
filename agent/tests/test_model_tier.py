"""Tests for the DeepSeek flash/pro model-tier contract (DORA-148 / P-01).

Covers the data-driven tier->model resolution, the ``LANGCHAIN_MODEL_TIER``
setting wiring, the ``build_llm`` integration, and the Web Settings API contract.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import api_server
from src.config.accessor import reset_env_config
from src.config.env_schema import EnvConfig
from src.providers import llm
from src.providers.capabilities import (
    provider_model_tiers,
    resolve_effective_model,
    resolve_model_tier,
)


# ---------------------------------------------------------------------------
# Capability resolution (pure, no external deps)
# ---------------------------------------------------------------------------


def test_deepseek_declares_flash_and_pro_tiers() -> None:
    assert provider_model_tiers("deepseek") == {
        "flash": "deepseek-v4-flash",
        "pro": "deepseek-v4-pro",
    }


def test_provider_without_tiers_returns_empty_mapping() -> None:
    assert provider_model_tiers("openai") == {}
    assert provider_model_tiers("") == {}


def test_resolve_model_tier_maps_label_to_concrete_model() -> None:
    assert resolve_model_tier("deepseek", "flash") == "deepseek-v4-flash"
    assert resolve_model_tier("deepseek", "pro") == "deepseek-v4-pro"
    # Case-insensitive, whitespace-tolerant.
    assert resolve_model_tier("deepseek", " FLASH ") == "deepseek-v4-flash"
    assert resolve_model_tier("deepseek", "PRO") == "deepseek-v4-pro"


def test_resolve_model_tier_unknown_or_unset_returns_none() -> None:
    assert resolve_model_tier("deepseek", "turbo") is None
    assert resolve_model_tier("deepseek", "") is None
    assert resolve_model_tier("openai", "pro") is None


def test_resolve_effective_model_prefers_explicit_model() -> None:
    assert (
        resolve_effective_model("deepseek", "flash", "custom-model") == "custom-model"
    )


def test_resolve_effective_model_uses_tier_when_no_explicit_model() -> None:
    assert resolve_effective_model("deepseek", "flash", "") == "deepseek-v4-flash"
    assert resolve_effective_model("deepseek", "pro", "") == "deepseek-v4-pro"
    assert resolve_effective_model("deepseek", None, "") is None


# ---------------------------------------------------------------------------
# Env schema
# ---------------------------------------------------------------------------


def test_langchain_model_tier_defaults_to_pro() -> None:
    monkeypatch_env = pytest.MonkeyPatch()
    monkeypatch_env.delenv("LANGCHAIN_MODEL_TIER", raising=False)
    cfg = EnvConfig()
    assert cfg.llm.langchain_model_tier == "pro"
    monkeypatch_env.undo()


def test_langchain_model_tier_reads_from_env() -> None:
    monkeypatch_env = pytest.MonkeyPatch()
    monkeypatch_env.setenv("LANGCHAIN_MODEL_TIER", "flash")
    cfg = EnvConfig()
    assert cfg.llm.langchain_model_tier == "flash"
    monkeypatch_env.undo()


# ---------------------------------------------------------------------------
# build_llm integration
# ---------------------------------------------------------------------------


class _RecordingLLM:
    """Stand-in for ChatOpenAIWithReasoning that records construction kwargs."""

    last_kwargs: dict | None = None

    def __init__(self, **kwargs) -> None:
        _RecordingLLM.last_kwargs = kwargs


def _build_deepseek(monkeypatch: pytest.MonkeyPatch, tier: str) -> dict:
    """Run ``build_llm`` for DeepSeek with a given tier and return its kwargs."""
    monkeypatch.setattr(llm, "ChatOpenAI", _RecordingLLM)
    monkeypatch.setattr(llm, "ChatOpenAIWithReasoning", _RecordingLLM)
    monkeypatch.setenv("LANGCHAIN_PROVIDER", "deepseek")
    monkeypatch.setenv("LANGCHAIN_MODEL_TIER", tier)
    monkeypatch.delenv("LANGCHAIN_MODEL_NAME", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")
    reset_env_config()
    llm.build_llm()
    return _RecordingLLM.last_kwargs or {}


def test_build_llm_flash_tier_resolves_to_flash_model(monkeypatch) -> None:
    kwargs = _build_deepseek(monkeypatch, "flash")
    assert kwargs["model"] == "deepseek-v4-flash"


def test_build_llm_pro_tier_resolves_to_pro_model(monkeypatch) -> None:
    kwargs = _build_deepseek(monkeypatch, "pro")
    assert kwargs["model"] == "deepseek-v4-pro"


def test_build_llm_explicit_model_wins_over_tier(monkeypatch) -> None:
    monkeypatch.setattr(llm, "ChatOpenAI", _RecordingLLM)
    monkeypatch.setattr(llm, "ChatOpenAIWithReasoning", _RecordingLLM)
    monkeypatch.setenv("LANGCHAIN_PROVIDER", "deepseek")
    monkeypatch.setenv("LANGCHAIN_MODEL_TIER", "pro")
    monkeypatch.setenv("LANGCHAIN_MODEL_NAME", "deepseek-v4-flash")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")
    reset_env_config()
    llm.build_llm()
    assert (_RecordingLLM.last_kwargs or {})["model"] == "deepseek-v4-flash"


def test_build_llm_non_tier_provider_requires_model(monkeypatch) -> None:
    """A provider without tiers still needs an explicit model name."""
    monkeypatch.setattr(llm, "ChatOpenAI", _RecordingLLM)
    monkeypatch.setattr(llm, "ChatOpenAIWithReasoning", _RecordingLLM)
    monkeypatch.setenv("LANGCHAIN_PROVIDER", "openrouter")
    monkeypatch.setenv("LANGCHAIN_MODEL_TIER", "pro")
    monkeypatch.delenv("LANGCHAIN_MODEL_NAME", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    reset_env_config()
    with pytest.raises(RuntimeError):
        llm.build_llm()


# ---------------------------------------------------------------------------
# Settings API
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    env_example = tmp_path / ".env.example"
    env_path = tmp_path / ".env"
    env_example.write_text(
        "\n".join(
            [
                "LANGCHAIN_PROVIDER=deepseek",
                "LANGCHAIN_MODEL_NAME=deepseek-v4-pro",
                "DEEPSEEK_BASE_URL=https://api.deepseek.com/v1",
                "DEEPSEEK_API_KEY=sk-deepseek-test",
                "LANGCHAIN_MODEL_TIER=pro",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(api_server, "ENV_PATH", env_path)
    monkeypatch.setattr(api_server, "LEGACY_ENV_PATH", tmp_path / "legacy" / ".env", raising=False)
    monkeypatch.setattr(api_server, "ENV_EXAMPLE_PATH", env_example)
    monkeypatch.setattr(api_server, "_baostock_supported", lambda: False)
    monkeypatch.setattr(api_server, "_baostock_installed", lambda: False)
    monkeypatch.delenv("API_AUTH_KEY", raising=False)
    return TestClient(api_server.app, client=("127.0.0.1", 50000))


def test_settings_response_includes_model_tier(client: TestClient) -> None:
    response = client.get("/settings/llm")
    assert response.status_code == 200
    body = response.json()
    assert body["provider"] == "deepseek"
    assert body["model_tier"] == "pro"
    assert body["model_name"] == "deepseek-v4-pro"


def test_settings_response_resolves_effective_model_from_tier(
    client: TestClient, tmp_path: Path,
) -> None:
    # Env-only tier switch with no explicit MODEL_NAME -> effective model is flash.
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "LANGCHAIN_PROVIDER=deepseek",
                "LANGCHAIN_MODEL_TIER=flash",
                "DEEPSEEK_BASE_URL=https://api.deepseek.com/v1",
                "DEEPSEEK_API_KEY=sk-deepseek-test",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    response = client.get("/settings/llm")
    assert response.status_code == 200
    body = response.json()
    assert body["model_tier"] == "flash"
    assert body["model_name"] == "deepseek-v4-flash"


def test_settings_provider_option_exposes_model_tiers(client: TestClient) -> None:
    body = client.get("/settings/llm").json()
    deepseek = next(p for p in body["providers"] if p["name"] == "deepseek")
    assert deepseek["model_tiers"] == {
        "flash": "deepseek-v4-flash",
        "pro": "deepseek-v4-pro",
    }


def test_update_deepseek_settings_persists_model_tier(
    client: TestClient, tmp_path: Path,
) -> None:
    response = client.put(
        "/settings/llm",
        json={
            "provider": "deepseek",
            "model_name": "deepseek-v4-flash",
            "model_tier": "flash",
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "sk-deepseek-test",
            "temperature": 0.0,
            "timeout_seconds": 120,
            "max_retries": 2,
            "reasoning_effort": "",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model_tier"] == "flash"
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "LANGCHAIN_MODEL_TIER=flash" in env_text
    assert "LANGCHAIN_MODEL_NAME=deepseek-v4-flash" in env_text


def test_update_deepseek_defaults_tier_to_pro_when_unset(
    client: TestClient, tmp_path: Path,
) -> None:
    response = client.put(
        "/settings/llm",
        json={
            "provider": "deepseek",
            "model_name": "deepseek-v4-pro",
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "sk-deepseek-test",
            "temperature": 0.0,
            "timeout_seconds": 120,
            "max_retries": 2,
            "reasoning_effort": "",
        },
    )
    assert response.status_code == 200
    assert response.json()["model_tier"] == "pro"
    assert "LANGCHAIN_MODEL_TIER=pro" in (tmp_path / ".env").read_text(encoding="utf-8")


def test_update_deepseek_rejects_unknown_tier(client: TestClient) -> None:
    response = client.put(
        "/settings/llm",
        json={
            "provider": "deepseek",
            "model_name": "deepseek-v4-pro",
            "model_tier": "turbo",
            "base_url": "https://api.deepseek.com/v1",
            "api_key": "sk-deepseek-test",
        },
    )
    assert response.status_code == 400
    assert "Model tier" in response.json()["detail"]


def test_update_non_tier_provider_clears_model_tier(
    client: TestClient, tmp_path: Path,
) -> None:
    # A stale LANGCHAIN_MODEL_TIER should be cleared when switching away from a
    # tiered provider.
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "LANGCHAIN_PROVIDER=openrouter",
                "LANGCHAIN_MODEL_NAME=deepseek/deepseek-v4-pro",
                "LANGCHAIN_MODEL_TIER=flash",
                "OPENROUTER_BASE_URL=https://openrouter.ai/api/v1",
                "OPENROUTER_API_KEY=sk-or-test",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    response = client.put(
        "/settings/llm",
        json={
            "provider": "openrouter",
            "model_name": "deepseek/deepseek-v4-pro",
            "base_url": "https://openrouter.ai/api/v1",
            "api_key": "sk-or-test",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["model_tier"] == ""
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    # Stale tier key is cleared to empty.
    assert "LANGCHAIN_MODEL_TIER=" in env_text
