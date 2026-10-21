
"""Tests for selecting the configured LLM provider."""

import pytest

from backend.app.main import (
    AzureOpenAIAnswerGenerator,
    create_rag_pipeline,
)


def test_demo_provider_is_selected_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Demo mode should be the default when no provider is configured."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    pipeline = create_rag_pipeline()

    assert pipeline.answer_generator.__name__ == "demo_answer_generator"


def test_demo_provider_can_be_selected_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Demo mode should work when selected explicitly."""
    monkeypatch.setenv("LLM_PROVIDER", "demo")

    pipeline = create_rag_pipeline()

    assert pipeline.answer_generator.__name__ == "demo_answer_generator"


def test_azure_provider_requires_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Azure mode should fail clearly when credentials are missing."""
    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)

    with pytest.raises(
        ValueError,
        match="Missing Azure OpenAI configuration",
    ):
        create_rag_pipeline()


def test_unknown_provider_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported providers should be rejected."""
    monkeypatch.setenv("LLM_PROVIDER", "unknown")

    with pytest.raises(
        ValueError,
        match="Unsupported LLM_PROVIDER",
    ):
        create_rag_pipeline()


def test_azure_provider_uses_azure_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Azure selection should use the generator without a network call."""
    def fake_generator(question: str, context: str) -> str:
        return "Mock Azure answer"

    monkeypatch.setenv("LLM_PROVIDER", "azure")
    monkeypatch.setattr(
        AzureOpenAIAnswerGenerator,
        "from_environment",
        classmethod(lambda cls: fake_generator),
    )

    pipeline = create_rag_pipeline()

    assert pipeline.answer_generator is fake_generator


def test_azure_generator_is_not_created_in_demo_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Demo mode must not initialize the Azure client."""
    def unexpected_azure_initialization():
        raise AssertionError(
            "Azure should not be initialized in demo mode"
        )

    monkeypatch.setenv("LLM_PROVIDER", "demo")
    monkeypatch.setattr(
        AzureOpenAIAnswerGenerator,
        "from_environment",
        unexpected_azure_initialization,
    )

    pipeline = create_rag_pipeline()

    assert pipeline.answer_generator.__name__ == "demo_answer_generator"
