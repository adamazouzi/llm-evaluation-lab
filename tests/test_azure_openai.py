
"""Network-free tests for the Azure OpenAI answer generator."""

from types import SimpleNamespace

import pytest

from backend.app.services.azure_openai import (
    AzureOpenAIAnswerGenerator,
)


class FakeCompletions:
    """Fake Azure chat completions endpoint."""

    def __init__(self, content: str | None = "A test answer.") -> None:
        self.content = content
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        message = SimpleNamespace(content=self.content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice])


class FakeAzureClient:
    """Fake SDK client that never performs network requests."""

    def __init__(self, content: str | None = "A test answer.") -> None:
        self.completions = FakeCompletions(content)
        self.chat = SimpleNamespace(completions=self.completions)


def make_generator(
    client: FakeAzureClient,
) -> AzureOpenAIAnswerGenerator:
    return AzureOpenAIAnswerGenerator(
        endpoint="https://example-resource.openai.azure.com",
        api_key="test-key-not-real",
        deployment="test-deployment",
        client=client,
    )


def test_generator_sends_question_and_context_to_client() -> None:
    client = FakeAzureClient("RAG retrieves relevant passages.")
    generator = make_generator(client)

    answer = generator(
        "What is RAG?",
        "RAG retrieves relevant passages from documents.",
    )

    assert answer == "RAG retrieves relevant passages."
    request = client.completions.last_kwargs
    assert request is not None
    assert request["model"] == "test-deployment"
    assert request["messages"][0]["role"] == "system"
    assert "only the supplied context" in (
        request["messages"][0]["content"]
    )
    assert request["messages"][1]["role"] == "user"
    assert "What is RAG?" in request["messages"][1]["content"]
    assert "RAG retrieves relevant passages from documents." in (
        request["messages"][1]["content"]
    )


def test_generator_rejects_empty_endpoint() -> None:
    with pytest.raises(
        ValueError,
        match="endpoint must not be empty",
    ):
        AzureOpenAIAnswerGenerator(
            endpoint=" ",
            api_key="test-key",
            deployment="test-deployment",
            client=FakeAzureClient(),
        )


def test_generator_rejects_empty_api_key() -> None:
    with pytest.raises(
        ValueError,
        match="API key must not be empty",
    ):
        AzureOpenAIAnswerGenerator(
            endpoint="https://example-resource.openai.azure.com",
            api_key=" ",
            deployment="test-deployment",
            client=FakeAzureClient(),
        )


def test_generator_rejects_empty_deployment() -> None:
    with pytest.raises(
        ValueError,
        match="deployment must not be empty",
    ):
        AzureOpenAIAnswerGenerator(
            endpoint="https://example-resource.openai.azure.com",
            api_key="test-key",
            deployment=" ",
            client=FakeAzureClient(),
        )


def test_generator_requires_environment_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY",
        "AZURE_OPENAI_DEPLOYMENT",
    ):
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(
        ValueError,
        match="Missing Azure OpenAI configuration",
    ):
        AzureOpenAIAnswerGenerator.from_environment()


def test_generator_rejects_empty_model_response() -> None:
    client = FakeAzureClient("   ")
    generator = make_generator(client)

    with pytest.raises(
        ValueError,
        match="Azure OpenAI returned an empty answer",
    ):
        generator("What is RAG?", "Some context.")
