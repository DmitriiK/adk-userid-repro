"""Agent-level repro: call ADK with user_id and inspect OTel spans (no live LLM)."""

from __future__ import annotations

import os
from typing import Any

import pytest
from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner
from google.genai import types
from google.genai.types import Candidate, Content, GenerateContentResponse, Part
from opentelemetry import trace
from opentelemetry.instrumentation.google_genai import GoogleGenAiSdkInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.semconv._incubating.attributes.user_attributes import USER_ID

EXPECTED_USER_ID = "repro-user-123"


def _fake_response() -> GenerateContentResponse:
    return GenerateContentResponse(
        candidates=[
            Candidate(
                content=Content(role="model", parts=[Part.from_text(text="ok")]),
                finish_reason="STOP",
            )
        ]
    )


@pytest.fixture
def span_exporter(monkeypatch: pytest.MonkeyPatch) -> InMemorySpanExporter:
    """Capture spans and force the google-genai instrumentor path (Vertex-like)."""
    monkeypatch.setenv(
        "OTEL_SEMCONV_STABILITY_OPT_IN", "gen_ai_latest_experimental"
    )
    monkeypatch.setenv(
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT", "EVENT_ONLY"
    )
    # Client construction may read this even though generate_content is mocked.
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy-not-used")
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "FALSE")

    exporter = InMemorySpanExporter()
    provider = TracerProvider(
        resource=Resource.create({"service.name": "adk-userid-repro"})
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace.set_tracer_provider(provider)

    # Patch *before* instrument() so the instrumentor wraps the mock and still
    # creates generate_content spans, without calling the real Gemini API.
    async def fake_async_generate_content(
        self: Any, *, model: str, contents: Any, config: Any = None, **kwargs: Any
    ) -> GenerateContentResponse:
        return _fake_response()

    def fake_sync_generate_content(
        self: Any, *, model: str, contents: Any, config: Any = None, **kwargs: Any
    ) -> GenerateContentResponse:
        return _fake_response()

    monkeypatch.setattr(
        "google.genai.models.AsyncModels.generate_content",
        fake_async_generate_content,
    )
    monkeypatch.setattr(
        "google.genai.models.Models.generate_content",
        fake_sync_generate_content,
    )

    instrumentor = GoogleGenAiSdkInstrumentor()
    if not instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.instrument()

    yield exporter

    if instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.uninstrument()


@pytest.mark.asyncio
async def test_agent_call_with_user_id_missing_from_traces(
    span_exporter: InMemorySpanExporter,
) -> None:
    """Product expectation: runner user_id should show up on traces as user.id.

    Without knowing about OTel internals, an app developer passes user_id into
    Runner / session and expects that identity in Cloud Trace. On the Vertex
    instrumentor path it does not (google/adk-python#6361).
    """
    agent = LlmAgent(
        name="repro_agent",
        model="gemini-2.0-flash",
        instruction="Reply with exactly: ok",
    )
    runner = InMemoryRunner(agent=agent, app_name="repro_app")
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id=EXPECTED_USER_ID
    )
    message = types.Content(role="user", parts=[types.Part.from_text(text="hi")])

    final_text = ""
    async for event in runner.run_async(
        user_id=EXPECTED_USER_ID,
        session_id=session.id,
        new_message=message,
    ):
        if event.is_final_response() and event.content:
            for part in event.content.parts or []:
                if part.text:
                    final_text += part.text

    assert final_text == "ok"

    spans = span_exporter.get_finished_spans()
    assert spans, "expected OTel spans from the instrumented generate_content path"

    # Helpful dump if this starts failing for other reasons.
    by_name = {
        s.name: dict(s.attributes or {}) for s in spans
    }

    generate_spans = [s for s in spans if s.name.startswith("generate_content")]
    assert generate_spans, (
        "expected generate_content spans (instrumentor path). "
        f"got spans={list(by_name)}"
    )

    user_ids_seen = {
        (s.attributes or {}).get(USER_ID)
        for s in spans
        if (s.attributes or {}).get(USER_ID) is not None
    }

    # BUG: the user_id we passed into the agent never appears on any span.
    assert EXPECTED_USER_ID not in user_ids_seen, (
        "user.id unexpectedly present on spans — upstream may be fixed. "
        f"spans={by_name}"
    )
    assert all(
        (s.attributes or {}).get(USER_ID) != EXPECTED_USER_ID for s in generate_spans
    ), f"user.id leaked onto generate_content spans: {by_name}"
