"""Minimal repro for google/adk-python#6361 (no live LLM calls)."""

from __future__ import annotations

import opentelemetry.instrumentation.google_genai as google_genai_instr
import pytest
from google.adk.telemetry.tracing import _use_extra_generate_content_attributes
from opentelemetry import context as otel_context
from opentelemetry.instrumentation.google_genai import (
    GENERATE_CONTENT_EXTRA_ATTRIBUTES_CONTEXT_KEY,
)
from opentelemetry.semconv._incubating.attributes.user_attributes import USER_ID


EXPECTED_USER_ID = "repro-user-123"


def test_event_only_context_key_is_missing():
    """ADK imports this name; instrumentation does not export it (through 1.0b0)."""
    assert not hasattr(
        google_genai_instr,
        "GENERATE_CONTENT_EVENT_ONLY_EXTRA_ATTRIBUTES_CONTEXT_KEY",
    ), (
        "EVENT_ONLY key is now exported — update this repro / close #6361 if ADK "
        "propagates user.id successfully"
    )
    assert hasattr(
        google_genai_instr,
        "GENERATE_CONTENT_EXTRA_ATTRIBUTES_CONTEXT_KEY",
    )


def test_user_id_silently_dropped_from_otel_context():
    """Mirror the Vertex/instrumentor path inside ADK telemetry.

    google.adk.telemetry.tracing.use_inference_span() puts session.user_id into
    log_only_extra_attributes and calls _use_extra_generate_content_attributes.
    When EVENT_ONLY import fails, ADK does `except: pass` and never attaches
    user.id to any OTel context the instrumentor reads.
    """
    common = {
        "gen_ai.agent.name": "repro_agent",
        "gen_ai.conversation.id": "session-1",
    }
    log_only = {USER_ID: EXPECTED_USER_ID}

    with _use_extra_generate_content_attributes(
        common,
        log_only_extra_attributes=log_only,
    ):
        span_extras = otel_context.get_value(
            GENERATE_CONTENT_EXTRA_ATTRIBUTES_CONTEXT_KEY
        )
        assert span_extras is not None
        assert span_extras.get("gen_ai.agent.name") == "repro_agent"
        # BUG: user.id was passed as log_only but never landed in OTel context.
        assert USER_ID not in span_extras, (
            "user.id unexpectedly present on span extras — upstream may be fixed"
        )
        assert span_extras.get(USER_ID) != EXPECTED_USER_ID


def test_import_error_is_swallowed_without_warning_path():
    """Document the silent failure: importing EVENT_ONLY raises; ADK catches it."""
    with pytest.raises((ImportError, AttributeError)):
        from opentelemetry.instrumentation.google_genai import (  # noqa: F401
            GENERATE_CONTENT_EVENT_ONLY_EXTRA_ATTRIBUTES_CONTEXT_KEY,
        )
