# ADK `user.id` silent drop — minimal repro

Reproduces [google/adk-python#6361](https://github.com/google/adk-python/issues/6361):
when `opentelemetry-instrumentation-google-genai` is installed, ADK collects
`session.user_id` into `log_only_extra_attributes` but never attaches `user.id`
to telemetry, because it depends on
`GENERATE_CONTENT_EVENT_ONLY_EXTRA_ATTRIBUTES_CONTEXT_KEY` — a constant that
the instrumentation package does **not** export. ADK catches the import error
and continues with `pass`.

Pinned to **`google-adk==2.4.0`** (as requested by maintainers) with
`opentelemetry-instrumentation-google-genai==0.7b1` (compatible with ADK 2.4’s
`opentelemetry-api` pin; same missing EVENT_ONLY export as `1.0b0`).

**No live Gemini / Vertex calls** — tests are fully in-process.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install "google-adk==2.4.0" "opentelemetry-sdk" \
  "opentelemetry-instrumentation-google-genai==0.7b1" \
  "pytest" "pytest-asyncio"
pytest -q
```

Or with `uv`:

```bash
uv venv && uv pip install --python .venv/bin/python \
  "google-adk==2.4.0" "opentelemetry-sdk" \
  "opentelemetry-instrumentation-google-genai==0.7b1" \
  "pytest" "pytest-asyncio"
.venv/bin/pytest -q
```

Expected: **4 passed** (tests assert the buggy behavior is present).

## What the tests show

| Test | Assertion |
|---|---|
| `test_event_only_context_key_is_missing` | Instrumentation does not export the EVENT_ONLY key ADK imports |
| `test_import_error_is_swallowed_without_warning_path` | `from … import GENERATE_CONTENT_EVENT_ONLY_…` raises |
| `test_user_id_silently_dropped_from_otel_context` | After `_use_extra_generate_content_attributes(..., log_only={user.id: …})`, span extras context has **no** `user.id` |
| `test_agent_call_with_user_id_missing_from_traces` | **Agent-level:** `InMemoryRunner.run_async(user_id=…)` with a mocked Gemini call; captured OTel spans never contain that `user.id` |

The agent-level test is the product view: pass `user_id` into the runner and expect it on traces, without caring about OTel context keys. Gemini HTTP is mocked; `GoogleGenAiSdkInstrumentor` still runs so spans are created (Vertex-like path).

That mirrors the Vertex Agent Engine instrumentor path:
`use_inference_span` → `_use_extra_generate_content_attributes` → silent drop.

## Related

- Issue: https://github.com/google/adk-python/issues/6361
- PR: https://github.com/google/adk-python/pull/6362
