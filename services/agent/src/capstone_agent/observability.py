"""Harness: Observability. LangSmith tracing of graph nodes, tool calls and model calls.

Off unless LANGSMITH_TRACING=true and LANGSMITH_API_KEY are both set. Traces leave the local network, so
compose hides model inputs and outputs by default (LANGSMITH_HIDE_INPUTS, LANGSMITH_HIDE_OUTPUTS): the trace
keeps the node path, latency, token counts and errors without the ad text, evidence or tile images.
Each run carries the review identifiers as metadata and its run id is returned as the result's traceId,
or as the X-Trace-Id header when the run fails, so a failed job can be found in LangSmith.
"""
import os
from uuid import uuid4


def tracing_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").strip().lower() == "true" and bool(os.getenv("LANGSMITH_API_KEY", "").strip())


def traced_client(client):
    """Record model name, latency and token usage of every OpenAI call when tracing is on."""
    if not tracing_enabled():
        return client
    from langsmith.wrappers import wrap_openai
    return wrap_openai(client)


def run_config(thread_id: str, *, run_name: str, review_id: str, tenant_id: str, version_no: int) -> dict:
    config = {"configurable": {"thread_id": thread_id}}
    if tracing_enabled():
        config.update(run_id=uuid4(), run_name=run_name, tags=["review", run_name],
                      metadata={"review_id": review_id, "tenant_id": tenant_id, "version_no": version_no})
    return config


def trace_id(config: dict) -> str | None:
    return str(config["run_id"]) if "run_id" in config else None


def trace_headers(config: dict) -> dict | None:
    return {"X-Trace-Id": trace_id(config)} if trace_id(config) else None
