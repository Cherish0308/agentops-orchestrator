"""
Multi-model LLM router — powered by NVIDIA NIM (free API).

NVIDIA NIM is OpenAI-compatible, so we use the OpenAI SDK
with a different base_url. Falls back to OpenAI/Anthropic
if NVIDIA key is not set.

Agent → Model mapping (all via NVIDIA NIM free tier):
  supervisor     → meta/llama-3.1-70b-instruct   (strong reasoning, planning)
  reviewer       → meta/llama-3.1-70b-instruct   (structured JSON review)
  data_agent     → nvidia/llama-3.1-nemotron-70b-instruct  (NVIDIA-tuned, great at code/data)
  research_agent → meta/llama-3.1-8b-instruct    (fast, good at summarising)
  writer_agent   → mistralai/mixtral-8x22b-instruct-v0.1   (excellent writing)
  synthesizer    → mistralai/mixtral-8x22b-instruct-v0.1   (polished long-form output)
  default        → meta/llama-3.1-70b-instruct
"""
import time
from contextvars import ContextVar
from typing import Optional

from openai import OpenAI

from app.config import (
    ANTHROPIC_API_KEY,
    NVIDIA_API_KEY,
    NVIDIA_BASE_URL,
    OPENAI_API_KEY,
    LLM_MODEL,
)

# ── Clients ────────────────────────────────────────────────────────────────────

# NVIDIA NIM client (OpenAI SDK, different base_url)
_nvidia = None
if NVIDIA_API_KEY:
    _nvidia = OpenAI(
        api_key=NVIDIA_API_KEY,
        base_url=NVIDIA_BASE_URL,
    )

# Fallback clients
_openai = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

_anthropic = None
if ANTHROPIC_API_KEY:
    try:
        import anthropic as _anthropic_sdk
        _anthropic = _anthropic_sdk.Anthropic(api_key=ANTHROPIC_API_KEY)
    except ImportError:
        pass

# ── Routing table ──────────────────────────────────────────────────────────────
# node → (provider, model)
ROUTING_TABLE: dict[str, tuple[str, str]] = {
    "supervisor":     ("nvidia", "meta/llama-3.3-70b-instruct"),       # latest llama, best at planning
    "reviewer":       ("nvidia", "meta/llama-3.3-70b-instruct"),       # structured JSON review
    "data_agent":     ("nvidia", "meta/llama-3.1-70b-instruct"),        # stable, good at code + data
    "research_agent": ("nvidia", "meta/llama-3.1-8b-instruct"),        # fast, good at summarising
    "writer_agent":   ("nvidia", "mistralai/mistral-nemotron"),        # excellent writing quality
    "synthesizer":    ("nvidia", "mistralai/mixtral-8x7b-instruct-v0.1"),  # MoE model, great synthesis
}

DEFAULT_PROVIDER = "nvidia" if NVIDIA_API_KEY else ("openai" if OPENAI_API_KEY else "anthropic")
DEFAULT_MODEL    = LLM_MODEL

# ── Context vars ───────────────────────────────────────────────────────────────
current_task_id: ContextVar[Optional[str]] = ContextVar("current_task_id", default=None)
current_node:    ContextVar[Optional[str]] = ContextVar("current_node",    default=None)


# ── Provider calls ─────────────────────────────────────────────────────────────

def _call_openai_compatible(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
    model: str,
) -> tuple[str, Optional[int], Optional[int]]:
    """Single function handles both NVIDIA NIM and OpenAI — same SDK, same format."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=4096,
    )
    text = response.choices[0].message.content
    usage = response.usage
    return (
        text,
        usage.prompt_tokens     if usage else None,
        usage.completion_tokens if usage else None,
    )


def _call_anthropic(
    system_prompt: str,
    user_prompt: str,
    model: str,
) -> tuple[str, Optional[int], Optional[int]]:
    response = _anthropic.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = response.content[0].text
    usage = response.usage
    return (
        text,
        usage.input_tokens  if usage else None,
        usage.output_tokens if usage else None,
    )


def _resolve_client_and_model(
    provider: str,
    model: str,
) -> tuple:
    """
    Returns (client_or_flag, model, provider_label).
    If the preferred provider is unavailable, cascades:
    nvidia → openai → anthropic → error.
    """
    if provider == "nvidia" and _nvidia:
        return _nvidia, model, "nvidia"
    if provider == "openai" and _openai:
        return _openai, model, "openai"
    if provider == "anthropic" and _anthropic:
        return None, model, "anthropic"   # handled separately

    # Cascade fallbacks
    if _nvidia:
        return _nvidia, "meta/llama-3.1-70b-instruct", "nvidia"
    if _openai:
        return _openai, "gpt-4o-mini", "openai"
    if _anthropic:
        return None, "claude-3-5-haiku-20241022", "anthropic"

    raise RuntimeError("No LLM provider configured. Set NVIDIA_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY.")


# ── Public interface ───────────────────────────────────────────────────────────

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Route to the right LLM based on the current agent node.
    Priority: NVIDIA NIM → OpenAI → Anthropic
    """
    node = current_node.get() or "default"
    provider, model = ROUTING_TABLE.get(node, (DEFAULT_PROVIDER, DEFAULT_MODEL))

    client, resolved_model, resolved_provider = _resolve_client_and_model(provider, model)

    start = time.time()

    if resolved_provider == "anthropic":
        text, prompt_tokens, completion_tokens = _call_anthropic(system_prompt, user_prompt, resolved_model)
    else:
        text, prompt_tokens, completion_tokens = _call_openai_compatible(client, system_prompt, user_prompt, resolved_model)

    latency_ms = (time.time() - start) * 1000

    # Trace the call
    task_id = current_task_id.get()
    if task_id:
        from app.observability.tracer import tracer
        tracer.llm_call(
            task_id=task_id,
            node=node,
            model=f"{resolved_provider}/{resolved_model}",
            prompt_preview=user_prompt,
            response_preview=text,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    return text


def get_routing_table() -> dict:
    """Return the active routing table — shown on GET /"""
    active_provider = "nvidia" if _nvidia else ("openai" if _openai else "anthropic")
    return {
        node: {
            "provider": provider,
            "model": model,
            "active": active_provider,
        }
        for node, (provider, model) in ROUTING_TABLE.items()
    }
