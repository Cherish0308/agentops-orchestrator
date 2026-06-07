"""
Multi-model LLM router.

Agent → Model mapping:
  supervisor   → openai  (gpt-4o-mini)   fast planning
  reviewer     → openai  (gpt-4o-mini)   structured JSON review
  data_agent   → openai  (gpt-4o)        code + data reasoning
  research_agent → openai (gpt-4o-mini)  search + summarise
  writer_agent → anthropic (claude-3-5-haiku-20241022)  long-form writing
  synthesizer  → anthropic (claude-3-5-haiku-20241022)  final polish
  default      → openai  (gpt-4o-mini)

Falls back to OpenAI if Anthropic key is missing.
"""
import time
from contextvars import ContextVar
from typing import Optional

from openai import OpenAI

from app.config import ANTHROPIC_API_KEY, LLM_MODEL, OPENAI_API_KEY

# ── Clients ────────────────────────────────────────────────────────────────────
_openai = OpenAI(api_key=OPENAI_API_KEY)

_anthropic = None
if ANTHROPIC_API_KEY:
    try:
        import anthropic as _anthropic_sdk
        _anthropic = _anthropic_sdk.Anthropic(api_key=ANTHROPIC_API_KEY)
    except ImportError:
        _anthropic = None

# ── Routing table ──────────────────────────────────────────────────────────────
# node name → (provider, model)
ROUTING_TABLE: dict[str, tuple[str, str]] = {
    "supervisor":     ("openai",    "gpt-4o-mini"),
    "reviewer":       ("openai",    "gpt-4o-mini"),
    "data_agent":     ("openai",    "gpt-4o"),
    "research_agent": ("openai",    "gpt-4o-mini"),
    "writer_agent":   ("anthropic", "claude-3-5-haiku-20241022"),
    "synthesizer":    ("anthropic", "claude-3-5-haiku-20241022"),
}

DEFAULT_PROVIDER = "openai"
DEFAULT_MODEL = LLM_MODEL

# ── Context vars (set by each agent node so the tracer knows who's calling) ───
current_task_id: ContextVar[Optional[str]] = ContextVar("current_task_id", default=None)
current_node: ContextVar[Optional[str]] = ContextVar("current_node", default=None)


# ── Provider calls ─────────────────────────────────────────────────────────────

def _call_openai(system_prompt: str, user_prompt: str, model: str) -> tuple[str, Optional[int], Optional[int]]:
    response = _openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,
    )
    text = response.choices[0].message.content
    usage = response.usage
    return text, (usage.prompt_tokens if usage else None), (usage.completion_tokens if usage else None)


def _call_anthropic(system_prompt: str, user_prompt: str, model: str) -> tuple[str, Optional[int], Optional[int]]:
    if not _anthropic:
        # Fallback to OpenAI if Anthropic not available
        return _call_openai(system_prompt, user_prompt, DEFAULT_MODEL)

    response = _anthropic.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = response.content[0].text
    usage = response.usage
    return text, (usage.input_tokens if usage else None), (usage.output_tokens if usage else None)


# ── Public interface ───────────────────────────────────────────────────────────

def call_llm(system_prompt: str, user_prompt: str, force_provider: Optional[str] = None) -> str:
    """
    Route to the right LLM based on the current node context.
    Optionally force a specific provider with force_provider='openai'|'anthropic'.
    """
    node = current_node.get() or "default"
    provider, model = ROUTING_TABLE.get(node, (DEFAULT_PROVIDER, DEFAULT_MODEL))

    if force_provider:
        provider = force_provider

    # If Anthropic isn't set up, always fall back to OpenAI
    if provider == "anthropic" and not _anthropic:
        provider = "openai"
        model = DEFAULT_MODEL

    start = time.time()

    if provider == "anthropic":
        text, prompt_tokens, completion_tokens = _call_anthropic(system_prompt, user_prompt, model)
    else:
        text, prompt_tokens, completion_tokens = _call_openai(system_prompt, user_prompt, model)

    latency_ms = (time.time() - start) * 1000

    # Trace the call
    task_id = current_task_id.get()
    if task_id:
        from app.observability.tracer import tracer
        tracer.llm_call(
            task_id=task_id,
            node=node,
            model=f"{provider}/{model}",
            prompt_preview=user_prompt,
            response_preview=text,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    return text


def get_routing_table() -> dict:
    """Return the current routing table for observability."""
    return {
        node: {
            "provider": provider,
            "model": model,
            "available": True if provider == "openai" else bool(_anthropic),
        }
        for node, (provider, model) in ROUTING_TABLE.items()
    }
