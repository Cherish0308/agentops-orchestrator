import time
from contextvars import ContextVar
from typing import Optional

from openai import OpenAI

from app.config import OPENAI_API_KEY, LLM_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

# Set these before calling call_llm so the tracer knows which task/node we're in
current_task_id: ContextVar[Optional[str]] = ContextVar("current_task_id", default=None)
current_node: ContextVar[Optional[str]] = ContextVar("current_node", default=None)


def call_llm(system_prompt: str, user_prompt: str) -> str:
    start = time.time()

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    latency_ms = (time.time() - start) * 1000
    text = response.choices[0].message.content

    # Trace the LLM call if we're inside a tracked task
    task_id = current_task_id.get()
    node = current_node.get() or "unknown"
    if task_id:
        from app.observability.tracer import tracer
        usage = response.usage
        tracer.llm_call(
            task_id=task_id,
            node=node,
            model=LLM_MODEL,
            prompt_preview=user_prompt,
            response_preview=text,
            latency_ms=latency_ms,
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
        )

    return text
