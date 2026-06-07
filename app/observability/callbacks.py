"""
LangChain/LangGraph callback handler.

Plugs into LangGraph's callback system to automatically fire tracer
events on every node entry, exit, LLM call, and error — without
touching individual agent files.
"""
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from app.observability.tracer import tracer


class ObservabilityCallback(BaseCallbackHandler):
    def __init__(self, task_id: str):
        super().__init__()
        self.task_id = task_id
        self._current_node = "unknown"

    # ── Graph / chain events ───────────────────────────────────────────────────

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        # LangGraph wraps each node as a chain; the node name is in serialized
        node_name = (
            serialized.get("name")
            or serialized.get("id", ["unknown"])[-1]
        )
        self._current_node = node_name
        tracer.node_enter(
            task_id=self.task_id,
            node=node_name,
            input_keys=list(inputs.keys()) if isinstance(inputs, dict) else [],
        )

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ):
        tracer.node_exit(
            task_id=self.task_id,
            node=self._current_node,
            output_keys=list(outputs.keys()) if isinstance(outputs, dict) else [],
        )

    def on_chain_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ):
        tracer.node_exit(
            task_id=self.task_id,
            node=self._current_node,
            error=str(error),
        )
        tracer.error(self.task_id, self._current_node, str(error))

    # ── LLM events ────────────────────────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        **kwargs,
    ):
        # Store start time under run_id for duration calc on end
        import time
        self._llm_start_times = getattr(self, "_llm_start_times", {})
        self._llm_start_times[str(run_id)] = time.time()
        self._llm_prompts = getattr(self, "_llm_prompts", {})
        self._llm_prompts[str(run_id)] = prompts[0] if prompts else ""

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs,
    ):
        import time
        start = getattr(self, "_llm_start_times", {}).pop(str(run_id), None)
        latency_ms = (time.time() - start) * 1000 if start else 0.0
        prompt = getattr(self, "_llm_prompts", {}).pop(str(run_id), "")

        text = ""
        token_usage = {}
        if response.generations:
            first = response.generations[0]
            if first:
                text = getattr(first[0], "text", "")
        if response.llm_output:
            token_usage = response.llm_output.get("token_usage", {})

        model = serialized.get("name", "unknown") if (serialized := kwargs.get("serialized")) else "unknown"

        tracer.llm_call(
            task_id=self.task_id,
            node=self._current_node,
            model=model,
            prompt_preview=prompt,
            response_preview=text,
            latency_ms=latency_ms,
            prompt_tokens=token_usage.get("prompt_tokens"),
            completion_tokens=token_usage.get("completion_tokens"),
        )

    def on_llm_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        **kwargs,
    ):
        tracer.error(self.task_id, self._current_node, f"LLM error: {error}")

    # ── Tool events ───────────────────────────────────────────────────────────

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        **kwargs,
    ):
        import time
        self._tool_start_times = getattr(self, "_tool_start_times", {})
        self._tool_start_times[str(run_id)] = time.time()
        self._tool_inputs = getattr(self, "_tool_inputs", {})
        self._tool_inputs[str(run_id)] = input_str
        self._tool_names = getattr(self, "_tool_names", {})
        self._tool_names[str(run_id)] = serialized.get("name", "unknown")

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        **kwargs,
    ):
        import time
        start = getattr(self, "_tool_start_times", {}).pop(str(run_id), None)
        latency_ms = (time.time() - start) * 1000 if start else 0.0
        tool_name = getattr(self, "_tool_names", {}).pop(str(run_id), "unknown")
        inputs = getattr(self, "_tool_inputs", {}).pop(str(run_id), "")

        tracer.tool_call(
            task_id=self.task_id,
            node=self._current_node,
            tool_name=tool_name,
            inputs={"raw": inputs},
            output=output,
            latency_ms=latency_ms,
            success=True,
        )

    def on_tool_error(
        self,
        error: Union[Exception, KeyboardInterrupt],
        *,
        run_id: UUID,
        **kwargs,
    ):
        tool_name = getattr(self, "_tool_names", {}).pop(str(run_id), "unknown")
        tracer.tool_call(
            task_id=self.task_id,
            node=self._current_node,
            tool_name=tool_name,
            inputs={},
            output=None,
            latency_ms=0.0,
            success=False,
            error=str(error),
        )
