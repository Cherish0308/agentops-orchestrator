"""
Observability tracer.

Records a structured event log for every task run:
  - node_enter / node_exit   : LangGraph node transitions
  - llm_call                 : every LLM request + response summary
  - tool_call                : every tool invocation (inputs, output, latency)
  - escalation               : human-in-the-loop events
  - error                    : any exception caught inside a node

All events are keyed by task_id so you can retrieve the full
trace for a single run via GET /traces/{task_id}.
"""
import time
from typing import Any, Dict, List, Optional

# ── Event types ────────────────────────────────────────────────────────────────
NODE_ENTER = "node_enter"
NODE_EXIT = "node_exit"
LLM_CALL = "llm_call"
TOOL_CALL = "tool_call"
ESCALATION = "escalation"
ERROR = "error"


class TraceEvent:
    def __init__(
        self,
        event_type: str,
        task_id: str,
        node: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ):
        self.event_type = event_type
        self.task_id = task_id
        self.node = node
        self.data = data or {}
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_type": self.event_type,
            "task_id": self.task_id,
            "node": self.node,
            "data": self.data,
            "timestamp": round(self.timestamp, 4),
        }


class Tracer:
    def __init__(self):
        # { task_id: [TraceEvent, ...] }
        self._traces: Dict[str, List[TraceEvent]] = {}
        # track node entry times for duration calculation
        self._node_start_times: Dict[str, float] = {}

    def _record(self, event: TraceEvent):
        self._traces.setdefault(event.task_id, []).append(event)

    # ── Public recording methods ───────────────────────────────────────────────

    def node_enter(self, task_id: str, node: str, input_keys: List[str] = []):
        key = f"{task_id}:{node}"
        self._node_start_times[key] = time.time()
        self._record(TraceEvent(
            NODE_ENTER, task_id, node=node,
            data={"input_keys": input_keys},
        ))

    def node_exit(self, task_id: str, node: str, output_keys: List[str] = [], error: Optional[str] = None):
        key = f"{task_id}:{node}"
        start = self._node_start_times.pop(key, None)
        duration_ms = round((time.time() - start) * 1000, 2) if start else None
        self._record(TraceEvent(
            NODE_EXIT, task_id, node=node,
            data={"output_keys": output_keys, "duration_ms": duration_ms, "error": error},
        ))

    def llm_call(
        self,
        task_id: str,
        node: str,
        model: str,
        prompt_preview: str,
        response_preview: str,
        latency_ms: float,
        prompt_tokens: Optional[int] = None,
        completion_tokens: Optional[int] = None,
    ):
        self._record(TraceEvent(
            LLM_CALL, task_id, node=node,
            data={
                "model": model,
                "prompt_preview": prompt_preview[:300],
                "response_preview": response_preview[:300],
                "latency_ms": round(latency_ms, 2),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            },
        ))

    def tool_call(
        self,
        task_id: str,
        node: str,
        tool_name: str,
        inputs: Dict[str, Any],
        output: Any,
        latency_ms: float,
        success: bool,
        error: Optional[str] = None,
    ):
        self._record(TraceEvent(
            TOOL_CALL, task_id, node=node,
            data={
                "tool_name": tool_name,
                "inputs": inputs,
                "output": output,
                "latency_ms": round(latency_ms, 2),
                "success": success,
                "error": error,
            },
        ))

    def escalation(self, task_id: str, reason: str, decision: Optional[str] = None, feedback: str = ""):
        self._record(TraceEvent(
            ESCALATION, task_id, node="human_escalation",
            data={"reason": reason, "decision": decision, "feedback": feedback},
        ))

    def error(self, task_id: str, node: str, message: str):
        self._record(TraceEvent(ERROR, task_id, node=node, data={"message": message}))

    # ── Query methods ──────────────────────────────────────────────────────────

    def get_trace(self, task_id: str) -> List[Dict[str, Any]]:
        return [e.to_dict() for e in self._traces.get(task_id, [])]

    def get_timeline(self, task_id: str) -> Dict[str, Any]:
        """Return a summary grouped by node with durations and event counts."""
        events = self._traces.get(task_id, [])
        if not events:
            return {"task_id": task_id, "nodes": [], "total_events": 0}

        nodes_seen = {}
        for e in events:
            if e.node and e.node not in nodes_seen:
                nodes_seen[e.node] = {"node": e.node, "events": 0, "duration_ms": None, "error": None}
            if e.node:
                nodes_seen[e.node]["events"] += 1
            if e.event_type == NODE_EXIT and e.node:
                nodes_seen[e.node]["duration_ms"] = e.data.get("duration_ms")
                nodes_seen[e.node]["error"] = e.data.get("error")

        total_ms = sum(
            n["duration_ms"] for n in nodes_seen.values() if n["duration_ms"] is not None
        )

        llm_calls = [e for e in events if e.event_type == LLM_CALL]
        tool_calls = [e for e in events if e.event_type == TOOL_CALL]

        return {
            "task_id": task_id,
            "total_events": len(events),
            "total_duration_ms": round(total_ms, 2),
            "llm_calls": len(llm_calls),
            "tool_calls": len(tool_calls),
            "nodes": list(nodes_seen.values()),
            "start_time": events[0].timestamp if events else None,
            "end_time": events[-1].timestamp if events else None,
        }

    def list_task_ids(self) -> List[str]:
        return list(self._traces.keys())


# Singleton — imported everywhere
tracer = Tracer()
