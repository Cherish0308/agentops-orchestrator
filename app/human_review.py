"""
Human-in-the-Loop store.

Pending escalations are held in memory (keyed by task_id).
In production this would be backed by Redis or a DB table.
"""
from typing import Any, Dict, Optional

# { task_id: { "state": AgentState, "reason": str, "decision": None | "approve" | "reject", "feedback": str } }
_pending: Dict[str, Dict[str, Any]] = {}


def enqueue(task_id: str, state: Dict[str, Any], reason: str):
    _pending[task_id] = {
        "task_id": task_id,
        "reason": reason,
        "original_request": state.get("original_request"),
        "execution_plan": state.get("execution_plan"),
        "agent_outputs": state.get("agent_outputs"),
        "confidence_score": state.get("execution_plan", {}).get("confidence_score"),
        "risk_level": state.get("execution_plan", {}).get("risk_level"),
        "decision": None,
        "feedback": "",
    }


def get_pending(task_id: str) -> Optional[Dict[str, Any]]:
    return _pending.get(task_id)


def list_pending() -> list:
    return [v for v in _pending.values() if v["decision"] is None]


def resolve(task_id: str, decision: str, feedback: str = "") -> bool:
    """decision must be 'approve' or 'reject'."""
    if task_id not in _pending:
        return False
    _pending[task_id]["decision"] = decision
    _pending[task_id]["feedback"] = feedback
    return True


def get_decision(task_id: str) -> Optional[str]:
    entry = _pending.get(task_id)
    return entry["decision"] if entry else None


def get_feedback(task_id: str) -> str:
    entry = _pending.get(task_id)
    return entry.get("feedback", "") if entry else ""
