"""
Human escalation node.

When the supervisor produces a low-confidence or high-risk plan,
this node pauses execution and waits for a human decision via the API.
"""
import time
from typing import Any, Dict

from app.human_review import enqueue, get_decision, get_feedback
from app.observability.tracer import tracer

POLL_INTERVAL_SECONDS = 2
POLL_TIMEOUT_SECONDS = 300  # 5 minutes before auto-reject


def escalation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    task_id = state["task_id"]
    reason = state.get("escalation_reason", "Low confidence or high-risk plan detected.")

    enqueue(task_id, state, reason)
    tracer.node_enter(task_id, "human_escalation", input_keys=["escalation_reason"])
    tracer.escalation(task_id, reason)

    # Poll until a human resolves it or we time out
    elapsed = 0
    decision = None
    while elapsed < POLL_TIMEOUT_SECONDS:
        decision = get_decision(task_id)
        if decision is not None:
            break
        time.sleep(POLL_INTERVAL_SECONDS)
        elapsed += POLL_INTERVAL_SECONDS

    if decision is None:
        decision = "reject"
        feedback = "Auto-rejected: no human response within timeout."
    else:
        feedback = get_feedback(task_id)

    tracer.escalation(task_id, reason, decision=decision, feedback=feedback)
    tracer.node_exit(task_id, "human_escalation", output_keys=["human_decision", "human_feedback"])

    return {
        "human_decision": decision,
        "human_feedback": feedback,
        "escalation_required": False,
    }
