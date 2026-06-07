"""
Human escalation node.

Handles all four approval levels:

  notify         → enqueue for logging, mark notified immediately, continue
  approve_action → pause, wait for human decision on the specific next action
  approve_plan   → pause, wait for human to approve / reject / modify full plan
  take_over      → pause, wait for human to directly supply the final output
"""
import time
from typing import Any, Dict

from app.human_review import (
    APPROVE_ACTION,
    APPROVE_PLAN,
    NOTIFY,
    TAKE_OVER,
    add_chat_message,
    enqueue,
    get_decision,
    get_feedback,
    get_human_output,
    get_modified_plan,
    mark_notified,
)
from app.llm import call_llm
from app.observability.tracer import tracer

POLL_INTERVAL_SECONDS = 2
POLL_TIMEOUT_SECONDS = 600   # 10 minutes


def _agent_answer_chat(task_id: str, state: Dict[str, Any]):
    """
    If the human asked a question in the chat, have the LLM answer it.
    Called once per poll cycle when new human messages arrive.
    """
    from app.human_review import get_chat
    messages = get_chat(task_id)
    unanswered = [m for m in messages if m["role"] == "human" and not m.get("answered")]
    if not unanswered:
        return

    last = unanswered[-1]
    context = f"""
Task: {state.get("original_request")}
Plan summary: {state.get("execution_plan", {}).get("task_summary", "")}
Human question: {last["message"]}
Answer concisely as the AI orchestrator.
"""
    answer = call_llm("You are an AI orchestrator answering a reviewer's question.", context)
    last["answered"] = True
    add_chat_message(task_id, "agent", answer)


def escalation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    task_id = state["task_id"]
    approval_level = state.get("approval_level", APPROVE_PLAN)
    reason = state.get("escalation_reason", "Escalation triggered.")

    tracer.node_enter(task_id, "human_escalation", input_keys=["escalation_reason", "approval_level"])
    tracer.escalation(task_id, reason)

    # ── NOTIFY: just log it and move on immediately ──────────────────────────
    if approval_level == NOTIFY:
        enqueue(task_id, state, reason, approval_level=NOTIFY)
        mark_notified(task_id)
        tracer.node_exit(task_id, "human_escalation", output_keys=["human_decision"])
        return {
            "human_decision": "approve",
            "human_feedback": f"[notify] {reason}",
            "escalation_required": False,
        }

    # ── APPROVE_ACTION / APPROVE_PLAN / TAKE_OVER: pause and wait ───────────
    enqueue(
        task_id,
        state,
        reason,
        approval_level=approval_level,
        proposed_action=_proposed_action_text(state),
    )

    elapsed = 0
    decision = None

    while elapsed < POLL_TIMEOUT_SECONDS:
        decision = get_decision(task_id)
        if decision is not None:
            break
        _agent_answer_chat(task_id, state)
        time.sleep(POLL_INTERVAL_SECONDS)
        elapsed += POLL_INTERVAL_SECONDS

    if decision is None:
        decision = "reject"
        feedback = "Auto-rejected: no human response within timeout."
        human_output = None
        modified_plan = None
    else:
        feedback = get_feedback(task_id)
        human_output = get_human_output(task_id)
        modified_plan = get_modified_plan(task_id)

    tracer.escalation(task_id, reason, decision=decision, feedback=feedback)
    tracer.node_exit(task_id, "human_escalation", output_keys=["human_decision", "human_feedback"])

    result: Dict[str, Any] = {
        "human_decision": decision,
        "human_feedback": feedback,
        "escalation_required": False,
    }

    # If reviewer modified the plan, swap it in
    if decision == "modify" and modified_plan:
        result["execution_plan"] = modified_plan
        result["current_subtask_index"] = 0
        result["human_decision"] = "approve"   # treat as approved after modification

    # If take_over, set final_output directly
    if decision == "take_over" and human_output:
        result["final_output"] = human_output
        result["human_decision"] = "take_over"

    return result


def _proposed_action_text(state: Dict[str, Any]) -> str:
    subtasks = state.get("execution_plan", {}).get("subtasks", [])
    idx = state.get("current_subtask_index", 0)
    if idx < len(subtasks):
        st = subtasks[idx]
        return (
            f"Agent: {st.get('assigned_agent')} | "
            f"Task: {st.get('description')} | "
            f"Expected output: {st.get('expected_output')}"
        )
    plan = state.get("execution_plan", {})
    return f"Execute plan: {plan.get('task_summary', 'N/A')}"
