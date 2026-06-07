"""
Human-in-the-Loop review queue.

Approval levels
───────────────
notify          Proceed immediately; human receives a notification only.
approve_action  Human must confirm the next specific action before it runs.
approve_plan    Human must approve the full execution plan before any work starts.
take_over       Human directly provides the output; all agents stand down.

Escalation triggers
───────────────────
- confidence_score below threshold          → approve_plan
- risk_level == "high"                      → approve_plan
- specialist failed twice on same subtask   → approve_action
- sensitive operation detected              → approve_action
- reviewer quality_score below threshold    → approve_action
- user explicitly requested review          → approve_plan
"""
import time
import uuid
from typing import Any, Dict, List, Optional

# ── Approval level constants ───────────────────────────────────────────────────
NOTIFY = "notify"
APPROVE_ACTION = "approve_action"
APPROVE_PLAN = "approve_plan"
TAKE_OVER = "take_over"

APPROVAL_LEVELS = {NOTIFY, APPROVE_ACTION, APPROVE_PLAN, TAKE_OVER}

# ── Sensitive operation keywords ───────────────────────────────────────────────
SENSITIVE_KEYWORDS = {
    "delete", "drop", "truncate", "remove all", "erase",
    "transfer", "payment", "send money", "wire", "transaction",
    "email", "send message", "post to", "publish", "deploy",
    "credentials", "password", "secret", "api key", "token",
}

# ── In-memory queue ────────────────────────────────────────────────────────────
# { task_id: ReviewEntry }
_queue: Dict[str, Dict[str, Any]] = {}


# ── Trigger detection ──────────────────────────────────────────────────────────

def detect_sensitive_operations(text: str) -> List[str]:
    """Return list of sensitive keywords found in text."""
    lower = text.lower()
    return [kw for kw in SENSITIVE_KEYWORDS if kw in lower]


def determine_approval_level(
    confidence_score: float = 1.0,
    risk_level: str = "low",
    specialist_failure_count: int = 0,
    quality_score: float = 1.0,
    sensitive_ops: List[str] = [],
    user_requested: bool = False,
    confidence_threshold: float = 0.6,
    quality_threshold: float = 0.5,
) -> tuple[str, str]:
    """
    Returns (approval_level, reason) based on escalation triggers.
    Higher-severity triggers win.
    """
    if user_requested:
        return APPROVE_PLAN, "User explicitly requested human review."

    if risk_level == "high" or confidence_score < confidence_threshold:
        reason = (
            f"Risk level is '{risk_level}'."
            if risk_level == "high"
            else f"Confidence score {confidence_score:.2f} is below threshold {confidence_threshold}."
        )
        return APPROVE_PLAN, reason

    if sensitive_ops:
        return APPROVE_ACTION, f"Sensitive operations detected: {', '.join(sensitive_ops)}."

    if specialist_failure_count >= 2:
        return APPROVE_ACTION, f"Specialist failed {specialist_failure_count} times on the same subtask."

    if quality_score < quality_threshold:
        return APPROVE_ACTION, f"Reviewer quality score {quality_score:.2f} is below threshold {quality_threshold}."

    # risk_level == "medium" → notify only
    if risk_level == "medium":
        return NOTIFY, f"Risk level is '{risk_level}' — proceeding with notification."

    return NOTIFY, "Low-level escalation — proceeding with notification."


# ── Queue management ───────────────────────────────────────────────────────────

def enqueue(
    task_id: str,
    state: Dict[str, Any],
    reason: str,
    approval_level: str = APPROVE_PLAN,
    current_step: Optional[str] = None,
    proposed_action: Optional[str] = None,
) -> str:
    """
    Package full context and push to review queue.
    Returns the review_id.
    """
    review_id = str(uuid.uuid4())
    plan = state.get("execution_plan", {})

    _queue[task_id] = {
        "review_id": review_id,
        "task_id": task_id,
        "approval_level": approval_level,
        "reason": reason,
        "status": "pending",          # pending | decided | notified
        "created_at": time.time(),

        # Full context for the reviewer
        "original_request": state.get("original_request"),
        "execution_plan": plan,
        "completed_steps": list(state.get("agent_outputs", {}).keys()),
        "current_step": current_step or _current_subtask_description(state),
        "proposed_action": proposed_action or "",
        "confidence_score": plan.get("confidence_score"),
        "risk_level": plan.get("risk_level"),

        # Decision fields (filled by reviewer)
        "decision": None,           # approve | reject | modify | take_over
        "feedback": "",
        "modified_plan": None,      # if reviewer modifies the plan
        "human_output": None,       # if take_over: human provides the final answer

        # Chat thread
        "chat": [],                 # [ {role: human|agent, message: str, ts: float} ]
    }
    return review_id


def _current_subtask_description(state: Dict[str, Any]) -> str:
    subtasks = state.get("execution_plan", {}).get("subtasks", [])
    idx = state.get("current_subtask_index", 0)
    if idx < len(subtasks):
        return subtasks[idx].get("description", "")
    return ""


# ── Decision submission ────────────────────────────────────────────────────────

def resolve(
    task_id: str,
    decision: str,
    feedback: str = "",
    modified_plan: Optional[Dict] = None,
    human_output: Optional[str] = None,
) -> bool:
    """
    decision: "approve" | "reject" | "modify" | "take_over"
    """
    entry = _queue.get(task_id)
    if not entry or entry["status"] != "pending":
        return False
    entry["decision"] = decision
    entry["feedback"] = feedback
    entry["modified_plan"] = modified_plan
    entry["human_output"] = human_output
    entry["status"] = "decided"
    entry["decided_at"] = time.time()
    return True


def mark_notified(task_id: str):
    entry = _queue.get(task_id)
    if entry:
        entry["status"] = "notified"


# ── Chat ───────────────────────────────────────────────────────────────────────

def add_chat_message(task_id: str, role: str, message: str) -> bool:
    entry = _queue.get(task_id)
    if not entry:
        return False
    entry["chat"].append({"role": role, "message": message, "ts": time.time()})
    return True


def get_chat(task_id: str) -> List[Dict[str, Any]]:
    entry = _queue.get(task_id)
    return entry["chat"] if entry else []


# ── Reads ──────────────────────────────────────────────────────────────────────

def get_pending(task_id: str) -> Optional[Dict[str, Any]]:
    return _queue.get(task_id)


def list_pending() -> List[Dict[str, Any]]:
    return [v for v in _queue.values() if v["status"] == "pending"]


def list_all() -> List[Dict[str, Any]]:
    return list(_queue.values())


def get_decision(task_id: str) -> Optional[str]:
    entry = _queue.get(task_id)
    return entry["decision"] if entry else None


def get_feedback(task_id: str) -> str:
    entry = _queue.get(task_id)
    return entry.get("feedback", "") if entry else ""


def get_human_output(task_id: str) -> Optional[str]:
    entry = _queue.get(task_id)
    return entry.get("human_output") if entry else None


def get_modified_plan(task_id: str) -> Optional[Dict]:
    entry = _queue.get(task_id)
    return entry.get("modified_plan") if entry else None
