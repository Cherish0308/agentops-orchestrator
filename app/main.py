from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.graph.workflow import build_graph
from app.human_review import (
    add_chat_message, get_chat, get_pending,
    list_all, list_pending, resolve,
)
from app.memory import long_term, short_term
from app.memory.manager import consolidate_user_memories, expire_stale_memories
from app.observability.tracer import tracer
from app.schemas import TaskRequest, TaskRunResponse

app = FastAPI(
    title="AgentOps Orchestrator",
    description="Multi-agent orchestration with tool use, persistent memory, human-in-the-loop escalation, and full observability.",
    version="0.3.0"
)

workflow = build_graph()


# ─── Health ────────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "AgentOps Orchestrator",
        "memory": {
            "short_term_redis": short_term.is_available(),
            "long_term_chromadb": long_term.is_available(),
        },
    }


# ─── Task execution ────────────────────────────────────────────────────────────

@app.post("/tasks/run", response_model=TaskRunResponse)
def run_task(task: TaskRequest):
    task_id = str(uuid4())

    initial_state = {
        "task_id": task_id,
        "user_id": task.user_id,
        "original_request": task.request,
        "execution_plan": {},
        "current_subtask_index": 0,
        "agent_outputs": {},
        "review_result": {},
        "final_output": None,
        "errors": [],
        "rework_count": 0,
        # Human-in-the-loop defaults
        "escalation_required": False,
        "escalation_reason": "",
        "approval_level": "notify",
        "human_decision": None,
        "human_feedback": "",
        "subtask_failure_counts": {},
        # Memory defaults
        "retrieved_memories": [],
        "saved_memory_ids": [],
    }

    result = workflow.invoke(initial_state)

    return {
        "task_id": task_id,
        "execution_plan": result["execution_plan"],
        "agent_outputs": result["agent_outputs"],
        "review_result": result["review_result"],
        "final_output": result["final_output"],
    }


# ─── Human-in-the-loop endpoints ───────────────────────────────────────────────

@app.get("/tasks/pending")
def list_pending_reviews():
    """List all tasks currently waiting for human review."""
    return {"pending": list_pending()}


@app.get("/tasks/{task_id}/pending")
def get_pending_review(task_id: str):
    """Get the details of a specific task awaiting human review."""
    entry = get_pending(task_id)
    if not entry:
        raise HTTPException(status_code=404, detail=f"No pending review found for task '{task_id}'")
    return entry


@app.get("/tasks/all")
def list_all_reviews():
    """List all review entries (pending + resolved)."""
    return {"reviews": list_all()}


class ReviewDecision(BaseModel):
    decision: str          # approve | reject | modify | take_over
    feedback: str = ""
    modified_plan: dict = None   # for "modify" decision
    human_output: str = None     # for "take_over" decision


@app.post("/tasks/{task_id}/review")
def submit_review(task_id: str, body: ReviewDecision):
    """
    Submit a human review decision for an escalated task.

    - approve    → task continues as planned
    - reject     → supervisor re-plans using feedback
    - modify     → reviewer edits the plan; pass modified_plan in body
    - take_over  → human provides the final output directly; pass human_output in body
    """
    valid = {"approve", "reject", "modify", "take_over"}
    if body.decision not in valid:
        raise HTTPException(status_code=400, detail=f"decision must be one of {valid}")

    ok = resolve(
        task_id,
        body.decision,
        feedback=body.feedback,
        modified_plan=body.modified_plan,
        human_output=body.human_output,
    )
    if not ok:
        raise HTTPException(status_code=404, detail=f"No pending review found for task '{task_id}'")

    return {
        "task_id": task_id,
        "decision": body.decision,
        "status": "resolved — task will resume shortly",
    }


class ChatMessage(BaseModel):
    message: str


@app.post("/tasks/{task_id}/chat")
def send_chat(task_id: str, body: ChatMessage):
    """Send a question to the AI about an escalated task. The agent will reply."""
    ok = add_chat_message(task_id, "human", body.message)
    if not ok:
        raise HTTPException(status_code=404, detail=f"No review found for task '{task_id}'")
    return {"status": "message sent — agent will reply shortly"}


@app.get("/tasks/{task_id}/chat")
def get_chat_history(task_id: str):
    """Get the full chat thread for a review."""
    return {"task_id": task_id, "chat": get_chat(task_id)}


# ─── Observability endpoints ───────────────────────────────────────────────────

@app.get("/traces")
def list_traces():
    """List all task IDs that have trace data."""
    return {"task_ids": tracer.list_task_ids()}


@app.get("/traces/{task_id}")
def get_trace(task_id: str):
    """Return the full event-by-event trace for a task run."""
    events = tracer.get_trace(task_id)
    if not events:
        raise HTTPException(status_code=404, detail=f"No trace found for task '{task_id}'")
    return {"task_id": task_id, "events": events}


@app.get("/traces/{task_id}/timeline")
def get_timeline(task_id: str):
    """Return a summarised timeline: nodes, durations, LLM call count, tool call count."""
    timeline = tracer.get_timeline(task_id)
    if not timeline.get("total_events"):
        raise HTTPException(status_code=404, detail=f"No trace found for task '{task_id}'")
    return timeline


# ─── Memory dashboard endpoints ────────────────────────────────────────────────

@app.get("/memory/{user_id}")
def get_user_memory(user_id: str):
    """Memory dashboard — show everything the system remembers about a user."""
    memories = long_term.get_user_memories(user_id)
    by_type: dict = {}
    for m in memories:
        t = m.get("type", "unknown")
        by_type.setdefault(t, []).append({
            "id": m["id"],
            "content": m["content"],
            "importance_score": m.get("importance_score"),
            "access_count": m.get("access_count"),
            "created_at": m.get("created_at"),
        })
    return {
        "user_id": user_id,
        "total_memories": len(memories),
        "by_type": by_type,
    }


@app.delete("/memory/{user_id}")
def delete_user_memory(user_id: str):
    """GDPR-style: delete ALL memories for a user."""
    count = long_term.delete_user_memories(user_id)
    return {"user_id": user_id, "deleted": count}


@app.delete("/memory/{user_id}/{memory_id}")
def delete_single_memory(user_id: str, memory_id: str):
    """Delete a single memory entry."""
    ok = long_term.delete_memory(memory_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Memory '{memory_id}' not found")
    return {"deleted": memory_id}


@app.post("/memory/{user_id}/consolidate")
def consolidate_memories(user_id: str):
    """Merge near-duplicate memories into summaries to keep the store clean."""
    removed = consolidate_user_memories(user_id)
    return {"user_id": user_id, "memories_consolidated": removed}


@app.post("/memory/{user_id}/expire")
def expire_memories(user_id: str):
    """Manually trigger expiry of stale / low-importance memories."""
    removed = expire_stale_memories(user_id)
    return {"user_id": user_id, "memories_expired": removed}
