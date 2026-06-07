from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app import db
from app.celery_app import celery, run_task_async
from app.graph.workflow import build_graph
from app.human_review import (
    add_chat_message, get_chat, get_pending,
    list_all, list_pending, resolve,
)
from app.llm import get_routing_table
from app.memory import long_term, short_term
from app.memory.manager import consolidate_user_memories, expire_stale_memories
from app.observability.tracer import tracer
from app.schemas import TaskRequest, TaskRunResponse

app = FastAPI(
    title="AgentOps Orchestrator",
    description="Multi-agent orchestration with multi-model LLM routing, tool use, persistent memory, human-in-the-loop escalation, async task queue, and full observability.",
    version="0.4.0"
)

workflow = build_graph()


# ─── Health ────────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "AgentOps Orchestrator",
        "version": "0.4.0",
        "infrastructure": {
            "redis": short_term.is_available(),
            "chromadb": long_term.is_available(),
            "postgres": db.is_available(),
        },
        "llm_routing": get_routing_table(),
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


# ─── Async task submission (Celery) ───────────────────────────────────────────

@app.post("/tasks/submit")
def submit_task(task: TaskRequest):
    """
    Submit a task asynchronously. Returns a job_id immediately.
    The task runs in a Celery worker — poll GET /tasks/{job_id}/status for result.
    """
    task_id = str(uuid4())
    db.record_submitted(task_id, task.user_id, task.request)
    job = run_task_async.apply_async(
        args=[task_id, task.user_id, task.request],
        task_id=task_id,
    )
    return {
        "job_id": job.id,
        "task_id": task_id,
        "status": "queued",
        "poll_url": f"/tasks/{task_id}/status",
    }


@app.get("/tasks/{task_id}/status")
def task_status(task_id: str):
    """Poll for the result of an async task submission."""
    # Check Celery backend first
    job = celery.AsyncResult(task_id)
    celery_state = job.state  # PENDING | STARTED | SUCCESS | FAILURE

    # Also pull from Postgres for richer audit info
    db_record = db.get_task(task_id)

    if celery_state == "SUCCESS":
        return {"task_id": task_id, "status": "completed", "result": job.result}
    elif celery_state == "FAILURE":
        return {"task_id": task_id, "status": "failed", "error": str(job.result)}
    elif celery_state == "STARTED":
        return {"task_id": task_id, "status": "running", "db": db_record}
    else:
        return {"task_id": task_id, "status": "queued", "db": db_record}


@app.get("/users/{user_id}/tasks")
def user_task_history(user_id: str):
    """Get recent task history for a user from Postgres."""
    tasks = db.get_user_tasks(user_id)
    return {"user_id": user_id, "tasks": tasks}


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
