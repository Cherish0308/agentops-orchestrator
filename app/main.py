from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.graph.workflow import build_graph
from app.human_review import get_pending, list_pending, resolve
from app.observability.tracer import tracer
from app.schemas import TaskRequest, TaskRunResponse

app = FastAPI(
    title="AgentOps Orchestrator",
    description="Multi-agent orchestration system with supervisor, specialist agents, reviewer, human-in-the-loop escalation, and trace-ready execution.",
    version="0.2.0"
)

workflow = build_graph()


# ─── Health ────────────────────────────────────────────────────────────────────

@app.get("/")
def health_check():
    return {"status": "ok", "service": "AgentOps Orchestrator"}


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
        "human_decision": None,
        "human_feedback": "",
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


class ReviewDecision(BaseModel):
    decision: str   # "approve" or "reject"
    feedback: str = ""


@app.post("/tasks/{task_id}/review")
def submit_review(task_id: str, body: ReviewDecision):
    """
    Submit a human review decision for an escalated task.

    - decision: "approve" → the task continues to specialist execution
    - decision: "reject"  → the supervisor re-plans using your feedback
    - feedback: optional explanation (required for reject to be useful)
    """
    if body.decision not in ("approve", "reject"):
        raise HTTPException(status_code=400, detail="decision must be 'approve' or 'reject'")

    ok = resolve(task_id, body.decision, body.feedback)
    if not ok:
        raise HTTPException(status_code=404, detail=f"No pending review found for task '{task_id}'")

    return {
        "task_id": task_id,
        "decision": body.decision,
        "feedback": body.feedback,
        "status": "resolved — task will resume shortly",
    }


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
