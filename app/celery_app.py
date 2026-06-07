"""
Celery async task queue.

Flow:
  POST /tasks/submit   → enqueues a Celery task, returns job_id immediately
  GET  /tasks/{id}/status → polls Celery for result

Workers run the LangGraph workflow in the background so the HTTP
request returns instantly instead of blocking for minutes.
"""
from celery import Celery
from app.config import REDIS_URL

celery = Celery(
    "agentops",
    broker=REDIS_URL,
    backend=REDIS_URL,
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=3600,          # results kept for 1 hour
    worker_prefetch_multiplier=1, # one task at a time per worker (agents are heavy)
    task_acks_late=True,          # re-queue on worker crash
)


@celery.task(bind=True, name="agentops.run_task")
def run_task_async(self, task_id: str, user_id: str, request: str):
    """
    Celery task that runs the full LangGraph workflow.
    Result stored in Redis backend, retrievable by task_id (= Celery job id).
    """
    from app.graph.workflow import build_graph
    from app import db

    db.record_started(task_id)

    initial_state = {
        "task_id": task_id,
        "user_id": user_id,
        "original_request": request,
        "execution_plan": {},
        "current_subtask_index": 0,
        "agent_outputs": {},
        "review_result": {},
        "final_output": None,
        "errors": [],
        "rework_count": 0,
        "escalation_required": False,
        "escalation_reason": "",
        "approval_level": "notify",
        "human_decision": None,
        "human_feedback": "",
        "subtask_failure_counts": {},
        "retrieved_memories": [],
        "saved_memory_ids": [],
    }

    try:
        workflow = build_graph()
        result = workflow.invoke(initial_state)

        output = {
            "task_id": task_id,
            "status": "completed",
            "execution_plan": result.get("execution_plan", {}),
            "agent_outputs": result.get("agent_outputs", {}),
            "review_result": result.get("review_result", {}),
            "final_output": result.get("final_output", ""),
            "errors": result.get("errors", []),
        }

        db.record_completed(
            task_id,
            execution_plan=result.get("execution_plan", {}),
            final_output=result.get("final_output", ""),
            errors=result.get("errors", []),
        )

        return output

    except Exception as exc:
        db.record_failed(task_id, str(exc))
        raise self.retry(exc=exc, countdown=5, max_retries=1)
