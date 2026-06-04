from uuid import uuid4
from fastapi import FastAPI
from app.schemas import TaskRequest, TaskRunResponse
from app.graph.workflow import build_graph

app = FastAPI(
    title="AgentOps Orchestrator",
    description="Multi-agent orchestration system with supervisor, specialist agents, reviewer, and trace-ready execution.",
    version="0.1.0"
)

workflow = build_graph()


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "service": "AgentOps Orchestrator"
    }


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
        "errors": []
    }

    result = workflow.invoke(initial_state)

    return {
        "task_id": task_id,
        "execution_plan": result["execution_plan"],
        "agent_outputs": result["agent_outputs"],
        "review_result": result["review_result"],
        "final_output": result["final_output"]
    }
