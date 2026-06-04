from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class TaskRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    request: str = Field(..., description="Original user task")


class SubTask(BaseModel):
    id: str
    description: str
    assigned_agent: str
    depends_on: List[str] = []
    expected_output: str
    complexity: str


class ExecutionPlan(BaseModel):
    task_summary: str
    confidence_score: float
    risk_level: str
    subtasks: List[SubTask]


class AgentOutput(BaseModel):
    agent_name: str
    subtask_id: str
    output: str
    confidence_score: float
    tools_used: List[str] = []
    errors: List[str] = []


class ReviewResult(BaseModel):
    approved: bool
    quality_score: float
    feedback: str
    requires_rework: bool


class TaskRunResponse(BaseModel):
    task_id: str
    execution_plan: Dict[str, Any]
    agent_outputs: Dict[str, Any]
    review_result: Dict[str, Any]
    final_output: str
