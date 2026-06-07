from typing import Annotated, Any, Dict, List, Optional, TypedDict


def append_lists(left: List[Dict[str, Any]], right: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return left + right


class AgentState(TypedDict):
    task_id: str
    user_id: str
    original_request: str

    execution_plan: Dict[str, Any]
    current_subtask_index: int

    agent_outputs: Dict[str, Any]
    review_result: Dict[str, Any]

    final_output: Optional[str]
    errors: Annotated[List[Dict[str, Any]], append_lists]
    rework_count: int

    # Human-in-the-loop fields
    escalation_required: bool
    escalation_reason: str
    human_decision: Optional[str]   # "approve" | "reject" | None
    human_feedback: str

    # Memory fields
    retrieved_memories: List[Dict[str, Any]]   # injected into supervisor planning
    saved_memory_ids: List[str]                # IDs of memories saved after task
