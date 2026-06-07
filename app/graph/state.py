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
    approval_level: str                # notify | approve_action | approve_plan | take_over
    human_decision: Optional[str]      # approve | reject | modify | take_over
    human_feedback: str
    subtask_failure_counts: Dict[str, int]   # subtask_id → failure count

    # Memory fields
    retrieved_memories: List[Dict[str, Any]]
    saved_memory_ids: List[str]
