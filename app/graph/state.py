from typing import TypedDict, Dict, Any, List, Optional


class AgentState(TypedDict):
    task_id: str
    user_id: str
    original_request: str

    execution_plan: Dict[str, Any]
    current_subtask_index: int

    agent_outputs: Dict[str, Any]
    review_result: Dict[str, Any]

    final_output: Optional[str]
    errors: List[Dict[str, Any]]
