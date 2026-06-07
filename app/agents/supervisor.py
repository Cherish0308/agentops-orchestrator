import json
from app.llm import call_llm, current_task_id, current_node
from app.memory.manager import retrieve_for_planning
from app.memory import short_term
from app.observability.tracer import tracer


def supervisor_node(state):
    task_id = state.get("task_id", "")
    current_task_id.set(task_id)
    current_node.set("supervisor")
    tracer.node_enter(task_id, "supervisor", input_keys=["original_request"])

    is_rework = state.get("review_result", {}).get("requires_rework", False)
    rework_count = state.get("rework_count", 0)

    if is_rework:
        rework_count += 1

    system_prompt = """
You are the Supervisor Agent in a multi-agent orchestration system.

Your job:
1. Understand the user request.
2. Break it into clear subtasks.
3. Assign each subtask to one of these agents:
   - research_agent
   - data_agent
   - writer_agent
4. Return ONLY valid JSON.

JSON schema:
{
  "task_summary": "short summary",
  "confidence_score": 0.0,
  "risk_level": "low | medium | high",
  "subtasks": [
    {
      "id": "subtask_1",
      "description": "what needs to be done",
      "assigned_agent": "research_agent | data_agent | writer_agent",
      "depends_on": [],
      "expected_output": "expected format",
      "complexity": "low | medium | high"
    }
  ]
}
"""

    rework_context = ""
    if is_rework:
        rework_context = f"""
This is a rework pass. Previous reviewer feedback:
{state.get("review_result", {})}

Previous agent outputs:
{state.get("agent_outputs", {})}

Create an improved execution plan addressing the feedback.
"""

    # Query long-term memory for similar past tasks
    memory_context = ""
    if not is_rework:
        memory_context = retrieve_for_planning(
            user_id=state.get("user_id", "anonymous"),
            request=state["original_request"],
        )

    # Store the request in short-term working memory
    short_term.write(state.get("task_id", ""), "original_request", state["original_request"])

    user_prompt = f"""
Original user request:
{state["original_request"]}

{memory_context}
{rework_context}
Create an execution plan.
"""

    raw_response = call_llm(system_prompt, user_prompt)

    try:
        plan = json.loads(raw_response)
    except json.JSONDecodeError:
        plan = {
            "task_summary": "Fallback plan because JSON parsing failed",
            "confidence_score": 0.5,
            "risk_level": "medium",
            "subtasks": [
                {
                    "id": "subtask_1",
                    "description": state["original_request"],
                    "assigned_agent": "writer_agent",
                    "depends_on": [],
                    "expected_output": "final response",
                    "complexity": "medium"
                }
            ]
        }

    tracer.node_exit(task_id, "supervisor", output_keys=["execution_plan"])

    return {
        "execution_plan": plan,
        "current_subtask_index": 0,
        "agent_outputs": {},
        "rework_count": rework_count,
    }
