import json
from app.llm import call_llm


def supervisor_node(state):
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

    user_prompt = f"""
Original user request:
{state["original_request"]}

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

    return {
        "execution_plan": plan,
        "current_subtask_index": 0
    }
