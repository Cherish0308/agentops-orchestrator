import json
from app.llm import call_llm


def specialist_execution_node(state):
    system_prompt = """
You are the Specialist Execution Agent.
Execute each subtask from the execution plan using appropriate strategies.
Return ONLY valid JSON with agent outputs.

JSON schema:
{
  "subtask_id": "id",
  "output": "result",
  "confidence_score": 0.0,
  "tools_used": [],
  "errors": []
}
"""

    user_prompt = f"""
Original request:
{state["original_request"]}

Execution plan:
{state["execution_plan"]}

Execute all subtasks and provide outputs.
"""

    raw_response = call_llm(system_prompt, user_prompt)

    try:
        outputs = json.loads(raw_response)
        if isinstance(outputs, dict):
            agent_outputs = {"specialist": outputs}
        else:
            agent_outputs = {"specialist": outputs}
    except json.JSONDecodeError:
        agent_outputs = {
            "specialist": {
                "output": raw_response,
                "confidence_score": 0.6,
                "tools_used": [],
                "errors": ["JSON parsing failed"]
            }
        }

    return {
        "agent_outputs": agent_outputs
    }
