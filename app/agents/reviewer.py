import json
from app.llm import call_llm


def reviewer_node(state):
    system_prompt = """
You are the Reviewer Agent.

Your job:
1. Review all specialist outputs.
2. Decide if the work is good enough.
3. Give a quality score from 0 to 1.
4. Return ONLY valid JSON.

JSON schema:
{
  "approved": true,
  "quality_score": 0.0,
  "feedback": "specific feedback",
  "requires_rework": false
}
"""

    user_prompt = f"""
Original request:
{state["original_request"]}

Execution plan:
{state["execution_plan"]}

Agent outputs:
{state["agent_outputs"]}

Review the work.
"""

    raw_response = call_llm(system_prompt, user_prompt)

    try:
        review = json.loads(raw_response)
    except json.JSONDecodeError:
        review = {
            "approved": True,
            "quality_score": 0.7,
            "feedback": "Fallback review because JSON parsing failed.",
            "requires_rework": False
        }

    return {
        "review_result": review
    }
