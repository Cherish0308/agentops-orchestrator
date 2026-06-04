from app.llm import call_llm


def synthesizer_node(state):
    system_prompt = """
You are the Final Synthesis Agent.
Create the final user-facing response using:
- the original request
- execution plan
- specialist outputs
- reviewer feedback

Be clear, practical, and concise.
"""

    user_prompt = f"""
Original request:
{state["original_request"]}

Execution plan:
{state["execution_plan"]}

Agent outputs:
{state["agent_outputs"]}

Review result:
{state["review_result"]}

Create the final answer.
"""

    final_output = call_llm(system_prompt, user_prompt)

    return {
        "final_output": final_output
    }
