from app.llm import call_llm, current_task_id, current_node
from app.memory import short_term
from app.memory.manager import extract_and_save
from app.observability.tracer import tracer


def synthesizer_node(state):
    task_id = state.get("task_id", "")
    user_id = state.get("user_id", "anonymous")

    current_task_id.set(task_id)
    current_node.set("synthesizer")
    tracer.node_enter(task_id, "synthesizer", input_keys=["agent_outputs", "review_result"])

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

    # --- Collect all tools used across agent outputs ---
    tools_used = []
    for output in state.get("agent_outputs", {}).values():
        if isinstance(output, dict):
            tools_used.extend(output.get("tools_used", []))
    tools_used = list(set(tools_used))

    # --- Save to short-term memory (for API inspection during the run) ---
    short_term.write(task_id, "final_output", final_output)
    short_term.write(task_id, "tools_used", tools_used)

    # --- Extract and save to long-term semantic memory ---
    saved_memory_ids = extract_and_save(
        user_id=user_id,
        task_id=task_id,
        original_request=state["original_request"],
        execution_plan=state.get("execution_plan", {}),
        agent_outputs=state.get("agent_outputs", {}),
        final_output=final_output,
        tools_used=tools_used,
        success=not bool(state.get("errors")),
    )

    # --- Clear short-term working memory now that the task is done ---
    short_term.clear(task_id)

    tracer.node_exit(task_id, "synthesizer", output_keys=["final_output"])

    return {
        "final_output": final_output,
        "saved_memory_ids": saved_memory_ids,
    }
