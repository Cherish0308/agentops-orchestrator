from app.agents.specialist_base import run_specialist_node


def writer_agent_node(state):
    return run_specialist_node(state, "writer_agent")
