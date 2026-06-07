from app.agents.specialist_base import run_specialist_node


def data_agent_node(state):
    return run_specialist_node(state, "data_agent")
