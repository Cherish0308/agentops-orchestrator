from app.agents.specialist_base import run_specialist_node


def research_agent_node(state):
    return run_specialist_node(state, "research_agent")
