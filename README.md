# AgentOps Orchestrator

A production-style multi-agent orchestration system built with Python, FastAPI, and LangGraph.

The goal of this project is to demonstrate how autonomous AI workflows can be structured beyond a single chatbot. The system uses a supervisor agent to decompose complex tasks, specialist agents to execute subtasks, a reviewer agent to validate outputs, and a synthesizer agent to produce the final response.

## Phase 1 Scope

Phase 1 builds the core agent architecture:

- FastAPI backend
- LangGraph state machine
- Supervisor Agent
- Specialist Agents
  - Research Agent
  - Data Agent
  - Writer Agent
- Reviewer Agent
- Final Synthesizer Agent
- Shared workflow state
- API endpoint for task execution

## Architecture

```text
User Request
   |
   v
FastAPI Task Endpoint
   |
   v
Supervisor Agent
   |
   v
Specialist Execution Layer
   |---- Research Agent
   |---- Data Agent
   |---- Writer Agent
   |
   v
Reviewer Agent
   |
   v
Final Synthesizer
   |
   v
Final Response
```

## Tech Stack

- Python 3.11+
- FastAPI
- LangGraph
- Pydantic
- OpenAI API
- Uvicorn
- python-dotenv

## Local Setup

Create a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-4o-mini
```

Run the API:

```bash
uvicorn app.main:app --reload
```

Open the API docs:

```text
http://localhost:8000/docs
```

## Example Request

```bash
curl -X POST http://localhost:8000/tasks/run \
  -H "Content-Type: application/json" \
  -d '{"user_id":"charish","request":"Find 3 AI job opportunities and draft personalized recruiter outreach messages."}'
```

## Planned Phases

### Phase 2: Tool Registry and Tool Logging

- Tool registry
- Tool input/output schema
- Tool-call logging
- Latency tracking
- Error tracking

### Phase 3: Memory System

- Redis short-term working memory
- ChromaDB long-term semantic memory
- Memory retrieval during planning

### Phase 4: Human-in-the-Loop Review

- Approval queue
- Escalation triggers
- Pause/resume execution
- Human review dashboard

### Phase 5: Observability

- Trace explorer
- Token and cost tracking
- OpenTelemetry spans
- Replay system

## Portfolio Positioning

This project demonstrates production-grade AI workflow design, including agent orchestration, task decomposition, validation loops, tool-use architecture, and future-ready AgentOps observability.
