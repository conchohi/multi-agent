# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LangGraph-based multi-agent chatbot system with dynamic planning, execution, and evaluation. The system orchestrates multiple specialized agents (ChatAgent, CodeAgent, SearchAgent, WeatherAgent) through a sophisticated workflow that can adapt and replan based on evaluation results.

## Development Commands

### Running the Application
```bash
# Start FastAPI server (default: http://0.0.0.0:8000)
python main.py

# Start with custom config
python main.py --config path/to/config.yaml

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup
- Copy `.env.example` to `.env` and configure:
  - `OPENAI_API_KEY` and `OPENAI_API_BASE` for OpenAI models
  - `ANTHROPIC_API_KEY` for Claude models
  - `OLLAMA_HOST` for local Ollama models

## Core Architecture

### Graph Workflow (app/graph.py)
The system uses a LangGraph StateGraph with the following node execution flow:

```
Entry → Planner → Supervisor → [Sub-Agents] → Supervisor → Evaluator → RePlanner/Synthesizer → END
                      ↑                |                        |              |
                      └────────────────┘                        └──────────────┘
                        (循環執行)                                  (再計劃 3次上限)
```

**Key Nodes:**
1. **Planner** (app/node/plan.py): Analyzes user query and creates ExecutionPlan with steps, reasoning, and execution mode (sequential/parallel)
2. **Supervisor** (app/node/supervisor.py): Routes tasks to appropriate agents based on the plan's execution mode (sequential or parallel via Send)
3. **Sub-Agents** (app/node/sub/agent_node.py): Execute tasks using ReAct pattern with MCP tools. Each agent has a specific prompt and MCP server configuration
4. **Evaluator** (app/node/evaluator.py): Evaluates if agent results are sufficient (returns SUFFICIENT, REPLAN, or CLARIFY status)
5. **RePlanner** (app/node/replan.py): Creates a new plan if evaluation is insufficient (max 3 replans via replan_count)
6. **Synthesizer** (app/node/synthesizer.py): Generates final user-facing answer from all agent results

### State Management (app/state.py)
- **AgentState**: Main state for the entire workflow
  - `plans`: List of all execution plans (including replans)
  - `agent_results`: Accumulated results from all agent executions
  - `evaluation`: Current evaluation status
  - `replan_count`: Counter to limit replanning (max 3)
  - `routing_decision`: Controls sequential vs parallel execution
  - Uses `list_reducer` for plans/agent_results to handle None initialization

- **SubAgentState**: Passed to individual agent nodes
  - `query`: Original user query
  - `task`: Specific task for the agent
  - `agent_results`: Recent results (最新 3件) from previous agents

### MCP Integration (app/node/sub/agent_node.py)
Agents use Model Context Protocol (MCP) to access external tools:
- MCP client initialization in `AgentNode.initialize()`
- Supports both stdio and HTTP transports
- Tools are loaded via `MultiServerMCPClient.get_tools()`
- Combined with base tools for ReAct agent execution
- **Important**: agent_summary now uses only the most recent 3 agent results (agent_results[-3:]) to avoid context overflow

### Configuration System

**Three-tier configuration:**
1. **config/config.yaml**: Main settings
   - LLM provider (OpenAI/Anthropic/Ollama), model, temperature
   - Session storage (memory/redis)
   - API server (host, port, reload)
   - Logging configuration
   - References to agent_config.yaml and mcp_config.yaml

2. **config/agent_config.yaml**: Agent definitions
   - Agent name, description, enabled status
   - MCP server assignments per agent
   - Each agent references a prompt file in prompts/

3. **config/mcp_config.yaml**: MCP server configurations
   - Transport type (stdio/sse)
   - Command/args for stdio, URL/headers for SSE

### API Endpoints (app/api/routes.py)

- `POST /api/query`: Non-streaming query with full response (plans, agent_results, final_answer)
- `POST /api/query/stream`: Server-Sent Events (SSE) streaming with real-time events (plan, agent_result, final_answer, done)
- `POST /api/session/create`: Generate new session ID for conversation continuity
- `GET /api/health`: Health check endpoint

**Session Management:**
- Sessions use `thread_id` in config for LangGraph checkpointing
- Enables multi-turn conversations with state persistence
- Supports memory or Redis-backed storage

### Checkpointing (app/util/checkpointer_factory.py)
- **Memory**: In-memory state (default, not persistent)
- **Redis**: Persistent state with `langgraph-checkpoint-redis`
- Configured via `session.storage` in config.yaml

## Important Implementation Patterns

### Adding New Agents
1. Create agent in `config/agent_config.yaml` with name, description, mcp_servers
2. Create prompt file in `prompts/{AgentName}_prompt.txt`
3. Add agent name to Step.agent Literal type in `app/state.py`
4. Graph automatically loads agents from config and adds nodes
5. AgentNode handles MCP tool loading and ReAct execution

### Modifying LLM Providers
- Edit `config/config.yaml` llm section
- Supported providers: "openai", "anthropic", "ollama"
- Each agent can override with custom LLM via `agent.llm` in config

### Prompt Engineering
- Prompts located in `prompts/` directory
- Loaded by LLMNode base class via `prompt_file` parameter
- Use jinja2 template syntax if needed (loaded as plain text by default)

### Parallel vs Sequential Execution
- Planner determines execution_mode based on task dependencies
- Supervisor uses `Send()` for parallel execution via `routing_decision.is_parallel`
- Parallel tasks are sent simultaneously to multiple agents
- Sequential tasks execute one at a time with accumulated agent_results

## Key Files Reference

- `main.py`: FastAPI application entry point with lifespan management
- `app/graph.py`: LangGraph workflow definition and routing logic
- `app/state.py`: State schemas and reducers
- `app/node/sub/agent_node.py`: ReAct agent with MCP tool integration
- `app/util/llm_builder.py`: LLM instance factory
- `app/util/config_loader.py`: Configuration loading and caching
- `app/util/react_agent_builder.py`: ReAct agent graph builder

## Notes

- Korean language is used extensively in logs and prompts
- Hot reload enabled by default in development (api.reload: true)
- Maximum 3 replanning attempts to prevent infinite loops
- Agent results summary in SubAgentState limited to most recent 3 to manage context size
