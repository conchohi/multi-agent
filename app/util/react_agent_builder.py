"""Base ReAct agent implementation."""
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

def create_react_agent_graph(
    llm: BaseChatModel,
    tools: list,
    system_prompt: str
) -> CompiledStateGraph:
    """Create a ReAct agent using StateGraph.

    Args:
        llm: Language model to use
        tools: List of tools available to the agent
        system_prompt: System instructions for the agent

    Returns:
        Compiled StateGraph
    """
    # Bind tools to LLM
    llm_with_tools = llm.bind_tools(tools)

    workflow = StateGraph(MessagesState)

    async def call_model(state: MessagesState):
        """Agent reasoning step - decide to use tools or respond.

        Args:
            state: Current message state

        Returns:
            Updated messages
        """
        messages = state["messages"]

        # Add system prompt if first call
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt)] + messages

        # Call LLM with tools bound (async for streaming callback propagation)
        response = await llm_with_tools.ainvoke(messages)

        return {"messages": [response]}
    
    # Tool executor mapping
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))

    # Add edges
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent", tools_condition
    )
    workflow.add_edge("tools", "agent")

    return workflow.compile()
