from typing import List
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, END
from langgraph.types import Send
from langgraph.checkpoint.base import BaseCheckpointSaver

from app.node.plan import Planner
from app.node.supervisor import Supervisor
from app.node.replan import RePlanner
from app.node.evaluator import Evaluator
from app.node.synthesizer import Synthesizer
from app.node.sub.agent_node import AgentNode
from app.state import AgentState
from app.util.logger import get_logger
from app.util.llm_builder import build_agent_llm
from app.util.config_loader import get_agent_config, get_mcp_config

logger = get_logger(__name__)

async def create_graph(llm: BaseChatModel, checkpointer: BaseCheckpointSaver) -> StateGraph:
    """
    그래프 생성 (PLAN + ReAct + Multi-Agent 구조)

    Returns:
        컴파일된 LangGraph StateGraph
    """
    logger.info("[GRAPH] PLAN + ReAct + Multi-Agent 기반 LangGraph 생성")
    
    # 설정 로드 (Agent, MCP 서버, LLM 세팅)
    agent_config = get_agent_config()
    mcp_server_config = get_mcp_config()
    
    # 노드 초기화
    planner = Planner(llm, agent_config)
    supervisor = Supervisor(agent_config)
    evaluator = Evaluator(llm)
    replanner = RePlanner(llm, agent_config)
    synthesizer = Synthesizer(llm)
    
    # 그래프 구성
    graph = StateGraph(AgentState)
    
    # 노드 추가
    graph.add_node("planner", planner.plan_node)
    graph.add_node("supervisor", supervisor.supervisor_node)
    graph.add_node("evaluator", evaluator.evaluator_node)
    graph.add_node("replanner", replanner.replan_node)
    graph.add_node("synthesizer", synthesizer.synthesizer_node)
    
    for agent in agent_config:
        agent_llm = build_agent_llm(agent.llm) if agent.llm else llm
        graph.add_node(agent.name, AgentNode(llm=agent_llm, agent=agent, mcp_configs=mcp_server_config).execute)
        
    graph.set_entry_point("planner")
    
    graph.add_edge("planner", "supervisor")
    graph.add_edge("replanner", "supervisor")
    
    graph.add_conditional_edges(
        "supervisor",
        _route_to_agent
    )
    
    for agent in agent_config:
        graph.add_edge(agent.name, "supervisor")
    
    graph.add_conditional_edges(
        'evaluator',
        _route_after_evaluation
    )
    
    graph.add_edge("synthesizer", END)
    
    return graph.compile(checkpointer=checkpointer)

        
def _route_to_agent(state: AgentState) -> str | Send | List[Send]:
    user_query = state.get('query')
    routing_decision = state.get("routing_decision")
    agent_results = state.get('agent_results')
    
    if routing_decision.is_parallel:
        next_steps = routing_decision.next_steps
        next_agents = [step.agent for step in next_steps]
        logger.info(f"[GRAPH] 병렬 실행 처리 : {next_agents}")
        return [Send(step.agent, {"agent_name" : step.agent, "query" : user_query, "task": step.task, "step_number" : step.step_number, "agent_results" : agent_results}) for step in next_steps]
    
    next_step = routing_decision.next_step

    if not next_step:
        logger.info("[GRAPH] 단계 실행 종료 -> EVALUATOR 노드 이동")
        return "evaluator"

    logger.info(f"[GRAPH] 순차 실행 처리 : {next_step.agent}")
    return Send(next_step.agent, {"agent_name" : next_step.agent, "query" : user_query, "task": next_step.task, "step_number" : next_step.step_number, "agent_results" : agent_results})

def _route_after_evaluation(state: AgentState) -> str:
    evaluation = state.get("evaluation")
    status = evaluation.status
    replan_count = state.get('replan_count')
    
    if status == 'REPLAN' and replan_count < 3:
        logger.info("[GRAPH] 결과 불충분 판단 -> REPLANNER 노드 이동")
        return "replanner"

    logger.info(f"[GRAPH] {"결과 충분" if status == 'SUFFICIENT' else "추가 정보 필요"} 판단 -> SYNTHESIZER 노드 이동")
    return "synthesizer"