from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, END
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

    # AgentNode 인스턴스들 먼저 생성
    agent_nodes = []
    for agent in agent_config:
        if agent.enabled:
            agent_llm = build_agent_llm(agent.llm) if agent.llm else llm
            agent_node = AgentNode(llm=agent_llm, agent=agent, mcp_configs=mcp_server_config)
            agent_nodes.append(agent_node)

    # 노드 초기화
    planner = Planner(llm)
    supervisor = Supervisor(llm, agent_nodes)
    evaluator = Evaluator(llm)
    replanner = RePlanner(llm)
    synthesizer = Synthesizer(llm)

    # 그래프 구성
    graph = StateGraph(AgentState)

    # 노드 추가
    graph.add_node("planner", planner.plan_node)
    graph.add_node("supervisor", supervisor.supervisor_node)
    graph.add_node("evaluator", evaluator.evaluator_node)
    graph.add_node("replanner", replanner.replan_node)
    graph.add_node("synthesizer", synthesizer.synthesizer_node)

    # 그래프 연결 (단순화된 구조)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "supervisor")
    graph.add_edge("supervisor", "evaluator")
    graph.add_edge("replanner", "supervisor")

    graph.add_conditional_edges(
        'evaluator',
        _route_after_evaluation
    )

    graph.add_edge("synthesizer", END)

    return graph.compile(checkpointer=checkpointer)

        
def _route_after_evaluation(state: AgentState) -> str:
    evaluation = state.get("evaluation")
    status = evaluation.status
    replan_count = state.get('replan_count')
    
    if status == 'REPLAN' and replan_count < 3:
        logger.info("[GRAPH] 결과 불충분 판단 -> REPLANNER 노드 이동")
        return "replanner"

    logger.info(f"[GRAPH] {"결과 충분" if status == 'SUFFICIENT' else "추가 정보 필요"} 판단 -> SYNTHESIZER 노드 이동")
    return "synthesizer"