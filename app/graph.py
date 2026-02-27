from typing import List
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.types import Send

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
    agent_nodes: List[AgentNode] = []
    for agent in agent_config:
        if agent.enabled:
            agent_llm = build_agent_llm(agent.llm) if agent.llm else llm
            agent_node = AgentNode(llm=agent_llm, agent=agent, mcp_configs=mcp_server_config)
            agent_nodes.append(agent_node)

    # 에이전트 그래프 노드 이름 목록 (예: "agent_ChatAgent")
    agent_node_names = [f"agent_{node.name}" for node in agent_nodes]

    # 노드 초기화
    planner = Planner(llm)
    supervisor = Supervisor(llm, agent_nodes)
    evaluator = Evaluator(llm)
    replanner = RePlanner(llm)
    synthesizer = Synthesizer(llm)

    # 서버 시작 시 Supervisor(Agent MCP tools) 미리 초기화
    await supervisor.initialize()
    logger.info("[GRAPH] Supervisor 초기화 완료")

    # 그래프 구성
    graph = StateGraph(AgentState)

    # 기본 노드 추가
    graph.add_node("planner", planner.plan_node)
    graph.add_node("supervisor", supervisor.supervisor_node)
    graph.add_node("evaluator", evaluator.evaluator_node)
    graph.add_node("replanner", replanner.replan_node)
    graph.add_node("synthesizer", synthesizer.synthesizer_node)

    # 각 AgentNode를 독립된 그래프 노드로 추가
    for agent_node in agent_nodes:
        node_name = f"agent_{agent_node.name}"
        graph.add_node(node_name, agent_node.graph_node)
        logger.info(f"[GRAPH] 에이전트 노드 추가: {node_name}")

    # 그래프 연결
    graph.set_entry_point("planner")
    graph.add_edge("planner", "supervisor")
    graph.add_edge("replanner", "supervisor")
    graph.add_edge("synthesizer", END)

    # supervisor → 에이전트 노드 (Send를 통한 동적 라우팅)
    graph.add_conditional_edges(
        "supervisor",
        _route_from_supervisor
    )

    # 각 에이전트 노드 → supervisor(순차, 다음 step) 또는 evaluator(완료)
    for node_name in agent_node_names:
        graph.add_conditional_edges(
            node_name,
            _route_after_agent,
            {"supervisor": "supervisor", "evaluator": "evaluator"}
        )

    # evaluator → replanner 또는 synthesizer
    graph.add_conditional_edges(
        'evaluator',
        _route_after_evaluation
    )

    return graph.compile(checkpointer=checkpointer)


def _route_from_supervisor(state: AgentState) -> List[Send]:
    """
    Supervisor가 저장한 task_assignments를 읽어 각 에이전트 노드로 Send
    """
    task_assignments = state.get('task_assignments') or []
    query = state.get('query', '')

    sends = []
    
    for assignment in task_assignments:
        sub_state = {
            "agent_name": assignment["agent_name"],
            "query": query,
            "task": assignment["task"],
            "step_number": assignment["step_number"],
            "agent_results": state["agent_results"]
        }
        node_name = f"agent_{assignment['agent_name']}"
        logger.info(f"[GRAPH] Send → {node_name}: {assignment['task']}")
        sends.append(Send(node_name, sub_state))

    return sends


def _route_after_agent(state: AgentState) -> str:
    """
    에이전트 노드 실행 후 라우팅:
    - sequential 모드에서 남은 step이 있으면 → supervisor (다음 step)
    - parallel 모드: 모든 Send() 노드가 evaluator로 수렴 (LangGraph fan-in)
    - 그 외 → evaluator
    """
    plan = state.get('plan')

    if plan:
        if plan.execution_mode == 'sequential':
            current_step = state.get('current_step', 0)
            if current_step < len(plan.steps):
                logger.info(f"[GRAPH] 순차 실행 계속: step {current_step + 1}/{len(plan.steps)} → supervisor")
                return "supervisor"
        else:  # parallel: 모든 병렬 에이전트가 evaluator로 fan-in
            logger.info("[GRAPH] 병렬 에이전트 완료 → evaluator (fan-in 대기)")
            return "evaluator"

    logger.info("[GRAPH] 모든 step 완료 → evaluator")
    return "evaluator"


def _route_after_evaluation(state: AgentState) -> str:
    evaluation = state.get("evaluation")
    status = evaluation.status
    replan_count = state.get('replan_count')

    if status == 'REPLAN' and replan_count < 3:
        logger.info("[GRAPH] 결과 불충분 판단 -> REPLANNER 노드 이동")
        return "replanner"

    logger.info(f"[GRAPH] {"결과 충분" if status == 'SUFFICIENT' else "추가 정보 필요"} 판단 -> SYNTHESIZER 노드 이동")
    return "synthesizer"
