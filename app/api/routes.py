from typing import AsyncGenerator, Optional
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph

from app.api.schemas import QueryRequest, QueryResponse, StreamEvent, AgentResultResponse, PlanResponse, SessionCreateResponse
from app.util.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api", tags=["query"])

# 그래프 인스턴스 (main.py에서 설정)
_graph:CompiledStateGraph = None


def set_graph(graph: CompiledStateGraph):
    """그래프 인스턴스 설정"""
    global _graph
    _graph = graph


def get_graph() -> CompiledStateGraph:
    """그래프 인스턴스 가져오기"""
    if _graph is None:
        raise HTTPException(status_code=500, detail="Graph not initialized")
    return _graph


def get_session_config(session_id: Optional[str] = None) -> dict:
    """
    세션 설정 생성

    Args:
        session_id: 사용자 제공 세션 ID (없으면 자동 생성)

    Returns:
        LangGraph config 딕셔너리
    """
    # session_id가 없으면 새로 생성
    if not session_id:
        session_id = str(uuid.uuid4())

    return {
        "configurable": {
            "thread_id": session_id
        }
    }


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    기본 쿼리 엔드포인트 (전체 응답 반환)

    Args:
        request: 쿼리 요청

    Returns:
        전체 실행 결과 포함한 응답
    """
    try:
        graph = get_graph()

        # 세션 설정 생성
        config = get_session_config(request.session_id)

        logger.info(f"[API] Query received: {request.query}, session_id: {config['configurable']['thread_id']}")

        # 초기 state 구성
        initial_state = {
            "query": request.query,
            "messages": [HumanMessage(content=request.query)],
            "plans": None,
            "plan": None,
            "replan_count": 0,
            "current_step": 0,
            "task_assignments": None,
            "agent_results": None,
            "evaluation": None,
            "final_answer": None,
            "conversation_histories": []
        }

        final_state = await graph.ainvoke(initial_state, config=config)
            
        # 응답 구성
        plan_response = [
            PlanResponse(
                reasoning=plan.reasoning,
                total_steps=plan.total_steps,
                execution_mode=plan.execution_mode
            )
            for plan in final_state.get("plans", [])
        ]

        agent_results = [
            AgentResultResponse(
                name=result.name,
                task=result.task,
                result=result.result,
                step_number=result.step_number,
                success=result.success
            )
            for result in final_state.get("agent_results", [])
        ]

        return QueryResponse(
            query=request.query,
            answer=final_state.get("final_answer", "답변을 생성할 수 없습니다."),
            plans=plan_response,
            agent_results=agent_results,
            success=True,
            error=None
        )

    except Exception as e:
        logger.error(f"[API] Query failed: {str(e)}", exc_info=True)
        return QueryResponse(
            query=request.query,
            answer="",
            plans=[],
            agent_results=[],
            success=False,
            error=str(e)
        )


@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    스트리밍 쿼리 엔드포인트 (실시간 이벤트 스트리밍)

    Args:
        request: 쿼리 요청

    Returns:
        Server-Sent Events (SSE) 스트림
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            graph = get_graph()

            # 세션 설정 생성
            config = get_session_config(request.session_id)

            logger.info(f"[API] Streaming query received: {request.query}, session_id: {config['configurable']['thread_id']}")

            # 초기 state 구성
            initial_state = {
                "query": request.query,
                "messages": [HumanMessage(content=request.query)],
                "plans": None,
                "plan": None,
                "replan_count": 0,
                "current_step": 0,
                "task_assignments": None,
                "agent_results": None,
                "evaluation": None,
                "final_answer": None,
                "conversation_histories": []
            }

            # 그래프 스트리밍 실행 (config 추가)
            async for event in graph.astream(initial_state, config=config):
                # event는 {node_name: state_update} 형태
                for node_name, state_update in event.items():
                    # 노드 실행 이벤트
                    stream_event = StreamEvent(
                        event="node",
                        node=node_name,
                        data={}
                    )

                    # Plan 이벤트 (planner / replanner)
                    if node_name in ("planner", "replanner") and state_update.get("plan"):
                        steps = []
                        plan = state_update["plan"]
                        
                        for step in plan.steps:
                            steps.append(step.task)

                        stream_event = StreamEvent(
                            event="plan",
                            node=node_name,
                            data={
                                "steps" : steps,
                                "reasoning": plan.reasoning,
                                "total_steps": plan.total_steps,
                                "execution_mode": plan.execution_mode
                            }
                        )
                        yield f"data: {stream_event.model_dump_json()}\n\n"

                    # supervisor 노드의 agent 지정 결과 이벤트
                    elif node_name == "supervisor" and state_update.get("task_assignments"):
                        assignments = []
                        for task_assignment in state_update.get("task_assignments"):
                            assignments.append({
                                "agent_name" : task_assignment.get("agent_name"),
                                "task": task_assignment.get("task"),
                                "step_number": task_assignment.get("step_number")
                            })
                            
                        stream_event = StreamEvent(
                            event="supervisor",
                            node=node_name,
                            data={
                                "assignments" : assignments
                            }
                        )
                        yield f"data: {stream_event.model_dump_json()}\n\n"
                    
                    # 에이전트 노드 실행 결과 이벤트 (agent_*)
                    elif node_name.startswith("agent_") and state_update.get("agent_results"):
                        result = state_update["agent_results"][-1]
                        stream_event = StreamEvent(
                            event="agent_result",
                            node=node_name,
                            data={
                                "name": result.name,
                                # "task": result.task,
                                "result": result.result,
                                "step_number": result.step_number,
                                "success": result.success
                            }
                        )
                        yield f"data: {stream_event.model_dump_json()}\n\n"
                    
                    # 에이전트 실행 결과 평가 이벤트
                    elif node_name.startswith("evaluator") and state_update.get("evaluation"):
                        evaluation = state_update["evaluation"]
                        stream_event = StreamEvent(
                            event="evaluation",
                            node=node_name,
                            data={
                                "evaluation" : evaluation
                            }
                        )
                        yield f"data: {stream_event.model_dump_json()}\n\n"

                    # 최종 답변 이벤트
                    elif node_name == "synthesizer" and state_update.get("final_answer"):
                        stream_event = StreamEvent(
                            event="final_answer",
                            node=node_name,
                            data={
                                "answer": state_update["final_answer"]
                            }
                        )
                        yield f"data: {stream_event.model_dump_json()}\n\n"

                    else:
                        yield f"data: {stream_event.model_dump_json()}\n\n"

            # 완료 신호
            yield "data: {\"event\": \"done\"}\n\n"

        except Exception as e:
            logger.error(f"[API] Streaming query failed: {str(e)}", exc_info=True)
            error_event = StreamEvent(
                event="error",
                node=None,
                data={"error": str(e)}
            )
            yield f"data: {error_event.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.post("/session/create", response_model=SessionCreateResponse)
async def create_session():
    """
    새 세션 ID 생성 엔드포인트

    Returns:
        생성된 세션 ID와 생성 시간
    """
    session_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat() + "Z"

    logger.info(f"[API] New session created: {session_id}")

    return SessionCreateResponse(
        session_id=session_id,
        created_at=created_at
    )


@router.get("/health")
async def health_check():
    """헬스 체크 엔드포인트"""
    return {"status": "healthy", "graph_initialized": _graph is not None}
