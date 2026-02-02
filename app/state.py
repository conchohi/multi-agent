import operator

from typing import List, Annotated, Optional, TypedDict, Dict, Literal, Any
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

def list_reducer(existing: List[str], new: List[str] | None) -> List[str]:
    """
    None이면 초기화, 리스트면 추가
    """
    if new is None:
        return []
    return existing + new

class Step(BaseModel):
    agent: Literal['ChatAgent', 'CodeAgent', 'SearchAgent', 'WeatherAgent'] = Field(..., description="단계 실행 에이전트")
    task: str = Field(..., description="에이전트의 담당 업무")
    step_number: int = Field(default=0, ge=0, description="실행 단계 순서")

# PLAN 상태 정의
class ExecutionPlan(BaseModel):
    steps: List[Step] = Field(..., description='실행 단계 리스트 [{agent: str, task: str,}]')
    reasoning: str = Field(..., description="계획 수립 이유")
    total_steps: int = Field(..., description="전체 단계 수")
    execution_mode: Literal['sequential', 'parallel'] = Field(..., description="계획 실행 모드 (순차적, 병렬적)")

class RoutingDecision(BaseModel):
    next_step: Optional[Step] = Field(default=None, description="순차 작업 시 다음 실행할 에이전트")
    next_steps: List[Step] = Field(default=[], description="병렬 작업 시 실행할 에이전트들")
    is_parallel: bool = Field(default=False, description="병렬 처리 여부")
    
class AgentResult(BaseModel):
    name: str = Field(..., description="서브 에이전트 명")
    task: str = Field(..., description="서브 에이전트 처리할 일")
    result: str = Field(..., description="서브 에이전트 실행 결과")
    step_number: int = Field(default=0, description="서브 에이전트 실행 순서")
    success: bool = Field(..., description="성공 여부")

class Evaulation(BaseModel):
    status: Literal['REPLAN', 'SUFFICIENT', 'CLARIFY']
    reasoning: str = Field(..., description="")
    missing_info: Optional[str] = None
    suggestions: Optional[str] = None

class AgentState(TypedDict):
    query: str
    messages: Annotated[List[BaseMessage], add_messages]

    # 계획, 재계획
    plans: Annotated[List[ExecutionPlan], list_reducer]
    plan: Optional[ExecutionPlan]
    current_step: int
    replan_count: int

    # 실행 관리
    routing_decision: Optional[RoutingDecision]
    running_steps: bool

    # 서브 에이전트 실행 결과
    agent_results: Annotated[List[AgentResult], list_reducer]
    
    # 평가
    evaluation: Optional[Evaulation]
    
    # 최종 결과 표시
    final_answer: Optional[str]
    
    # 대화내역
    conversation_histories: Annotated[List[str], operator.add]
    
class SubAgentState(TypedDict):
    agent_name: str
    query: str
    task: str
    step_number: int
    agent_results: List[AgentResult]
    
