from typing import Optional, List, Literal
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """쿼리 요청 스키마"""
    query: str = Field(..., description="사용자 질문", min_length=1)
    session_id: Optional[str] = Field(default=None, description="세션 ID (선택)")


class AgentResultResponse(BaseModel):
    """에이전트 실행 결과"""
    name: str = Field(..., description="에이전트 이름")
    task: str = Field(..., description="수행한 작업")
    result: str = Field(..., description="실행 결과")
    step_number: int = Field(..., description="실행 단계 번호")
    success: bool = Field(..., description="성공 여부")


class PlanResponse(BaseModel):
    """실행 계획"""
    reasoning: str = Field(..., description="계획 수립 이유")
    total_steps: int = Field(..., description="전체 단계 수")
    execution_mode: Literal['sequential', 'parallel'] = Field(..., description="실행 모드")

class QueryResponse(BaseModel):
    """쿼리 응답 스키마 (기본)"""
    query: str = Field(..., description="원본 질문")
    answer: str = Field(..., description="최종 답변")
    plans: List[PlanResponse] = Field(default=[], description="실행 계획")
    agent_results: List[AgentResultResponse] = Field(default=[], description="에이전트 실행 결과 목록")
    success: bool = Field(..., description="전체 성공 여부")
    error: Optional[str] = Field(default=None, description="에러 메시지 (실패 시)")


class StreamEvent(BaseModel):
    """스트리밍 이벤트"""
    event: Literal['node', 'plan', 'agent_result', 'final_answer', 'error'] = Field(..., description="이벤트 타입")
    node: Optional[str] = Field(default=None, description="현재 노드 이름")
    data: Optional[dict] = Field(default=None, description="이벤트 데이터")


class SessionCreateResponse(BaseModel):
    """세션 생성 응답"""
    session_id: str = Field(..., description="생성된 세션 ID")
    created_at: str = Field(..., description="생성 시간 (ISO 8601 형식)")
