"""
사용자 질문에 따라 실행 계획 수립
"""
import json
from typing import Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from app.node.llm_node import LLMNode
from app.state import AgentState, Step, ExecutionPlan
from app.util.logger import get_logger

logger = get_logger(__name__)

class Planner(LLMNode):
    def __init__(self, llm: BaseChatModel, prompt_file: str = None):
        super().__init__("plan", llm, prompt_file)

    async def plan_node(self, state: AgentState) -> Dict[str, Any]:
        """
        실행 계획 수립

        사용자 쿼리를 분석하여:
        1. 작업 복잡도 판단 (순차/병렬)
        2. 적절한 실행 계획 생성
        3. State에 계획 저장
        Args:
            state: 현재 에이전트 상태 (query 필드 포함)

        Returns:
            계획 정보를 포함한 부분 State 업데이트
            - plan: 생성된 ExecutionPlan 객체
            - plans: 계획 히스토리 리스트
        """
        user_query = state['query']
        conversation_histories = state.get('conversation_histories', [])

        logger.info(f"[PLAN] 초기 계획 수립 중...")
        logger.info(f"사용자 요청: {user_query}")

        # 대화 내역을 문자열로 포맷팅
        conversation_history = '\n'.join(conversation_histories) if conversation_histories else "No previous conversation"

        logger.debug(f"대화 내용 : {conversation_history}")

        try:
            structured_llm = self.llm.with_structured_output(ExecutionPlan)
            plan_chain = self.prompt_template | structured_llm

            execution_plan = await plan_chain.ainvoke({
                "user_query" : user_query,
                "conversation_history" : conversation_history
            })
            
            logger.info(f"[PLAN] 계획 수립 완료 - 총 {execution_plan.total_steps}단계, 모드: {execution_plan.execution_mode}")
        
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"[PLAN] 계획 스키마 검증 실패: {str(e)}")
            execution_plan = self._create_fallback_plan(user_query, str(e))
            
        except Exception as e:
            logger.error(f"[PLAN] 계획 수립 예외 발생: {str(e)}")
            execution_plan = self._create_fallback_plan(user_query, str(e))

        return {
                "messages" : [AIMessage(content=f"[PLAN] 계획 수립 [{execution_plan.execution_mode} {execution_plan.total_steps} Step] : {execution_plan.reasoning}")], 
                "plan": execution_plan,
                "plans" : [execution_plan]
            }
    
    def _create_fallback_plan(self, user_query: str, error_message: str):
        step = Step(
            task = user_query,
            step_number = 1
        )

        execution_plan = ExecutionPlan(
            steps = [step],
            reasoning = f"계획 수립 중 예외 발생으로 기본 계획 생성 : {error_message}",
            total_steps=1,
            execution_mode="sequential"
        )

        return execution_plan