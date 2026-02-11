"""
최종 답변이 부족할 경우 계획 재수립
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

class RePlanner(LLMNode):
    def __init__(self, llm: BaseChatModel, prompt_file: str = None):
        super().__init__("replan", llm, prompt_file)

    async def replan_node(self, state: AgentState) -> Dict[str, Any]:
        """
        재실행 계획 수립

        사용자 쿼리와 생성된 최종 답변과 부족한점, Evaluator 제안 기반:
        1. 작업 복잡도 판단 (단순/병렬/복잡)
        2. 적절한 재실행 계획 생성
        3. State에 계획 저장
        Args:
            state: 현재 에이전트 상태 (query 필드 포함)

        Returns:
            계획 정보를 포함한 부분 State 업데이트
            - plan: 생성된 ExecutionPlan 객체
            - plans: 계획 히스토리 리스트
        """
        user_query = state['query']
        replan_count = state['replan_count']
        evaluation = state['evaluation']
        missing_info = evaluation.missing_info
        suggestions = evaluation.suggestions
        conversation_histories = state.get('conversation_histories', [])

        logger.info(f"[REPLAN] 재계획 수립 중...")
        logger.debug(f"""사용자 요청: {user_query},\n
                    부족한 정보 : {missing_info},\n
                    제안 : {suggestions}""")

        conversation_history = '\n'.join(conversation_histories) if conversation_histories else "No previous conversation"

        try:
            structured_llm = self.llm.with_structured_output(ExecutionPlan)
            replan_chain = self.prompt_template | structured_llm

            execution_plan = await replan_chain.ainvoke({
                "user_query": user_query,
                "missing_info": missing_info,
                "suggestions": suggestions,
                "conversation_history": conversation_history
            })
            
            logger.info(f"[REREPLAN] 계획 수립 완료 - 총 {execution_plan.total_steps}단계, 모드: {execution_plan.execution_mode}")
        
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"[REPLAN] 계획 스키마 검증 실패: {str(e)}")
            execution_plan = self._create_fallback_plan(user_query, missing_info, suggestions, str(e))
            
        except Exception as e:
            logger.error(f"[REPLAN] 계획 수립 예외 발생: {str(e)}")
            execution_plan = self._create_fallback_plan(user_query, missing_info, suggestions, str(e))

        return {
                "messages" : [AIMessage(content=f"[REPLAN] 계획 수립 [{execution_plan.execution_mode} {execution_plan.total_steps} Step] : {execution_plan.reasoning}")],
                "plan": execution_plan,
                "plans" : [execution_plan],
                "replan_count" : replan_count + 1,
                "current_step": 0,
                "task_assignments" : []
            }
    
    def _create_fallback_plan(self, user_query: str, missing_info: str, suggestions: str, error_message: str):
        task = f"""
            User Query: {user_query}
            Missing Information: {missing_info}
            Suggestions: {suggestions}
        """

        step = Step(
            task = task,
            step_number = 1
        )

        execution_plan = ExecutionPlan(
            steps = [step],
            reasoning = f"계획 수립 중 예외 발생으로 기본 계획 생성 : {error_message}",
            total_steps=1,
            execution_mode="sequential"
        )

        return execution_plan