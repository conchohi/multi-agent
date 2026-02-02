"""
최종 답변 평가
"""
import json
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from pydantic import ValidationError
from typing import List, Dict, Any

from app.state import AgentState, Evaulation
from app.node.llm_node import LLMNode
from app.util.logger import get_logger

logger = get_logger(__name__)

class Evaluator(LLMNode):
    def __init__(self, llm: BaseChatModel, prompt_file: str = None):
        super().__init__("evaluator", llm, prompt_file)
        
    async def evaluator_node(self, state: AgentState) -> Dict[str, Any]:
        """
        결과 평가 및 다음 액션 결정

        현재까지 수집된 정보를 평가하여:
        1. 충분한 경우 (SUFFICIENT) : synthesizer로 이동, 최종 답변 생성
        2. 부족한 경우 (REPLAN) : replan으로 이동
        3. 사용자의 추가 정보가 필요한 경우 (CLARIFY) : synthesizer로 이동, 사용자 추가 질의 수집 답변 제공
        
        평가 결과 State에 저장
        Args:
            state: 현재 에이전트 상태 (query 필드 포함)
        
        Returns:
            평가 결과를 포함한 부분 State 업데이트
        """
        
        user_query = state['query']
        agent_results = state["agent_results"]
        
        agent_summary = '\n'.join(
            f"{agent_result.name} : {agent_result.result}" if agent_result.success else f"[Error] {agent_result.name} : {agent_result.result}"
            for agent_result in agent_results
        )
        
        logger.info(f"[EVALUATOR] 계획 완료 후 평가중...")
        logger.debug(f"에이전트 결과: {agent_summary}")
        
        try:
            structured_llm = self.llm.with_structured_output(Evaulation)
            plan_chain = self.prompt_template | structured_llm 
            
            evaluation = await plan_chain.ainvoke({
                "user_query" : user_query,
                "agent_summary" : agent_summary
            })
            
            logger.info(f"[EVALUATOR] 평가 완료 - {evaluation.status}, 평가 이유: {evaluation.reasoning}")
        
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"[EVALUATOR] 평가 스키마 검증 실패: {str(e)}")
            evaluation = self._create_fallback_evaluation(user_query, str(e))
            
        except Exception as e:
            logger.error(f"[EVALUATOR] 평가 예외 발생: {str(e)}")
            evaluation = self._create_fallback_evaluation(str(e))

        return {
                "messages" : [AIMessage(content=f"[EVALUATOR] 평가 [{evaluation.status} : {evaluation.reasoning}")], 
                "evaluation": evaluation
            }
    
    def _create_fallback_evaluation(self, error_message: str):
        return Evaulation(
            status='SUFFICIENT',
            reasoning=f"평가 중 예외 발생으로 답변 생성 : {error_message}"
        )