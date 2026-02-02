"""
모든 Step 처리 후 최종 답변 처리
"""

from typing import List, Dict, Any
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import StrOutputParser

from app.node.llm_node import LLMNode
from app.state import AgentState
from app.util.logger import get_logger

logger = get_logger(__name__)

class Synthesizer(LLMNode):
    def __init__(self, llm: BaseChatModel, prompt_file: str = None):
        super().__init__("synthesizer", llm, prompt_file)
        
    async def synthesizer_node(self, state: AgentState) -> Dict[str, Any]:
        """
        최종 답변 생성

        모든 에이전트의 결과를 통합하여 최종 답변을 생성합니다.

        Args:
            state: 현재 에이전트 상태 (query 필드 포함)

        Returns:
            최종 답변을 포함한 부분 State
        """
        
        user_query = state['query']
        agent_results = state["agent_results"]
        evaluation = state['evaluation']
        missing_info = evaluation.missing_info if evaluation.missing_info else ""
        
        status = evaluation.status
        
        agent_summary = '\n'.join(
            f"{agent_result.name} : {agent_result.result}" if agent_result.success else f"[Error] {agent_result.name} : {agent_result.result}"
            for agent_result in agent_results
        )
        
        logger.info(f"[SYNTHESIZER] 최종 답변 생성중...")
        
        try:
            synthesizer_chain = self.prompt_template | self.llm | StrOutputParser()
            
            final_answer  = await synthesizer_chain.ainvoke({
                "user_query" : user_query,
                "agent_summary" : agent_summary,
                "status" : status,
                "missing_info" : missing_info
            })
            logger.info(f"[SYNTHESIZER] 최종 답변 생성 : {final_answer[:10]}")
            
        except Exception as e:
            logger.error(f"[SYNTHESIZER] 최종 답변 생성 중 예외 발생: {str(e)}")
            final_answer = f"답변 생성 중 예외 발생했습니다. 서버 로그를 확인하세요."
            
        return {
            "messages" : [AIMessage(content=f"[SYNTHESIZER] 최종 답변 : {final_answer}")],
            "final_answer" : final_answer,
            "conversation_histories" : [f"USER : {user_query}", f"AI : {final_answer}"]
        }