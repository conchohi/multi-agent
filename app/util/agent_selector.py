"""
Task를 분석하여 최적의 Agent를 선택하는 클래스
"""
import json
import random
from typing import Dict, List, Optional
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

from app.node.sub.agent_node import AgentNode
from app.util.logger import get_logger

logger = get_logger(__name__)


class AgentSelection(BaseModel):
    """LLM이 선택한 agent와 reasoning"""
    agent: str = Field(..., description="선택된 에이전트 이름")
    reasoning: str = Field(..., description="이 에이전트를 선택한 이유")

class AgentSelector:
    """
    Task를 분석하여 최적의 Agent를 선택하는 클래스

    Responsibilities:
    - Task와 user query를 분석하여 적절한 agent 선택
    - LLM 기반 선택 실패 시 폴백 전략 적용
    - Agent 정보 포맷팅
    """

    def __init__(
        self, 
        llm: BaseChatModel, 
        agent_nodes: List[AgentNode], 
        prompt_template: ChatPromptTemplate, 
        fallback_strategy: str = "first"  # "first" | "random" | "round_robin"
    ):
        self.llm = llm
        self.agent_nodes = agent_nodes
        self.agent_nodes_map = {node.name: node for node in agent_nodes}
        self.prompt_template = prompt_template
        self.fallback_strategy = fallback_strategy
        self._round_robin_index = 0

        # Agent 정보를 초기화 시점에 한 번만 생성
        self.agent_info = self._build_agent_info()

    async def select_agent(self, task: str, user_query: str) -> str:
        """
        Task를 분석하여 최적의 agent 선택

        Args:
            task: 수행할 업무
            user_query: 원본 사용자 쿼리

        Returns:
            선택된 agent 이름
        """
        try:
            structured_llm = self.llm.with_structured_output(AgentSelection)
            selection_chain = self.prompt_template | structured_llm

            result = await selection_chain.ainvoke({
                "task": task,
                "user_query": user_query,
                "agent_info": self.agent_info  # 인스턴스 변수 사용
            })

            # 선택된 agent가 실제로 존재하는지 검증
            if result.agent in self.agent_nodes_map:
                logger.debug(
                    f"[AGENT_SELECTOR] Task '{task}...' → {result.agent} (이유: {result.reasoning})"
                )
                return result.agent
            else:
                logger.warning(
                    f"[AGENT_SELECTOR] 선택된 agent '{result.agent}'이 존재하지 않음. "
                    f"폴백 전략 적용"
                )
                return self._apply_fallback_strategy()

        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"[AGENT_SELECTOR] Agent 선택 스키마 검증 실패: {str(e)}")
            return self._apply_fallback_strategy()

        except Exception as e:
            logger.error(f"[AGENT_SELECTOR] Agent 선택 예외 발생: {str(e)}")
            return self._apply_fallback_strategy()

    def _apply_fallback_strategy(self) -> str:
        """
        폴백 전략에 따라 기본 agent 선택

        Args:
            task: 수행할 업무 (로깅용)

        Returns:
            선택된 agent 이름
        """
        if self.fallback_strategy == "first":
            selected = self.agent_nodes[0].name
        elif self.fallback_strategy == "round_robin":
            selected = self.agent_nodes[self._round_robin_index].name
            self._round_robin_index = (self._round_robin_index + 1) % len(self.agent_nodes)
        elif self.fallback_strategy == "random":
            selected = random.choice(self.agent_nodes).name
        else:
            selected = self.agent_nodes[0].name

        logger.info(
            f"[AGENT_SELECTOR] 폴백 전략 '{self.fallback_strategy}' 적용 → {selected}"
        )
        return selected

    def get_agent_node(self, agent_name: str) -> AgentNode:
        """Agent 이름으로 AgentNode 인스턴스 조회"""
        return self.agent_nodes_map.get(agent_name)

    def _build_agent_info(self) -> str:
        """
        Agent 정보 구성 (설명 + 실제 도구 목록)

        Returns:
            포맷팅된 agent 정보 문자열
        """
        info_lines = []
        for agent_node in self.agent_nodes:
            all_tools = agent_node.mcp_tools + agent_node.base_tools

            if all_tools:
                tools_str = "\n    ".join([
                    f"- {tool.name}: {tool.description or '설명 없음'}"
                    for tool in all_tools
                ])
                info_lines.append(
                    f"- {agent_node.name}: {agent_node.description}\n"
                    f"  사용 가능한 도구:\n    {tools_str}"
                )
            else:
                info_lines.append(
                    f"- {agent_node.name}: {agent_node.description}\n"
                    f"  사용 가능한 도구: 없음 (대화만 가능)"
                )

        return "\n\n".join(info_lines)
