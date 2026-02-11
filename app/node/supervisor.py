"""
계획에 기반하여 task를 분석하고 적절한 agent를 선택하여 라우팅
Supervisor는 에이전트를 선택하고 task_assignments에 저장하는 라우터 역할
"""
import json
import random
from typing import List, Dict, Any, Optional
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field, ValidationError

from app.node.llm_node import LLMNode
from app.node.sub.agent_node import AgentNode
from app.state import AgentState
from app.util.logger import get_logger

logger = get_logger(__name__)


class AgentSelection(BaseModel):
    """LLM이 선택한 agent와 reasoning"""
    agent: str = Field(..., description="선택된 에이전트 이름")
    reasoning: str = Field(..., description="이 에이전트를 선택한 이유")


class Supervisor(LLMNode):
    def __init__(
        self,
        llm: BaseChatModel,
        agent_nodes: List[AgentNode],
        prompt_file: str = None,
        fallback_strategy: str = "first"
    ):
        super().__init__("supervisor", llm, prompt_file)
        self.agent_nodes = agent_nodes
        self.agent_nodes_map: Dict[str, AgentNode] = {}
        self._initialized = False
        self.fallback_strategy = fallback_strategy
        self._round_robin_index = 0
        self.agent_info: str = ""

    async def initialize(self):
        """각 AgentNode의 MCP tools 초기화"""
        if self._initialized:
            return

        logger.info("[SUPERVISOR] Agent 노드 초기화 중...")

        for agent_node in self.agent_nodes:
            await agent_node.initialize()
            tools_count = len(agent_node.mcp_tools) + len(agent_node.base_tools)
            logger.info(f"[SUPERVISOR] {agent_node.name}: {tools_count}개 도구 준비됨")

        self.agent_nodes_map = {node.name: node for node in self.agent_nodes}
        self.agent_info = self._build_agent_info()
        self._initialized = True

    async def supervisor_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Plan의 step을 분석하여 적절한 에이전트를 선택하고 task_assignments에 저장

        - sequential 모드: current_step에 해당하는 step 하나만 선택
        - parallel 모드: 모든 step을 동시에 선택

        Args:
            state: AgentState

        Returns:
            task_assignments가 포함된 state 업데이트
        """
        if not self._initialized:
            await self.initialize()

        execution_plan = state.get('plan')
        steps = execution_plan.steps
        execution_mode = execution_plan.execution_mode
        user_query = state.get('query')
        agent_results = state.get('agent_results', [])

        logger.info(f"[SUPERVISOR] {execution_mode} 모드, {len(steps)}개 step 에이전트 선택 시작")

        task_assignments = []

        if execution_mode == 'parallel':
            prev_assignments = state.get('task_assignments', [])
            
            if not prev_assignments:
                # 병렬 실행: 모든 step의 에이전트를 한 번에 선택
                for i, step in enumerate(steps):
                    agent_name = await self._select_agent(task=step.task, user_query=user_query)
                    logger.info(f"[SUPERVISOR] 병렬 Step {i + 1}/{len(steps)} → {agent_name}: {step.task[:60]}")
                    task_assignments.append({
                        "agent_name": agent_name,
                        "task": step.task,
                        "step_number": step.step_number,
                        "agent_results": agent_results[:-3]
                    })
        else:
            # 순차 실행: current_step에 해당하는 step 하나만 선택
            current_step = state.get('current_step', 0)
            if current_step < len(steps):
                step = steps[current_step]
                agent_name = await self._select_agent(task=step.task, user_query=user_query)
                logger.info(f"[SUPERVISOR] 순차 Step {current_step + 1}/{len(steps)} → {agent_name}: {step.task[:60]}")
                task_assignments.append({
                    "agent_name": agent_name,
                    "task": step.task,
                    "step_number": step.step_number,
                    "agent_results": agent_results[:-3]
                })

        return {"task_assignments": task_assignments}

    async def _select_agent(self, task: str, user_query: str) -> str:
        """
        Task를 분석하여 최적의 agent 선택 (LLM 기반, 실패 시 폴백)
        """
        try:
            structured_llm = self.llm.with_structured_output(AgentSelection)
            selection_chain = self.prompt_template | structured_llm

            result = await selection_chain.ainvoke({
                "task": task,
                "user_query": user_query,
                "agent_info": self.agent_info
            })

            if result.agent in self.agent_nodes_map:
                logger.debug(f"[SUPERVISOR] '{task[:40]}...' → {result.agent} (이유: {result.reasoning})")
                return result.agent
            else:
                logger.warning(f"[SUPERVISOR] 선택된 agent '{result.agent}'이 존재하지 않음, 폴백 적용")
                return self._apply_fallback()

        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"[SUPERVISOR] Agent 선택 스키마 검증 실패: {str(e)}")
            return self._apply_fallback()

        except Exception as e:
            logger.error(f"[SUPERVISOR] Agent 선택 예외 발생: {str(e)}")
            return self._apply_fallback()

    def _apply_fallback(self) -> str:
        """폴백 전략에 따라 기본 agent 선택"""
        if self.fallback_strategy == "round_robin":
            selected = self.agent_nodes[self._round_robin_index].name
            self._round_robin_index = (self._round_robin_index + 1) % len(self.agent_nodes)
        elif self.fallback_strategy == "random":
            selected = random.choice(self.agent_nodes).name
        else:  # "first"
            selected = self.agent_nodes[0].name

        logger.info(f"[SUPERVISOR] 폴백 '{self.fallback_strategy}' 적용 → {selected}")
        return selected

    def _build_agent_info(self) -> str:
        """Agent 정보 구성 (설명 + 실제 도구 목록)"""
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
