"""
계획에 기반하여 task를 분석하고 적절한 agent를 선택하여 실행
Supervisor는 모든 step을 내부적으로 처리하는 단일 노드
"""
import json
import asyncio
from typing import List, Dict, Any
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field, ValidationError

from app.node.llm_node import LLMNode
from app.node.sub.agent_node import AgentNode
from app.state import AgentState, AgentResult, Step
from app.util.logger import get_logger

logger = get_logger(__name__)

class AgentSelection(BaseModel):
    """LLM이 선택한 agent와 reasoning"""
    agent: str = Field(..., description="선택된 에이전트 이름")
    reasoning: str = Field(..., description="이 에이전트를 선택한 이유")

class Supervisor(LLMNode):
    def __init__(self, llm: BaseChatModel, agent_nodes: List[AgentNode], prompt_file: str = None):
        super().__init__("supervisor", llm, prompt_file)
        self.agent_nodes = agent_nodes
        self.agent_nodes_map = {node.name: node for node in agent_nodes}
        self._initialized = False
        self._agent_info = ""

    async def initialize(self):
        """각 AgentNode의 MCP tools 초기화"""
        if self._initialized:
            return

        logger.info("[SUPERVISOR] Agent 노드 초기화 중...")
        for agent_node in self.agent_nodes:
            await agent_node.initialize()
            tools_count = len(agent_node.mcp_tools) + len(agent_node.base_tools)
            logger.info(f"[SUPERVISOR] {agent_node.name}: {tools_count}개 도구 준비됨")
        
        # Agent 정보 구성 (설명 + 실제 사용 가능한 도구들)
        self._agent_info = self._build_agent_info()
        
        self._initialized = True

    async def supervisor_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Plan의 모든 step을 내부적으로 처리

        각 step마다:
        1. task 분석하여 최적의 agent 선택
        2. 선택된 agent로 task 실행
        3. 결과를 agent_results에 누적

        Args:
            state: AgentState

        Returns:
            모든 step 실행 완료 후 agent_results
        """
        if not self._initialized:
            await self.initialize()

        execution_plan = state.get('plan')
        steps = execution_plan.steps
        execution_mode = execution_plan.execution_mode
        user_query = state.get('query')
        agent_results = state.get('agent_results', [])

        logger.info(f"[SUPERVISOR] {execution_mode} 모드로 {len(steps)}개 step 실행 시작")

        if execution_mode == 'sequential':
            # 순차 실행
            new_results = await self._execute_sequential(steps, user_query, agent_results)
        else:
            # 병렬 실행
            new_results = await self._execute_parallel(steps, user_query, agent_results)

        logger.info(f"[SUPERVISOR] 모든 step 실행 완료, {len(new_results)}개 결과 반환")

        return {
            "agent_results": new_results
        }

    async def _execute_sequential(self, steps: List[Step], user_query: str, agent_results: List[AgentResult]) -> List[AgentResult]:
        """순차 실행: step을 하나씩 실행"""
        results = []
        for step in steps:
            # Agent 선택
            selected_agent_name = await self._select_agent_for_task(step.task, user_query)

            # Agent 실행
            result = await self._execute_agent(
                agent_name=selected_agent_name,
                task=step.task,
                step_number=step.step_number,
                user_query=user_query,
                agent_results=agent_results + results  # 누적된 결과 전달
            )

            results.append(result)

        return results

    async def _execute_parallel(self, steps: List, user_query: str, agent_results: List[AgentResult]) -> List[AgentResult]:
        """병렬 실행: 모든 step을 동시에 실행"""

        tasks = []
        for step in steps:
            task_coro = self._execute_step_with_selection(
                step=step,
                user_query=user_query,
                agent_results=agent_results
            )
            tasks.append(task_coro)

        results = await asyncio.gather(*tasks)
        return list(results)

    async def _execute_step_with_selection(self, step: Step, user_query: str, agent_results: List[AgentResult]) -> AgentResult:
        """step에 대해 agent 선택 후 실행"""
        selected_agent_name = await self._select_agent_for_task(step.task, user_query)
        return await self._execute_agent(
            agent_name=selected_agent_name,
            task=step.task,
            step_number=step.step_number,
            user_query=user_query,
            agent_results=agent_results
        )

    async def _execute_agent(
        self,
        agent_name: str,
        task: str,
        step_number: int,
        user_query: str,
        agent_results: List[AgentResult]
    ) -> AgentResult:
        """선택된 agent로 task 실행"""
        agent_node = self.agent_nodes_map.get(agent_name)

        if not agent_node:
            logger.error(f"[SUPERVISOR] Agent '{agent_name}'을 찾을 수 없음")
            return AgentResult(
                name=agent_name,
                task=task,
                result=f"Agent '{agent_name}'을 찾을 수 없습니다.",
                step_number=step_number,
                success=False
            )

        # SubAgentState 구성
        sub_state = {
            "agent_name": agent_name,
            "query": user_query,
            "task": task,
            "step_number": step_number,
            "agent_results": agent_results
        }

        # Agent 실행
        result_state = await agent_node.execute(sub_state)

        # AgentResult 추출
        return result_state["agent_result"]

    async def _select_agent_for_task(self, task: str, user_query: str) -> str:
        """
        task를 분석하여 최적의 agent 선택

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
                "agent_info": self._agent_info
            })

            logger.debug(f"[SUPERVISOR] Task '{task}...' → {result.agent} (이유: {result.reasoning})")
            return result.agent

        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"[SUPERVISOR] Agent 선택 실패: {str(e)}, 기본 agent 사용")
            return self.agent_nodes[0].name

        except Exception as e:
            logger.error(f"[SUPERVISOR] Agent 선택 예외: {str(e)}, 기본 agent 사용")
            return self.agent_nodes[0].name

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
            