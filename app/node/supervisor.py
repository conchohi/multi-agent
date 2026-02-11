"""
계획에 기반하여 task를 분석하고 적절한 agent를 선택하여 실행
Supervisor는 모든 step을 내부적으로 처리하는 단일 노드
"""
import asyncio
from typing import List, Dict, Any
from langchain_core.language_models import BaseChatModel

from app.node.llm_node import LLMNode
from app.node.sub.agent_node import AgentNode
from app.state import AgentState, AgentResult, Step
from app.util.logger import get_logger
from app.util.agent_selector import AgentSelector

logger = get_logger(__name__)

class Supervisor(LLMNode):
    def __init__(
        self, llm: BaseChatModel, agent_nodes: List[AgentNode], prompt_file: str = None, fallback_strategy: str = "first"
    ):
        super().__init__("supervisor", llm, prompt_file)
        self.agent_nodes = agent_nodes
        self._initialized = False
        self.fallback_strategy = fallback_strategy

        # AgentSelector 초기화 (initialize 후에 설정됨)
        self.agent_selector: AgentSelector = None

    async def initialize(self):
        """각 AgentNode의 MCP tools 초기화"""
        if self._initialized:
            return

        logger.info("[SUPERVISOR] Agent 노드 초기화 중...")

        # 각 AgentNode 초기화
        for agent_node in self.agent_nodes:
            await agent_node.initialize()
            tools_count = len(agent_node.mcp_tools) + len(agent_node.base_tools)
            logger.info(f"[SUPERVISOR] {agent_node.name}: {tools_count}개 도구 준비됨")

        # AgentSelector 초기화
        self.agent_selector = AgentSelector(
            llm=self.llm,
            agent_nodes=self.agent_nodes,
            prompt_template=self.prompt_template,
            fallback_strategy=self.fallback_strategy
        )

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

    async def _execute_sequential(
        self, steps: List[Step], user_query: str, agent_results: List[AgentResult]
    ) -> List[AgentResult]:
        """순차 실행: step을 하나씩 실행"""
        results = []

        for step in steps:
            result = await self._execute_step(
                step=step,
                user_query=user_query,
                agent_results=agent_results + results  # 누적된 결과 전달
            )
            results.append(result)

        return results

    async def _execute_parallel(
        self, steps: List[Step], user_query: str, agent_results: List[AgentResult]
    ) -> List[AgentResult]:
        """병렬 실행: 모든 step을 동시에 실행"""
        tasks = [
            self._execute_step(
                step=step,
                user_query=user_query,
                agent_results=agent_results
            )
            for step in steps
        ]

        results = await asyncio.gather(*tasks)
        return list(results)

    async def _execute_step(self,
        step: Step, user_query: str, agent_results: List[AgentResult]
    ) -> AgentResult:
        """
        단일 step 실행 (agent 선택 + 실행)

        Args:
            step: 실행할 단계
            user_query: 사용자 쿼리
            agent_results: 이전 결과들

        Returns:
            실행 결과
        """
        # Agent 선택 (AgentSelector에 위임)
        selected_agent_name = await self.agent_selector.select_agent(
            task=step.task,
            user_query=user_query
        )

        # Agent 실행
        return await self._execute_agent(
            agent_name=selected_agent_name,
            task=step.task,
            step_number=step.step_number,
            user_query=user_query,
            agent_results=agent_results
        )

    async def _execute_agent(self,
        agent_name: str,
        task: str,
        step_number: int,
        user_query: str,
        agent_results: List[AgentResult]
    ) -> AgentResult:
        """
        선택된 agent로 task 실행

        Args:
            agent_name: Agent 이름
            task: 실행할 작업
            step_number: 단계 번호
            user_query: 사용자 쿼리
            agent_results: 이전 결과들

        Returns:
            실행 결과
        """
        # AgentSelector를 통해 AgentNode 조회
        agent_node = self.agent_selector.get_agent_node(agent_name)

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
        try:
            result_state = await agent_node.execute(sub_state)
            return result_state["agent_result"]
        except Exception as e:
            logger.error(f"[SUPERVISOR] Agent 실행 예외 발생: {str(e)}")
            return AgentResult(
                name=agent_name,
                task=task,
                result=f"Agent 실행 중 예외 발생: {str(e)}",
                step_number=step_number,
                success=False
            )

