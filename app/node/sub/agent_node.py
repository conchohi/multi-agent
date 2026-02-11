from typing import List, Dict, Optional, Any

from langchain_core.tools.base import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import HumanMessage, AIMessage

from app.node.llm_node import LLMNode
from app.model.settings import AgentConfig, McpConfig
from app.state import SubAgentState, AgentResult
from app.util.react_agent_builder import create_react_agent_graph
from app.util.logger import get_logger

logger = get_logger(__name__)

class AgentNode(LLMNode):
    def __init__(
        self,
        llm: BaseChatModel,
        agent: AgentConfig,
        mcp_configs: Dict[str, McpConfig] = {},
        tools: List[BaseTool] = [],
        max_recent_results: int = 3  # 설정 가능한 제한값
    ):
        super().__init__(name=agent.name, llm=llm, prompt_file=agent.prompt_file)
        self.description = agent.description
        self.base_tools: List[BaseTool] = tools if tools else []
        self.mcp_tools: List[BaseTool] = []
        self.mcp_configs: Dict[str, Optional[McpConfig]] = {
            mcp_server: mcp_configs.get(mcp_server)
            for mcp_server in agent.mcp_servers
        } if mcp_configs else {}
        self.mcp_client: Optional[MultiServerMCPClient] = None
        self.max_recent_results = max_recent_results
        self._initialized = False
        
    async def initialize(self):
        """Initialize MCP client and load tools."""
        if self._initialized or not self.mcp_configs:
            return

        mcp_connections = {}
        # Connect to each server
        for server_name, mcp_config in self.mcp_configs.items():
            if mcp_config:
                if mcp_config.transport == 'stdio':
                    mcp_connections[mcp_config.name] = {
                        "transport" : mcp_config.transport,
                        "command" : mcp_config.command,
                        "args" : mcp_config.args,
                        "env" : mcp_config.env,
                    }
                else :
                    mcp_connections[mcp_config.name] = {
                        "transport" : mcp_config.transport,
                        "url" : mcp_config.url,
                        "headers" : mcp_config.headers
                    }
            else :
                logger.warning(f"[{self.name}] {server_name} MCP 서버 설정이 존재하지 않습니다. MCP 서버 설정 파일을 확인하세요.")
            
        # Create MultiServerMCPClient
        self.mcp_client = MultiServerMCPClient(mcp_connections)

        # Load tools
        await self._load_mcp_tools()
        self._initialized = True
    
    async def _load_mcp_tools(self):
        """Load all tools from MCP servers."""
        if not self.mcp_client:
            return

        # Use MultiServerMCPClient's get_tools method
        self.mcp_tools = await self.mcp_client.get_tools()
        logger.info(f"  📦 Loaded {len(self.mcp_tools)} MCP tools")
    
    async def execute(self, state: SubAgentState) -> Dict[str, Any]:
        """
        Agent 실행

        Args:
            state: SubAgentState

        Returns:
            agent_result를 포함한 딕셔너리
        """
        if not self.mcp_client and self.mcp_configs:
            await self.initialize()

        # State에서 필드 추출
        task = state['task']
        query = state['query']
        agent_name = state['agent_name']
        step_number = state.get('step_number', 0)
        agent_results = state.get("agent_results", [])

        logger.info(f"[{self.name}] 태스크 실행: {task}")

        try:
            # Agent 메시지 구성
            message = self._build_agent_message(query, task, agent_results)

            # ReAct agent 실행
            all_tools = self.base_tools + self.mcp_tools

            agent_graph = create_react_agent_graph(
                llm=self.llm,
                tools=all_tools,
                system_prompt=self.prompt
            )

            graph_result = await agent_graph.ainvoke({
                "messages": [HumanMessage(message)]
            })

            result = graph_result['messages'][-1].content
            
            # 성공 결과 반환
            return {
                "agent_result": AgentResult(
                    name=agent_name,
                    task=task,
                    result=result,
                    step_number=step_number,
                    success=True
                )
            }

        except Exception as e:
            logger.error(f"[{self.name}] 태스크 실행 예외 발생: {str(e)}")
            # 실패 결과 반환
            return {
                "agent_result": AgentResult(
                    name=agent_name,
                    task=task,
                    result=f"태스크 실행 중 예외 발생: {str(e)}",
                    step_number=step_number,
                    success=False
                )
            }

    def _build_agent_message(
        self,
        query: str,
        task: str,
        agent_results: List[AgentResult]
    ) -> str:
        """
        Agent에 전달할 메시지 구성

        Args:
            query: 사용자 쿼리
            task: 실행할 작업
            agent_results: 이전 결과들

        Returns:
            포맷팅된 메시지
        """
        # 최신 N건만 가져오기 (설정 가능)
        recent_agent_results = agent_results[-self.max_recent_results:] if agent_results else []

        agent_summary = '\n'.join(
            f"{agent_result.name}: {agent_result.result}"
            for agent_result in recent_agent_results
        ) if recent_agent_results else "No previous agent execution"

        return f"User Query: {query}\nTask: {task}\nAgent Results: {agent_summary}"
