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
    def __init__(self, llm: BaseChatModel, agent: AgentConfig, mcp_configs: Dict[str, McpConfig] = {}, tools: List[BaseTool] = []):
        super().__init__(name=agent.name, llm=llm, prompt_file=agent.prompt_file)
        self.description = agent.description
        self.base_tools: List[BaseTool] = tools if tools else []
        self.mcp_tools: List[BaseTool] = []
        self.mcp_configs: Dict[str, Optional[McpConfig]] = {
            mcp_server: mcp_configs.get(mcp_server)
            for mcp_server in agent.mcp_servers
        } if mcp_configs else {}
        self.mcp_client: Optional[MultiServerMCPClient] = None
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
        mcp_tools = await self.mcp_client.get_tools()
        self.mcp_tools = mcp_tools

        logger.info(f"  📦 Loaded {len(mcp_tools)} MCP tools")
    
    async def execute(self, state: SubAgentState) -> Dict[str, Any]:
        if not self.mcp_client:
            await self.initialize()
            
        task = state['task'] # 수행할 업무
        query = state['query']  # 전체 컨텍스트
        agent_name = state['agent_name']
        step_number = state.get('step_number', 0)  # 단계 번호
        
        agent_results = state["agent_results"]

        # 최신 3건만 가져오기
        recent_agent_results = agent_results[-3:] if agent_results else []

        agent_summary = '\n'.join(
            f"{agent_result.name} : {agent_result.result}"
            for agent_result in recent_agent_results
        ) if recent_agent_results else "No previous agent execution"
        
        logger.info(f"[{self.name}] 태스크 실행 : {task}")
        # 작업 수행
        try:
            # ReAct agent로 작업 실행
            all_tools = self.base_tools + self.mcp_tools
            agent_graph = create_react_agent_graph(
                llm=self.llm,
                tools=all_tools,
                system_prompt=self.prompt
            )
            
            message = f"User Query : {query}\nTask : {task}\nAgent Results : {agent_summary}"
            
            graph_result = await agent_graph.ainvoke({
                "messages": [HumanMessage(message)]
            })
            
            result = graph_result['messages'][-1].content
            
            # 성공 결과를 AgentResult로 반환
            # operator.add에 의해 메인 State의 agent_results에 추가됨
            return {
                "messages" : [AIMessage(f"[{self.name}] : {result}")],
                "agent_results": [AgentResult(
                    name=agent_name,
                    task=task,
                    result=result,
                    step_number=step_number,
                    success=True
                )]
            }
            
        except Exception as e:
            # 실패 결과 반환
            return {
                "messages" : [AIMessage(f"[{self.name}] 요청 예외 발생 : {str(e)}")],
                "agent_results": [AgentResult(
                    name=agent_name,
                    task=task,
                    result=f"요청 예외 발생 {str(e)}",
                    step_number=step_number,
                    success=False
                )]
            }
