from typing import Optional, List, Dict, Literal
from pydantic import BaseModel, Field

class LLMConfig(BaseModel):
    provider: Literal['openai', 'ollama', 'anthropic'] = Field(default="openai", description="LLM 제공자 명칭")
    model: str = Field(..., description="LLM 모델")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0, description="생성 온도")
    max_tokens: Optional[int] = Field(default=8192, gt=0, description="생성할 최대 토큰 수")
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0, description="핵 샘플링 확률 (0.0-1.0)")
    frequency_penalty: Optional[float] = Field(default=1.0, ge=-2.0, le=2.0, description="토큰 반복 패널티 (-2.0 ~ 2.0)")

class McpConfig(BaseModel):
    name: str = Field(..., description="MCP 서버 이름")
    description: Optional[str] = Field(default=None, description="MCP 서버 설명 (선택)")
    enabled: bool = Field(default=True, description="MCP 서버 사용 여부")
    transport: Literal['http', 'stdio', 'sse', 'websocket', 'streamable-http', 'streamable_http'] = Field(..., description="MCP 서버 transport 타입")
    command: Optional[str] = Field(default=None, description="프로세스 기반 서버 실행 명령어")
    args: List[str] = Field(default=[], description="프로세스 실행 명령어")
    env: Dict[str, str] = Field(default={}, description="프로세스 실행을 위한 환경변수")
    url: Optional[str] = Field(default=None, description="HTTP 기반 MCP 서버 요청 URL")
    headers : Dict[str, str] = Field(default={}, description="HTTP 기반 MCP 서버 요청 헤더 (선택)")
    
class AgentConfig(BaseModel):
    name: str = Field(..., description="서브 에이전트 이름")
    description: str = Field(..., description="서브 에이전트 역할 설명") 
    enabled: bool = Field(default=True, description="서브 에이전트 사용 여부")
    prompt_file: Optional[str] = Field(default=None, description="서브 에이전트 프롬포트 설정 파일")
    mcp_servers: List[str] = Field(default=[], description="서브 에이전트 MCP 도구")
    llm: Optional[LLMConfig] = Field(default=None, description="서브 에이전트 별도 LLM 설정")