# LangGraph Multi-Agent Chatbot

LangGraph 기반의 지능형 멀티에이전트 챗봇 시스템입니다. 사용자 질의를 분석하여 동적으로 실행 계획을 수립하고, 여러 전문 에이전트를 조율하여 최적의 답변을 생성합니다.

## 주요 기능

- **동적 계획 수립**: 사용자 질의를 분석하여 최적의 실행 계획을 자동 생성
- **멀티에이전트 협업**: 채팅, 코드 생성, 검색, 날씨 등 전문화된 에이전트들의 협업
- **지능형 재계획**: 결과 평가 후 필요시 자동으로 계획을 재수립 (최대 3회)
- **병렬/순차 실행**: 작업 의존성에 따라 에이전트를 병렬 또는 순차적으로 실행
- **MCP 통합**: Model Context Protocol을 통한 확장 가능한 도구 시스템
- **세션 관리**: 대화 맥락을 유지하는 멀티턴 대화 지원
- **실시간 스트리밍**: SSE 기반 실시간 이벤트 스트리밍 지원

## 시스템 아키텍처

```
사용자 질의
    ↓
Planner (계획 수립)
    ↓
Supervisor (작업 라우팅)
    ↓
[에이전트 실행] → ChatAgent, CodeAgent, SearchAgent, WeatherAgent
    ↓
Evaluator (결과 평가)
    ↓
RePlanner (재계획) 또는 Synthesizer (최종 답변 생성)
    ↓
최종 답변
```

### 핵심 구성요소

1. **Planner**: 사용자 질의를 분석하여 단계별 실행 계획 생성
2. **Supervisor**: 계획에 따라 적절한 에이전트에게 작업 할당
3. **Sub-Agents**: 각 분야의 전문 에이전트 (채팅, 코드, 검색, 날씨)
4. **Evaluator**: 에이전트 실행 결과의 충분성 평가
5. **RePlanner**: 결과가 불충분할 경우 새로운 계획 수립
6. **Synthesizer**: 모든 결과를 종합하여 최종 답변 생성

## 설치 방법

### 1. 저장소 클론

```bash
git clone <repository-url>
cd chatbot
```

### 2. 의존성 설치

```bash
pip install -r requirements.txt
```

### 3. 환경 변수 설정

`.env.example` 파일을 `.env`로 복사하고 API 키를 설정합니다:

```bash
cp .env.example .env
```

`.env` 파일 편집:

```env
# OpenAI 사용 시
OPENAI_API_KEY=your_openai_api_key
OPENAI_API_BASE=your_openai_api_base

# Anthropic Claude 사용 시
ANTHROPIC_API_KEY=your_anthropic_api_key

# Ollama 사용 시 (로컬 모델)
OLLAMA_HOST=http://localhost:11434

# Session Storage Redis 사용 시
REDIS_URL=http://localhost:6379/0
```

### 4. 설정 파일 구성

#### config/config.yaml

```yaml
llm:
  provider: "openai" # openai, anthropic, ollama 중 선택
  model: "gpt-4"
  temperature: 0.7

session:
  storage: "memory" # memory 또는 redis

api:
  host: "0.0.0.0"
  port: 8000
  reload: true
```

#### config/agent_config.yaml

에이전트 정의 및 MCP 서버 할당

#### config/mcp_config.yaml

MCP 서버 연결 설정 (stdio 또는 SSE 방식)

## 실행 방법

### 기본 실행

```bash
python main.py
```

서버가 시작되면 http://localhost:8000 에서 API에 접근할 수 있습니다.

### 사용자 정의 설정 파일 사용

```bash
python main.py --config path/to/custom_config.yaml
```

## API 사용법

### 1. 세션 생성 (선택사항)

```bash
curl -X POST http://localhost:8000/api/session/create
```

응답:

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "created_at": "2024-02-02T10:00:00Z"
}
```

### 2. 일반 질의 (전체 응답)

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "오늘 서울 날씨 알려주고, 파이썬으로 간단한 계산기 코드 작성해줘",
    "session_id": "550e8400-e29b-41d4-a716-446655440000"
  }'
```

응답:

```json
{
  "query": "오늘 서울 날씨 알려주고...",
  "answer": "서울의 오늘 날씨는...",
  "plans": [
    {
      "reasoning": "날씨 정보와 코드 생성이 필요하므로...",
      "total_steps": 2,
      "execution_mode": "parallel",
      "use_agents": ["WeatherAgent", "CodeAgent"]
    }
  ],
  "agent_results": [
    {
      "name": "WeatherAgent",
      "task": "서울의 오늘 날씨 조회",
      "result": "서울 날씨: 맑음, 기온 15도",
      "step_number": 1,
      "success": true
    },
    {
      "name": "CodeAgent",
      "task": "파이썬 계산기 코드 작성",
      "result": "def calculator()...",
      "step_number": 2,
      "success": true
    }
  ],
  "success": true
}
```

### 3. 스트리밍 질의 (실시간 이벤트)

```bash
curl -X POST http://localhost:8000/api/query/stream \
  -H "Content-Type: application/json" \
  -d '{
    "query": "최신 AI 뉴스 검색해줘",
    "session_id": "550e8400-e29b-41d4-a716-446655440000"
  }'
```

SSE 이벤트 스트림:

```
data: {"event":"plan","node":"planner","data":{"reasoning":"...","total_steps":1}}

data: {"event":"agent_result","node":"SearchAgent","data":{"name":"SearchAgent","result":"..."}}

data: {"event":"final_answer","node":"synthesizer","data":{"answer":"최신 AI 뉴스..."}}

data: {"event":"done"}
```

### 4. 헬스 체크

```bash
curl http://localhost:8000/api/health
```

## 에이전트 추가하기

### 1. 에이전트 설정 추가

`config/agent_config.yaml`에 새 에이전트 정의:

```yaml
agents:
  - name: "TranslationAgent"
    description: "텍스트를 번역하는 에이전트, 보유 MCP 서버: translation_mcp"
    enabled: true
    mcp_servers:
      - translation_mcp
```

### 2. 프롬프트 파일 생성

`prompts/TranslationAgent_prompt.txt` 파일 생성:

```
당신은 전문 번역가입니다.
주어진 텍스트를 정확하고 자연스럽게 번역합니다.
...
```

### 3. State 타입 업데이트

`app/state.py`의 Step 클래스에 새 에이전트 추가:

```python
class Step(BaseModel):
    agent: Literal['ChatAgent', 'CodeAgent', 'SearchAgent', 'WeatherAgent', 'TranslationAgent']
    ...
```

### 4. MCP 서버 설정 (필요 시)

`config/mcp_config.yaml`에 MCP 서버 설정 추가:

```yaml
servers:
  translation_mcp:
    transport: "stdio"
    command: "python"
    args: ["mcp/translation.py"]
```

시스템이 자동으로 새 에이전트를 로드하고 그래프에 추가합니다.

## 프로젝트 구조

```
chatbot/
├── app/
│   ├── api/                  # FastAPI 라우터 및 스키마
│   │   ├── routes.py        # API 엔드포인트
│   │   └── schemas.py       # Pydantic 모델
│   ├── node/                # LangGraph 노드
│   │   ├── plan.py          # Planner 노드
│   │   ├── supervisor.py    # Supervisor 노드
│   │   ├── evaluator.py     # Evaluator 노드
│   │   ├── replan.py        # RePlanner 노드
│   │   ├── synthesizer.py   # Synthesizer 노드
│   │   └── sub/             # 서브 에이전트
│   │       ├── agent_node.py      # AgentNode 클래스
│   │       └── agent_factory.py   # 에이전트 팩토리
│   ├── util/                # 유틸리티
│   │   ├── config_loader.py       # 설정 로더
│   │   ├── llm_builder.py         # LLM 인스턴스 빌더
│   │   ├── checkpointer_factory.py # 체크포인터 팩토리
│   │   ├── react_agent_builder.py  # ReAct 에이전트 빌더
│   │   └── logger.py              # 로깅 설정
│   ├── graph.py             # LangGraph 워크플로우 정의
│   ├── state.py             # State 스키마 정의
│   └── model/               # 데이터 모델
│       └── settings.py      # 설정 모델
├── config/                  # 설정 파일
│   ├── config.yaml         # 메인 설정
│   ├── agent_config.yaml   # 에이전트 설정
│   └── mcp_config.yaml     # MCP 서버 설정
├── prompts/                # 에이전트 프롬프트
│   ├── plan_prompt.txt
│   ├── evaluator_prompt.txt
│   ├── ChatAgent_prompt.txt
│   └── ...
├── mcp/                    # MCP 서버 구현
├── logs/                   # 로그 파일
├── main.py                 # 애플리케이션 엔트리포인트
├── requirements.txt        # Python 의존성
├── .env.example           # 환경 변수 템플릿
└── README.md              # 이 파일
```

## 기술 스택

- **LangGraph 1.0.7**: 에이전트 워크플로우 오케스트레이션
- **LangChain 1.2.7**: LLM 통합 및 체인 구성
- **FastAPI 0.128.0**: REST API 서버
- **Pydantic 2.12.5**: 데이터 검증 및 스키마
- **MCP 1.26.0**: Model Context Protocol 통합
- **Redis**: 세션 영속화 (선택사항)
- **Uvicorn**: ASGI 서버

## 세션 관리

### 메모리 기반 (기본)

```yaml
session:
  storage: "memory"
```

서버 재시작 시 세션이 초기화됩니다.

### Redis 기반 (영속성)

```yaml
session:
  storage: "redis"

redis:
  url: "redis://localhost:6379"
```

Redis를 사용하면 서버 재시작 후에도 대화 맥락이 유지됩니다.

## 로깅

로그 설정은 `config/config.yaml`에서 관리:

```yaml
logging:
  level: "INFO" # DEBUG, INFO, WARNING, ERROR
  file: "logs/app.log"
  enable_file_logging: false # true로 설정 시 파일에 로그 저장
```

## 문제 해결

### MCP 서버 연결 실패

```
[AgentName] server_name MCP 서버 설정이 존재하지 않습니다.
```

- `config/mcp_config.yaml`에 서버 설정이 있는지 확인
- `config/agent_config.yaml`의 `mcp_servers` 항목과 이름이 일치하는지 확인

### LLM API 오류

```
OpenAI API key not found
```

- `.env` 파일에 올바른 API 키가 설정되어 있는지 확인
- `config/config.yaml`의 `llm.provider` 설정이 올바른지 확인

### 재계획 반복 (Replan loop)

시스템은 무한 루프를 방지하기 위해 최대 3회까지만 재계획을 수행합니다. 3회 초과 시 현재 결과로 답변을 생성합니다.

## 개발 모드

개발 시 Hot Reload 활성화:

```yaml
api:
  reload: true
```

코드 변경 시 자동으로 서버가 재시작됩니다.

## 라이선스

이 프로젝트의 라이선스는 별도로 명시되지 않았습니다.
