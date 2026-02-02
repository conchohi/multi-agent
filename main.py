import argparse
import os
from pathlib import Path

from dotenv import load_dotenv
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.util.logger import setup_logging, get_logger
from app.util.config_loader import ConfigLoader, get_config
from app.util.llm_builder import build_llm
from app.util.checkpointer_factory import CheckpointerFactory
from app.graph import create_graph
from app.api import routes

load_dotenv()

# 설정 파일 경로 초기화
config_path = os.environ.get("CHATBOT_CONFIG_PATH", "config/config.yaml")
if not ConfigLoader.is_initialized():
    ConfigLoader.initialize(config_path)

config = get_config()
api_config = config.get("api", {})
cors_config = config.get("cors", {})

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    애플리케이션 생명주기 관리

    시작 시:
    - 설정 로드
    - MCP 클라이언트 초기화
    - Supervisor 그래프 생성

    종료 시:
    - MCP 클라이언트 정리
    """
    
    # reload 모드에서 워커 프로세스 초기화 대응
    if not ConfigLoader.is_initialized():
        config_path = os.environ.get("CHATBOT_CONFIG_PATH", "config/config.yaml")
        ConfigLoader.initialize(config_path)

    config = get_config()
    log_config = config.get("logging", {})
    
    setup_logging(
        level=log_config.get('level', 'INFO'),
        log_file=log_config.get('file', './logs/app.log'),
        enable_file_logging=log_config.get('enable_file_logging', False)
    )
    
    logger = get_logger(__name__)
    
    logger.info("Starting application...")

    logger.info("Configuration loaded")

    # LLM 인스턴스 생성
    llm = build_llm()
    logger.info("LLM initialized")
    
    # checkpointer 인스턴스 생성
    checkpointer_type = config.get('session', {}).get('storage', 'memory')
    redis_url = config.get('redis', {}).get('url', None)
    checkpointer = CheckpointerFactory.create(checkpointer_type, redis_url)

    # 그래프 생성
    graph = await create_graph(llm, checkpointer)
    logger.info("Graph created")

    # API 라우터에 그래프 인스턴스 설정
    routes.set_graph(graph)

    logger.info("Application started successfully")

    yield  # 애플리케이션 실행

    # 종료 시 정리
    logger.info("Shutting down application...")
    logger.info("Application shut down")


# FastAPI 앱 생성 (모듈 레벨)
app = FastAPI(
    title="LangGraph Multi-Agent API",
    description="LangGraph 기반 멀티에이전트 시스템",
    version="0.1.0",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_config.get("allow_origins", []),
    allow_credentials=cors_config.get("allow_credentials", False),
    allow_methods=cors_config.get("allow_methods", []),
    allow_headers=cors_config.get("allow_headers", []),
    max_age=cors_config.get("max_age", 600)
)

# 라우터 등록
app.include_router(routes.router)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangGraph Multi-Agent 서버")

    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="메인 설정 파일 경로 (기본값: config/config.yaml)"
    )

    args = parser.parse_args()

    # reload 모드 워커에 경로 전달을 위해 환경변수 저장
    os.environ["CHATBOT_CONFIG_PATH"] = str(Path(args.config).resolve())

    # 설정 재로드 (커스텀 config 경로를 사용하는 경우)
    if args.config != "config/config.yaml":
        ConfigLoader.initialize(args.config)
        config = get_config()
        api_config = config.get("api", {})

    # 서버 실행
    uvicorn.run(
        "main:app",
        host=api_config.get("host", "0.0.0.0"),
        port=api_config.get("port", 8000),
        reload=api_config.get("reload", True)
    )
