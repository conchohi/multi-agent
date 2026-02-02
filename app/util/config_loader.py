import os
import re
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional

from app.model.settings import AgentConfig, McpConfig


class ConfigLoader:
    """
    싱글턴 패턴으로 설정을 로드하고 관리하는 클래스.

    메인 config.yaml을 먼저 로드한 후, 내부의 config.agent / config.mcp 경로를
    기반으로 서브 config를 자동으로 로드합니다.
    서브 config 경로는 메인 config 파일의 디렉토리를 기준으로 해석됩니다.
    """

    _instance: Optional["ConfigLoader"] = None

    def __init__(self, config_path: str):
        self._config_path = Path(config_path).resolve()
        self._config_dir = self._config_path.parent
        self.config: Dict[str, Any] = self._load_yaml(self._config_path)
        self.agent_config: List[AgentConfig] = self._load_agent_config()
        self.mcp_config: Dict[str, McpConfig] = self._load_mcp_config()

    @classmethod
    def initialize(cls, config_path: str) -> "ConfigLoader":
        """싱글턴 인스턴스를 생성하고 반환"""
        cls._instance = cls(config_path)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "ConfigLoader":
        """초기화된 싱글턴 인스턴스를 반환"""
        if cls._instance is None:
            raise RuntimeError(
                "ConfigLoader가 초기화되지 않았습니다. initialize()를 먼저 호출하세요."
            )
        return cls._instance

    @classmethod
    def is_initialized(cls) -> bool:
        return cls._instance is not None

    def _expand_env_vars(self, config: Any) -> Any:
        """
        재귀적으로 환경 변수 치환

        ${VAR_NAME} 형식의 문자열을 환경 변수 값으로 치환합니다.
        환경 변수가 없으면 원본 문자열 그대로 유지합니다.
        """
        if isinstance(config, dict):
            return {k: self._expand_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._expand_env_vars(item) for item in config]
        elif isinstance(config, str):
            # ${VAR_NAME} 형식을 환경 변수로 치환
            pattern = r'\$\{([^}]+)\}'
            return re.sub(
                pattern,
                lambda m: os.getenv(m.group(1), m.group(0)),
                config
            )
        return config

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return self._expand_env_vars(data)

    def _load_agent_config(self) -> List[AgentConfig]:
        agent_path = self.config.get("config", {}).get("agent")
        if not agent_path:
            agent_path = self._config_dir / "agent_config.yaml"
        data = self._load_yaml(agent_path)
        return [AgentConfig(**agent) for agent in data.get("agents", [])]

    def _load_mcp_config(self) -> Dict[str, McpConfig]:
        mcp_path = self.config.get("config", {}).get("mcp")
        if not mcp_path:
            mcp_path = self._config_dir / "mcp_config.yaml"
        data = self._load_yaml(mcp_path)
        result = {}
        for name, server in (data.get("mcp_severs") or {}).items():
            if server and server.get("transport"):
                result[name] = McpConfig(name=name, **server)
        return result

# 글로벌 접근 헬퍼 함수
def get_config() -> Dict[str, Any]:
    """메인 설정 딕셔너리를 반환"""
    return ConfigLoader.get_instance().config


def get_agent_config() -> List[AgentConfig]:
    """에이전트 설정 리스트를 반환"""
    return ConfigLoader.get_instance().agent_config


def get_mcp_config() -> Dict[str, McpConfig]:
    """MCP 서버 설정 딕셔너리를 반환"""
    return ConfigLoader.get_instance().mcp_config
