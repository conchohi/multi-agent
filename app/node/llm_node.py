from pathlib import Path
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts.chat import ChatPromptTemplate

from app.util.logger import get_logger

logger = get_logger(__name__)

class LLMNode:
    def __init__(self, name: str, llm: BaseChatModel, prompt_file: str = None):
        self.name = name
        self.llm = llm
        self.prompt_file = prompt_file
        self.prompt = ""
        self.prompt_template = ""
        self._load_prompt()
        
    def _load_prompt(self) -> ChatPromptTemplate:
        if not self.prompt_file:
            logger.debug(f"[{self.name}] No prompt file specified, using default")
            # 기본 프롬프트 파일 시도
            self.prompt_file = f'prompts/{self.name}_prompt.txt'
        
        try:
            prompt_path = Path(self.prompt_file)
            if not prompt_path.exists():
                logger.warning(f"[{self.name}] Prompt file not found: {self.prompt_file}")
                return None

            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
                logger.debug(f"[{self.name}] Loaded prompt from {self.prompt_file}")
                self.prompt = prompt
                self.prompt_template = ChatPromptTemplate.from_template(prompt)

        except Exception as e:
            logger.error(f"[{self.name}] Failed to load prompt file: {e}")
            return None