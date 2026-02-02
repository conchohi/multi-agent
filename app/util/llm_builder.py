import os
from functools import cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama

from app.util.config_loader import get_config
from app.model.settings import LLMConfig


@cache
def _get_llm_config() -> LLMConfig:
    config = get_config()
    return LLMConfig(**config.get('llm', {}))


def _build_openai(llm_config: LLMConfig) -> ChatOpenAI:
    if not os.environ.get('OPENAI_API_KEY'):
        raise ValueError('OPENAI_API_KEY not set in environment')
    return ChatOpenAI(
        model=llm_config.model,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
        top_p=llm_config.top_p,
        frequency_penalty=llm_config.frequency_penalty,
    )


def _build_anthropic(llm_config: LLMConfig) -> ChatAnthropic:
    if not os.environ.get('ANTHROPIC_API_KEY'):
        raise ValueError('ANTHROPIC_API_KEY not set in environment')
    return ChatAnthropic(
        model=llm_config.model,
        temperature=llm_config.temperature,
        max_tokens=llm_config.max_tokens,
        top_p=llm_config.top_p,
    )


def _build_ollama(llm_config: LLMConfig) -> ChatOllama:
    ollama_host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')
    return ChatOllama(
        model=llm_config.model,
        temperature=llm_config.temperature,
        num_predict=llm_config.max_tokens,
        top_p=llm_config.top_p,
        base_url=ollama_host,
    )


def _dispatch(llm_config: LLMConfig) -> BaseChatModel:
    if llm_config.provider == 'openai':
        return _build_openai(llm_config)
    elif llm_config.provider == 'anthropic':
        return _build_anthropic(llm_config)
    elif llm_config.provider == 'ollama':
        return _build_ollama(llm_config)
    raise ValueError(f"Unsupported LLM provider: '{llm_config.provider}'")


def build_llm() -> BaseChatModel:
    return _dispatch(_get_llm_config())


def build_agent_llm(agent_llm_config: LLMConfig) -> BaseChatModel:
    merged = _get_llm_config().model_copy(update=agent_llm_config.model_dump(exclude_none=True))
    return _dispatch(merged)
