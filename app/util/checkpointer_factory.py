"""Checkpointer configuration for state persistence."""
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.redis import RedisSaver
import redis
from typing import Literal, Optional
import os

class CheckpointerFactory:
    """Factory for creating different types of checkpointers."""

    @staticmethod
    def create(
        checkpointer_type: Literal["memory", "redis"] = "memory",
        redis_url: Optional[str] = None
    ):
        """Create a checkpointer based on type.

        Args:
            checkpointer_type: Type of checkpointer ("memory" or "redis")
            redis_url: Redis connection URL (required for redis type)

        Returns:
            Configured checkpointer instance
        """
        if checkpointer_type == "memory":
            return MemorySaver()

        elif checkpointer_type == "redis":
            try:


                if not redis_url:
                    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

                redis_client = redis.from_url(redis_url)
                return RedisSaver(redis_client)

            except ImportError:
                raise ImportError(
                    "Redis checkpointer requires: pip install langgraph-checkpoint-redis redis"
                )

        else:
            raise ValueError(f"Unknown checkpointer type: {checkpointer_type}")
