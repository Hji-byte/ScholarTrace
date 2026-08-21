import time
from collections.abc import Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage


TRANSIENT_ERROR_NAMES = {
    "APIConnectionError",
    "APITimeoutError",
    "ConnectError",
    "InternalServerError",
    "RemoteProtocolError",
    "RateLimitError",
    "ReadTimeout",
}


def is_transient_llm_error(exc: Exception) -> bool:
    names = {type(exc).__name__}
    names.update(type(parent).__name__ for parent in type(exc).__mro__)
    if names & TRANSIENT_ERROR_NAMES:
        return True
    return "connection error" in str(exc).lower()


def invoke_with_retry(
    llm: BaseChatModel,
    messages: Sequence[BaseMessage],
    attempts: int = 3,
    backoff_seconds: float = 5.0,
):
    for attempt in range(1, attempts + 1):
        try:
            return llm.invoke(list(messages))
        except Exception as exc:
            if attempt >= attempts or not is_transient_llm_error(exc):
                raise
            time.sleep(backoff_seconds * attempt)
