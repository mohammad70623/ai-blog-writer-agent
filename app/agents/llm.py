from functools import lru_cache

from langchain_groq import ChatGroq

from app.core.config import GROQ_API_KEY, GROQ_MODEL


@lru_cache(maxsize=1)
def get_llm() -> ChatGroq:
    """
    Returns a cached ChatGroq instance.
    Using lru_cache so we don't recreate the client on every node call.
    """
    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. "
            "Add it to your .env file or environment variables."
        )
    return ChatGroq(
        model=GROQ_MODEL,
        api_key=GROQ_API_KEY,
        temperature=0.4,
    )
