from langchain_core.messages import HumanMessage, SystemMessage

from .config import llm


def call_llm(system_prompt: str, user_prompt: str) -> str:
    """Helper to call the LLM with a simple system + human prompt."""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    resp = llm.invoke(messages)
    return resp.content
