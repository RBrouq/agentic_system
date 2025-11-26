import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load env vars if running locally
load_dotenv()

# Global LLM config
llm = ChatOpenAI(
    model="gpt-4.1-mini",   # or gpt-4o-mini / gpt-4.1 depending on your access
    temperature=0.4,
)

# Graph / checkpoints config
CHECKPOINTS_DB = os.getenv("CHECKPOINTS_DB", "checkpoints.sqlite")
DEFAULT_RECURSION_LIMIT = 50
