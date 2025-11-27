import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# --------- Charger .env explicitement à la racine du projet ---------
ROOT_DIR = Path(__file__).resolve().parent.parent  # .../agentic_system
env_path = ROOT_DIR / ".env"
load_dotenv(env_path)

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError(
        f"OPENAI_API_KEY is not set. "
        f"Vérifie ton fichier .env à la racine : {env_path}"
    )

# Debug léger (sans afficher toute la clé)
print("[config] OPENAI_API_KEY loaded, length =", len(api_key))
print("[config] OPENAI_API_KEY starts with:", api_key[:7] + "...")

# --------- LLM global ---------
llm = ChatOpenAI(
    model="gpt-5.1",   # ou gpt-4o-mini / gpt-4.1
    temperature=0.4,
    api_key=api_key,        # <--- on force explicitement la clé ici
)

# Graph / checkpoints config
CHECKPOINTS_DB = os.getenv("CHECKPOINTS_DB", "checkpoints.sqlite")
DEFAULT_RECURSION_LIMIT = 50
