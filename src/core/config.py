"""YAML and Environment configuration loader."""
import os
from pathlib import Path
import yaml
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"

with open(CONFIG_PATH, "r") as f:
    CONFIG = yaml.safe_load(f)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY") or CONFIG.get("apis", {}).get("alpha_vantage_key", "")


def get_openai_api_key() -> str | None:
    """Read the current key so Streamlit runtime configuration takes effect."""
    return os.getenv("OPENAI_API_KEY") or OPENAI_API_KEY


def set_runtime_api_keys(openai_api_key: str = "", tavily_api_key: str = "") -> None:
    """Update optional runtime keys without writing secrets to project files."""
    for env_name, value in {
        "OPENAI_API_KEY": openai_api_key,
        "TAVILY_API_KEY": tavily_api_key,
    }.items():
        if value:
            os.environ[env_name] = value
