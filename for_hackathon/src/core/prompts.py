import json
import logging
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)


def load_judges_registry(path: str) -> Dict[str, dict]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Failed to load judges registry from {path}: {e}")
        raise


def get_judge_prompt(judge_id: str, registry_path: str = "prompts/judges.json") -> str:
    """Get judge prompt by ID."""
    registry = load_judges_registry(registry_path)
    
    if judge_id not in registry:
        raise KeyError(f"Judge '{judge_id}' not found in registry")
    
    judge_data = registry[judge_id]
    if isinstance(judge_data, dict):
        return judge_data.get("prompt", "")
    elif isinstance(judge_data, str):
        return judge_data
    else:
        raise ValueError(f"Invalid format for judge '{judge_id}'")


def load_text_prompt(path: str) -> str:
    try:
        return Path(path).read_text(encoding='utf-8')
    except Exception as e:
        logger.error(f"Failed to load prompt from {path}: {e}")
        raise
