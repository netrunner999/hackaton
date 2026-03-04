import json
import logging
import random
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

_DATASETS_REGISTRY_PATH = "data/datasets_registry.json"


def load_dataset(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def validate_dataset(ds: dict) -> None:
    required_splits = ['train', 'val']
    optional_splits = ['test']
    
    for split in required_splits:
        if split not in ds:
            raise ValueError(f"Missing required split: {split}")
        if not isinstance(ds[split], dict):
            raise ValueError(f"Split '{split}' must be a dict")
    
    for split_name, split_data in ds.items():
        if not isinstance(split_data, dict):
            raise ValueError(f"Split '{split_name}' must be a dict")
        
        for dialogue_id, dialogue_data in split_data.items():
            if not isinstance(dialogue_data, dict):
                raise ValueError(
                    f"Dialogue '{dialogue_id}' in split '{split_name}' must be a dict"
                )
            
            if 'turns' not in dialogue_data:
                raise ValueError(
                    f"Dialogue '{dialogue_id}' in split '{split_name}' missing 'turns'"
                )
            
            turns = dialogue_data['turns']
            if not isinstance(turns, list):
                raise ValueError(
                    f"Dialogue '{dialogue_id}' in split '{split_name}': 'turns' must be a list"
                )
            
            for i, turn in enumerate(turns):
                if not isinstance(turn, dict):
                    raise ValueError(
                        f"Dialogue '{dialogue_id}' in split '{split_name}': "
                        f"turn {i} must be a dict"
                    )
                
                if 'role' not in turn:
                    raise ValueError(
                        f"Dialogue '{dialogue_id}' in split '{split_name}': "
                        f"turn {i} missing 'role'"
                    )
                
                if turn['role'] not in ['user', 'assistant']:
                    raise ValueError(
                        f"Dialogue '{dialogue_id}' in split '{split_name}': "
                        f"turn {i} has invalid role '{turn['role']}', "
                        f"must be 'user' or 'assistant'"
                    )
                
                if 'content' not in turn:
                    raise ValueError(
                        f"Dialogue '{dialogue_id}' in split '{split_name}': "
                        f"turn {i} missing 'content'"
                    )
                
                if not isinstance(turn['content'], str):
                    raise ValueError(
                        f"Dialogue '{dialogue_id}' in split '{split_name}': "
                        f"turn {i} 'content' must be a string"
                    )


def get_split(ds: dict, split: str) -> Dict[str, dict]:
    if split not in ds:
        raise ValueError(f"Split '{split}' not found in dataset")
    return ds[split]


def load_datasets_registry(path: str = None) -> dict:
    """Load datasets registry."""
    if path is None:
        path = _DATASETS_REGISTRY_PATH
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_dataset_path(dataset_id: str) -> str:
    """Get dataset path by ID."""
    registry = load_datasets_registry()
    if dataset_id not in registry:
        raise ValueError(f"Dataset '{dataset_id}' not found in registry")
    return registry[dataset_id]["path"]


def prepare_dialogues_for_regen(split_dict: dict, num_examples: int, seed: int = 42) -> list:
    """Prepare dialogues for regeneration."""
    dialogue_ids = list(split_dict.keys())
    
    if num_examples > len(dialogue_ids):
        logger.warning(
            f"Requested {num_examples} dialogues but only {len(dialogue_ids)} available. "
            f"Using all {len(dialogue_ids)} dialogues."
        )
        num_examples = len(dialogue_ids)
    
    random.seed(seed)
    selected_ids = random.sample(dialogue_ids, num_examples)
    
    dialogues = []
    for dialogue_id in selected_ids:
        golden_turns = split_dict[dialogue_id]["turns"]
        
        last_user_idx = -1
        for i in range(len(golden_turns) - 1, -1, -1):
            if golden_turns[i].get("role") == "user":
                last_user_idx = i
                break
        
        if last_user_idx == -1:
            logger.warning(f"Dialogue {dialogue_id}: No user turn found, skipping")
            continue
        
        last_assistant_idx = -1
        for i in range(len(golden_turns) - 1, last_user_idx, -1):
            if golden_turns[i].get("role") == "assistant":
                last_assistant_idx = i
                break
        
        golden_history = golden_turns[:last_assistant_idx] if last_assistant_idx != -1 else golden_turns[:last_user_idx + 1]
        
        golden_answer = ""
        if last_assistant_idx != -1:
            golden_answer = golden_turns[last_assistant_idx].get("content", "")
        
        dialogues.append({
            "dialogue_id": dialogue_id,
            "golden_history": golden_history,
            "golden_answer": golden_answer
        })
    
    return dialogues
