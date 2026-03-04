import random

from langchain_core.messages import HumanMessage, SystemMessage

from src.core.dataset import (
    load_dataset, validate_dataset, get_split,
    get_dataset_path, prepare_dialogues_for_regen
)


async def BaselineAlg(
    dataset_id: str,
    num_examples: int,
    split: str,
    judge_cfg: dict,
    model_cfg: dict,
    agent_prompt_current: str,
    llm_client,
    seed: int = 42
) -> tuple[str, dict]:
    """
    Baseline algorithm for prompt improvement: Alg(D, E, M) = Prompt
    
    Simple baseline: selects 1 random dialogue and improves prompt based on it.
    """
    dataset_path = get_dataset_path(dataset_id)
    dataset = load_dataset(dataset_path)
    validate_dataset(dataset)
    
    split_dict = get_split(dataset, split)
    dialogues = prepare_dialogues_for_regen(split_dict, num_examples, seed)
    
    # Select 1 random dialogue
    random.seed(seed)
    selected_dialogue = random.choice(dialogues)
    
    # Prepare dialogue text
    golden_history_text = "\n".join(
        f"{turn.get('role')}: {turn.get('content')}"
        for turn in selected_dialogue.get("golden_history", [])
    )
    
    all_dialogues_text = f"""
История диалога:
{golden_history_text}
""".strip()
    
    improver_prompt_text = """Ты опытный инженер промптов. Улучши промпт AI-ассистента на основе анализа диалога.

Текущий промпт:
{agent_prompt_current}

Диалог:
{all_dialogues}

Проанализируй паттерны, найди проблемы, добавь конкретные инструкции для их избежания. Сохрани хорошие аспекты промпта. Верни улучшенный промпт без комментариев."""
    
    formatted_prompt = improver_prompt_text.format(
        agent_prompt_current=agent_prompt_current,
        all_dialogues=all_dialogues_text,
    )
    
    messages = [
        SystemMessage(content="Ты эксперт по улучшению промптов для AI-ассистентов."),
        HumanMessage(content=formatted_prompt),
    ]
    
    model_cfg_improver = {
        "name": model_cfg.get("name"),
        "temperature": model_cfg.get("temperature", 0.5),
        "max_tokens": model_cfg.get("max_tokens", 2000),
    }
    if "base_url" in model_cfg:
        model_cfg_improver["base_url"] = model_cfg["base_url"]
    if "api_key" in model_cfg:
        model_cfg_improver["api_key"] = model_cfg["api_key"]
    
    ctx = {}
    
    result = await llm_client.call(
        module="improver",
        model_cfg=model_cfg_improver,
        messages=messages,
        ctx=ctx,
    )
    
    new_prompt = result["text"].strip()
    
    metadata = {
        "total_dialogues_analyzed": 1,
        "dialogue_id": selected_dialogue.get("dialogue_id", "unknown"),
        "improved_prompt_length": len(new_prompt),
    }
    
    return new_prompt, metadata

