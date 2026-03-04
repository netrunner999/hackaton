from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from src.core.dataset import (
    get_dataset_path,
    get_split,
    load_dataset,
    prepare_dialogues_for_regen,
    validate_dataset,
)


def _format_short_examples(dialogues: List[dict], limit: int = 4) -> str:
    """Build compact examples block to avoid oversized prompts."""
    blocks: List[str] = []
    for i, dlg in enumerate(dialogues[:limit], start=1):
        history = dlg.get("golden_history", [])
        last_user = ""
        for turn in reversed(history):
            if turn.get("role") == "user":
                last_user = turn.get("content", "")
                break

        golden_answer = dlg.get("golden_answer", "")
        blocks.append(
            f"[EXAMPLE {i}]\n"
            f"USER_LAST: {last_user}\n"
            f"TARGET_ASSISTANT_STYLE: {golden_answer}"
        )
    return "\n\n".join(blocks)


async def _improve_prompt_once(
    llm_client,
    model_cfg: dict,
    current_prompt: str,
    judge_name: str,
    examples_block: str,
) -> str:
    improver_text = f"""
Ты senior prompt engineer. Улучши system prompt ассистента.

Цели:
- стабильное качество на разных датасетах и разных judge/evaluator;
- точность, полезность, структурированность;
- запрет на галлюцинации: если данных не хватает — явно сообщать и задавать уточнение.

Evaluator hint: {judge_name}

Текущий prompt:
---
{current_prompt}
---

Короткие примеры целевого поведения:
{examples_block}

Верни ТОЛЬКО итоговый улучшенный system prompt (без пояснений).
""".strip()

    cfg = {
        "name": model_cfg.get("name"),
        "temperature": 0.3,
        "max_tokens": min(700, int(model_cfg.get("max_tokens", 1000))),
    }
    if "base_url" in model_cfg:
        cfg["base_url"] = model_cfg["base_url"]
    if "api_key" in model_cfg:
        cfg["api_key"] = model_cfg["api_key"]

    result = await llm_client.call(
        module="alg_improve_single_pass",
        model_cfg=cfg,
        messages=[
            SystemMessage(content="Ты эксперт по проектированию промптов."),
            HumanMessage(content=improver_text),
        ],
        ctx={},
    )
    return result["text"].strip()


async def MyAlg(
    dataset_id: str,
    num_examples: int,
    split: str,
    judge_cfg: dict,
    model_cfg: dict,
    agent_prompt_current: str,
    llm_client,
    seed: int = 42,
) -> Tuple[str, Dict[str, Any]]:
    """
    Robust prompt improver:
    - compact context (few examples)
    - single LLM pass (no expensive noisy proxy loop)
    - strict fallback to current prompt on empty/weak output
    """
    dataset_path = get_dataset_path(dataset_id)
    dataset = load_dataset(dataset_path)
    validate_dataset(dataset)

    split_dict = get_split(dataset, split)
    dialogues = prepare_dialogues_for_regen(split_dict, num_examples, seed)
    if not dialogues:
        return agent_prompt_current, {
            "total_dialogues_analyzed": 0,
            "fallback_used": True,
            "reason": "no_dialogues",
        }

    examples_block = _format_short_examples(dialogues, limit=min(4, len(dialogues)))
    judge_name = judge_cfg.get("judge_name", "unknown_judge")

    candidate = await _improve_prompt_once(
        llm_client=llm_client,
        model_cfg=model_cfg,
        current_prompt=agent_prompt_current,
        judge_name=judge_name,
        examples_block=examples_block,
    )

    fallback_used = False
    # Guardrails: empty/too short prompt is considered invalid
    if not candidate or len(candidate.strip()) < 40:
        candidate = agent_prompt_current
        fallback_used = True

    metadata = {
        "total_dialogues_analyzed": len(dialogues),
        "num_examples_in_prompt": min(4, len(dialogues)),
        "fallback_used": fallback_used,
        "improved_prompt_length": len(candidate),
    }

    return candidate, metadata
