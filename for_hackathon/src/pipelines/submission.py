import asyncio
import json
import re
from statistics import mean
from typing import Any, Dict, List, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from src.core.dataset import (
    get_dataset_path,
    get_split,
    load_dataset,
    prepare_dialogues_for_regen,
    validate_dataset,
)


async def _llm_text(
    llm_client,
    module: str,
    model_cfg: dict,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    cfg = {
        "name": model_cfg.get("name"),
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if "base_url" in model_cfg:
        cfg["base_url"] = model_cfg["base_url"]
    if "api_key" in model_cfg:
        cfg["api_key"] = model_cfg["api_key"]

    result = await llm_client.call(
        module=module,
        model_cfg=cfg,
        messages=[
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ],
        ctx={},
    )
    return result["text"].strip()


def _format_history(turns: List[Dict[str, str]]) -> str:
    return "\n".join(f"{t.get('role')}: {t.get('content', '')}" for t in turns)


def _build_examples_block(dialogues: List[dict], limit: int = 20) -> str:
    chunks = []
    for i, dlg in enumerate(dialogues[:limit], start=1):
        chunks.append(
            f"[EXAMPLE {i}]\n"
            f"HISTORY:\n{_format_history(dlg.get('golden_history', []))}\n\n"
            f"GOLDEN_ASSISTANT:\n{dlg.get('golden_answer', '')}"
        )
    return "\n\n".join(chunks)


def _extract_score(text: str) -> float:
    match = re.search(r"(?:score|балл)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
    if match:
        return float(match.group(1))

    numbers = re.findall(r"[0-9]+(?:\.[0-9]+)?", text)
    if numbers:
        return float(numbers[0])

    return 0.0


async def _generate_candidates(
    llm_client,
    model_cfg: dict,
    agent_prompt_current: str,
    judge_name: str,
    examples_block: str,
) -> List[str]:
    generator_prompt = f"""
Создай 4 варианта system prompt для AI-ассистента.

Контекст:
- Нужно обобщаться на новые датасеты и новые эвалюаторы.
- Текущий prompt:
---
{agent_prompt_current}
---
- Judge ID / evaluator hint: {judge_name}

Ниже примеры из train, где GOLDEN_ASSISTANT — целевой стиль:
{examples_block}

Требования к каждому кандидату:
1) Краткость и ясность, без лишней "воды".
2) Инструкции по точности, полезности, структурированности и безопасным отказам.
3) Явное правило: не выдумывать факты, если не хватает данных — задавать уточнения.
4) Универсальность под разные домены.

Верни СТРОГО JSON объект формата:
{{
  "candidates": ["prompt1", "prompt2", "prompt3", "prompt4"]
}}
""".strip()

    raw = await _llm_text(
        llm_client=llm_client,
        module="alg_candidate_gen",
        model_cfg=model_cfg,
        system_prompt="Ты senior prompt engineer. Отвечай строго по формату.",
        user_prompt=generator_prompt,
        max_tokens=min(3000, int(model_cfg.get("max_tokens", 3000))),
        temperature=0.7,
    )

    try:
        parsed = json.loads(raw)
        candidates = [c.strip() for c in parsed.get("candidates", []) if isinstance(c, str) and c.strip()]
    except Exception:
        candidates = []

    if not candidates:
        candidates = [agent_prompt_current]

    return candidates[:4]


async def _proxy_score_candidate(
    llm_client,
    model_cfg: dict,
    judge_name: str,
    candidate_prompt: str,
    dialogue: dict,
) -> float:
    history = dialogue.get("golden_history", [])
    golden_answer = dialogue.get("golden_answer", "")

    generated = await _llm_text(
        llm_client=llm_client,
        module="alg_proxy_regen",
        model_cfg=model_cfg,
        system_prompt=candidate_prompt,
        user_prompt=_format_history(history),
        max_tokens=min(900, int(model_cfg.get("max_tokens", 1000))),
        temperature=0.2,
    )

    judge_prompt = f"""
Ты оцениваешь качество ответа assistant относительно эталона.

Evaluator hint: {judge_name}

Диалог (история):
{_format_history(history)}

Эталонный ответ:
{golden_answer}

Кандидатный ответ:
{generated}

Оцени по шкале 0..10 по критериям:
- семантическое совпадение с эталоном,
- полезность для пользователя,
- отсутствие галлюцинаций,
- уместный тон.

Верни ОДНУ строку в формате: SCORE: <число>
""".strip()

    verdict = await _llm_text(
        llm_client=llm_client,
        module="alg_proxy_judge",
        model_cfg=model_cfg,
        system_prompt="Ты строгий, но справедливый оценщик диалогов.",
        user_prompt=judge_prompt,
        max_tokens=100,
        temperature=0.0,
    )

    return max(0.0, min(10.0, _extract_score(verdict)))


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
    dataset_path = get_dataset_path(dataset_id)
    dataset = load_dataset(dataset_path)
    validate_dataset(dataset)

    split_dict = get_split(dataset, split)
    dialogues = prepare_dialogues_for_regen(split_dict, num_examples, seed)
    if not dialogues:
        return agent_prompt_current, {"total_dialogues_analyzed": 0, "reason": "no_dialogues"}

    examples_block = _build_examples_block(dialogues, limit=20)
    judge_name = judge_cfg.get("judge_name", "unknown_judge")

    candidates = await _generate_candidates(
        llm_client=llm_client,
        model_cfg=model_cfg,
        agent_prompt_current=agent_prompt_current,
        judge_name=judge_name,
        examples_block=examples_block,
    )

    eval_set_size = min(6, len(dialogues))
    eval_dialogues = dialogues[:eval_set_size]

    candidate_scores: List[Dict[str, Any]] = []
    for cand in candidates:
        scores = await asyncio.gather(
            *[
                _proxy_score_candidate(
                    llm_client=llm_client,
                    model_cfg=model_cfg,
                    judge_name=judge_name,
                    candidate_prompt=cand,
                    dialogue=dlg,
                )
                for dlg in eval_dialogues
            ]
        )
        candidate_scores.append({"prompt": cand, "scores": scores, "mean": mean(scores) if scores else 0.0})

    best = max(candidate_scores, key=lambda x: x["mean"])

    metadata = {
        "total_dialogues_analyzed": len(dialogues),
        "proxy_eval_dialogues": eval_set_size,
        "num_candidates": len(candidates),
        "mean_score": best["mean"],
        "all_candidate_scores": [round(c["mean"], 4) for c in candidate_scores],
    }

    return best["prompt"], metadata
