import asyncio
import logging
from typing import List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


async def regen_answer(
    golden_history: list,
    agent_prompt: str,
    agent_cfg: dict
) -> str:
    """Regenerate answer."""
    llm_client = agent_cfg.get("llm_client")
    if not llm_client:
        raise ValueError("llm_client must be provided in agent_cfg")
    
    messages = []
    for turn in golden_history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant":
            messages.append(AIMessage(content=turn["content"]))
    
    messages.insert(0, SystemMessage(content=agent_prompt))
    
    model_cfg = {
        k: v for k, v in agent_cfg.items() 
        if k != "llm_client"
    }
    
    result = await llm_client.call(
        module="regen",
        model_cfg=model_cfg,
        messages=messages,
        ctx={}
    )
    
    regen_answer = result["text"].strip()
    logger.debug(f"Generated assistant response (length: {len(regen_answer)} chars)")
    
    return regen_answer


async def regen_split(
    dialogues: List[dict],
    agent_prompt: str,
    agent_cfg: dict,
    max_concurrent: int = 10
) -> dict:
    """Regenerate split."""
    logger.info(f"Regenerating {len(dialogues)} dialogues (max_concurrent={max_concurrent})")
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def regen_one(dialogue: dict) -> tuple[str, dict]:
        async with semaphore:
            dialogue_id = dialogue.get("dialogue_id", "unknown")
            golden_history = dialogue.get("golden_history", [])
            golden_answer = dialogue.get("golden_answer", "")
            
            regen_answer_text = await regen_answer(
                golden_history,
                agent_prompt,
                agent_cfg
            )
            
            return dialogue_id, {
                "golden_history": golden_history,
                "golden_answer": golden_answer,
                "regen_answer": regen_answer_text
            }
    
    tasks = [regen_one(dlg) for dlg in dialogues]
    results = await asyncio.gather(*tasks)
    
    regen_data = {did: data for did, data in results}
    
    logger.info(f"Completed regeneration: {len(regen_data)} dialogues")
    
    return regen_data
