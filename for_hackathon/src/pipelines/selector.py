from src.pipelines.improve_algorithm import BaselineAlg
from src.pipelines.submission import MyAlg


async def Alg(
    dataset_id: str,
    num_examples: int,
    split: str,
    judge_cfg: dict,
    model_cfg: dict,
    agent_prompt_current: str,
    llm_client,
    seed: int = 42
) -> tuple[str, dict]:
    # Switch algorithms here: replace BaselineAlg with MyAlg or any other algorithm
    return await BaselineAlg(
        dataset_id=dataset_id,
        num_examples=num_examples,
        split=split,
        judge_cfg=judge_cfg,
        model_cfg=model_cfg,
        agent_prompt_current=agent_prompt_current,
        llm_client=llm_client,
        seed=seed
    )
