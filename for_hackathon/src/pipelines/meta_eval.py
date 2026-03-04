import asyncio
import json
import logging
from pathlib import Path

import httpx

from src.core.llm import LLMClient
from src.core.runio import (
    generate_run_id, create_run_path, save_config,
    save_usage_report, save_final_report
)
from src.core.usage import log_usage, build_usage_report, calculate_dataset_tokens, TokenProfiler
from src.core.dataset import (
    load_dataset, validate_dataset, get_split,
    get_dataset_path, prepare_dialogues_for_regen
)
from src.pipelines.selector import Alg as selector_Alg

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


DEFAULT_EVAL_API_URL = "https://hackathon.intellemma.ru"
DEFAULT_EVAL_API_KEY = "NOW6UvFTtf-XM545PO4baX19vRd2T9jN267widVeRGA"
DEFAULT_LLM_API_URL = "https://provider.intellemma.ru/v1"
DEFAULT_LLM_API_KEY = "0726dd7e0fff495120de20e640be4cd341bb7720fdb9d77101e88975e2870e82"
DEFAULT_MODEL_NAME = "openai/gpt-oss-20b"
DEFAULT_DATASET_ID = "mtbench101"
DEFAULT_JUDGE_ID = "golden_semantic_match_v1"
DEFAULT_AGENT_PROMPT = "Ты универсальный AI-ассистент. Отвечай точно, структурированно и по делу. Не выдумывай факты; при нехватке данных задай уточняющий вопрос."


async def metaeval(
    Alg,
    dataset_id: str,
    num_examples: int,
    split: str,
    judge_cfg: dict,
    model_cfg: dict,
    eval_api_url: str,
    eval_api_key: str,
    profiler: TokenProfiler,
    run_path: Path,
    seed: int = 42
) -> dict:
    '''Meta-evaluation pipeline'''
    logger.info("=" * 80)
    logger.info("META-EVALUATION: metaeval(Alg, D, E, M)")
    logger.info("=" * 80)
    
    # Prepare dialogues and calculate dataset size
    logger.info("Preparing dataset and calculating size...")
    dataset_path = get_dataset_path(dataset_id)
    dataset = load_dataset(dataset_path)
    validate_dataset(dataset)
    split_dict = get_split(dataset, split)
    dialogues = prepare_dialogues_for_regen(split_dict, num_examples, seed)
    data_size = calculate_dataset_tokens(dialogues)
    logger.info(f"Dataset size: {data_size} tokens ({len(dialogues)} dialogues)")
    
    # Start profiling
    logger.info("Starting token profiling for Alg...")
    profiler.start()
    
    logger.info("Step 1: Running algorithm Alg(D, E, M) to get improved prompt...")
    prompt, prompt_metadata = await Alg(
        dataset_id=dataset_id,
        num_examples=num_examples,
        split=split,
        judge_cfg=judge_cfg,
        model_cfg=model_cfg,
        seed=seed
    )
    
    # Stop profiling and calculate coefficients
    profiler.stop()
    alg_stats = profiler.get_stats()
    
    # Calculate total used tokens (prompt + completion + cached)
    total_used_tokens = alg_stats["total_tokens"]
    
    if data_size > 0:
        # Usage coefficient: total tokens used by Alg / dataset size
        usage_coefficient = total_used_tokens / data_size
        input_coefficient = alg_stats["total_prompt_tokens"] / data_size
        output_coefficient = alg_stats["total_completion_tokens"] / data_size
    else:
        usage_coefficient = 0.0
        input_coefficient = 0.0
        output_coefficient = 0.0
    
    logger.info(f"Algorithm completed. Prompt length: {len(prompt)}")
    logger.info(f"Alg token usage: {alg_stats['total_prompt_tokens']} prompt, "
                f"{alg_stats['total_completion_tokens']} completion, "
                f"{alg_stats['total_cached_tokens']} cached, "
                f"{total_used_tokens} total")
    logger.info(f"Usage coefficients: usage={usage_coefficient:.6f}, "
                f"input={input_coefficient:.6f}, output={output_coefficient:.6f}")
    
    logger.info("Step 2: Evaluating improved prompt with evaluate(E, D, M, Prompt) via API...")
    
    request_data = {
        "dataset_id": dataset_id,
        "num_examples": num_examples,
        "split": split,
        "seed": seed,
        "judge_id": judge_cfg.get("judge_name"),
        "agent_prompt": prompt,
        "model": {
            "name": model_cfg.get("name"),
            "temperature": model_cfg.get("temperature", 0.7),
            "max_tokens": model_cfg.get("max_tokens", 1000),
        },
        "judge": {
            "name": judge_cfg.get("name"),
            "temperature": judge_cfg.get("temperature", 0.0),
            "max_tokens": judge_cfg.get("max_tokens", 1500),
        }
    }
    
    if "base_url" in model_cfg:
        request_data["model"]["base_url"] = model_cfg["base_url"]
    if "api_key" in model_cfg:
        request_data["model"]["api_key"] = model_cfg["api_key"]
    if "base_url" in judge_cfg:
        request_data["judge"]["base_url"] = judge_cfg["base_url"]
    if "api_key" in judge_cfg:
        request_data["judge"]["api_key"] = judge_cfg["api_key"]
    
    headers = {
        "Content-Type": "application/json"
    }
    if eval_api_key:
        headers["Authorization"] = f"Bearer {eval_api_key}"
    
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            f"{eval_api_url}/evaluate",
            json=request_data,
            headers=headers
        )
        response.raise_for_status()
        api_response = response.json()
    
    evaluation_results = {
        "summary": api_response["summary"],
        "usage": api_response["usage"],
        "regen_data": api_response.get("regen_data", {}),
        "judge_data": api_response.get("judge_data", {})
    }
    
    logger.info(f"Evaluation completed. Mean score: {evaluation_results['summary']['mean']:.2f}")
    
    # Сохраняем regen и judge данные напрямую в meta_eval run
    regen_data = evaluation_results.get("regen_data", {})
    judge_data = evaluation_results.get("judge_data", {})
    
    if regen_data:
        meta_regen_path = run_path / "regen"
        meta_regen_path.mkdir(parents=True, exist_ok=True)
        regen_filepath = meta_regen_path / "eval_regen.json"
        with open(regen_filepath, 'w', encoding='utf-8') as f:
            json.dump(regen_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved regen data to {regen_filepath}")
    
    if judge_data:
        meta_judge_path = run_path / "judge"
        meta_judge_path.mkdir(parents=True, exist_ok=True)
        # Определяем judge_id из конфигурации для имени файла
        judge_id = judge_cfg.get("judge_name", "judge")
        judge_filename = f"eval_{split}__{judge_id}.json"
        judge_filepath = meta_judge_path / judge_filename
        with open(judge_filepath, 'w', encoding='utf-8') as f:
            json.dump(judge_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved judge data to {judge_filepath}")
        
        # Сохраняем summary
        summary_filepath = meta_judge_path / "summary_eval.json"
        with open(summary_filepath, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results["summary"], f, ensure_ascii=False, indent=2)
        logger.info(f"Saved summary to {summary_filepath}")
    
    usage_coefficients = {
        "data_size": data_size,
        "alg_prompt_tokens": alg_stats["total_prompt_tokens"],
        "alg_completion_tokens": alg_stats["total_completion_tokens"],
        "alg_cached_tokens": alg_stats["total_cached_tokens"],
        "alg_total_tokens": alg_stats["total_tokens"],
        "alg_num_calls": alg_stats["num_calls"],
        "usage_coefficient": usage_coefficient,
        "input_coefficient": input_coefficient,
        "output_coefficient": output_coefficient
    }
    
    return {
        "prompt": prompt,
        "prompt_metadata": prompt_metadata,
        "evaluation": evaluation_results,
        "usage_coefficients": usage_coefficients
    }


async def main(config: dict):
    '''Main function for running meta-evaluation'''
    run_id = generate_run_id()
    run_path = create_run_path(run_id)
    save_config(run_path, config)
    logger.info(f"Created run: {run_id}")
    
    # Инициализация TokenProfiler и LLM клиента
    profiler = TokenProfiler()
    
    # Combined usage logger: logs to both file and profiler
    def combined_usage_logger(rec: dict):
        log_usage(run_path, rec)  # Always log to file
        profiler.log(rec)  # Also log to profiler if active
    
    llm_client = LLMClient(combined_usage_logger)
    
    judge_id = config.get("judge_id", "dialogue_similarity_v1")
    logger.info(f"Using judge ID: {judge_id}")
    
    initial_prompt = config.get("agent_prompt", "")
    if not initial_prompt:
        raise ValueError("agent_prompt must be provided in config")
    
    judge_cfg = {
        **config["models"]["judge"],
        "llm_client": llm_client,
        "judge_name": judge_id
    }
    
    model_cfg = {
        **config["models"]["model"],
        "llm_client": llm_client
    }
    
    async def Alg(
        dataset_id: str,
        num_examples: int,
        split: str,
        judge_cfg: dict,
        model_cfg: dict,
        seed: int
    ) -> tuple[str, dict]:
        return await selector_Alg(
            dataset_id=dataset_id,
            num_examples=num_examples,
            split=split,
            judge_cfg=judge_cfg,
            model_cfg=model_cfg,
            agent_prompt_current=initial_prompt,
            llm_client=llm_client,
            seed=seed
        )
    
    logger.info("=" * 80)
    logger.info("META-EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Dataset: {config['dataset_id']}")
    logger.info(f"Split: {config['split']}")
    logger.info(f"Num examples: {config['num_examples']}")
    logger.info(f"Seed: {config.get('seed', 42)}")
    logger.info(f"Judge: {judge_id}")
    logger.info(f"Model: {config['models']['model']['name']}")
    logger.info("=" * 80)
    
    eval_api_url = config.get("eval_api_url", "http://localhost:8000")
    eval_api_key = config.get("eval_api_key")
    
    results = await metaeval(
        Alg=Alg,
        dataset_id=config["dataset_id"],
        num_examples=config["num_examples"],
        split=config["split"],
        judge_cfg=judge_cfg,
        model_cfg=model_cfg,
        eval_api_url=eval_api_url,
        eval_api_key=eval_api_key,
        profiler=profiler,
        run_path=run_path,
        seed=config.get("seed", 42)
    )
    
    logger.info("Saving results...")
    
    from src.core.runio import save_prompt
    save_prompt(run_path, "agent_prompt_improved.txt", results["prompt"])
    
    import json
    eval_filepath = run_path / "meta_eval_results.json"
    with open(eval_filepath, 'w', encoding='utf-8') as f:
        json.dump({
            "prompt_metadata": results["prompt_metadata"],
            "evaluation_summary": results["evaluation"]["summary"]
        }, f, ensure_ascii=False, indent=2)
    
    logger.info("=" * 80)
    logger.info("META-EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Improved prompt length: {len(results['prompt'])}")
    logger.info(f"Total dialogues analyzed: {results['prompt_metadata']['total_dialogues_analyzed']}")
    
    if 'mean_score' in results['prompt_metadata'] or 'min_score' in results['prompt_metadata'] or 'max_score' in results['prompt_metadata']:
        logger.info(f"Algorithm score stats: mean={results['prompt_metadata'].get('mean_score', 0):.2f}, "
                    f"min={results['prompt_metadata'].get('min_score', 0):.2f}, "
                    f"max={results['prompt_metadata'].get('max_score', 0):.2f}")
    
    logger.info("")
    logger.info("Evaluation of improved prompt:")
    eval_summary = results["evaluation"]["summary"]
    logger.info(f"  Judge: {eval_summary['judge_name']}")
    logger.info(f"  Overall Score - Mean: {eval_summary['mean']:.2f}, "
                f"Min: {eval_summary['min']:.2f}, Max: {eval_summary['max']:.2f}")
    logger.info(f"  Count: {eval_summary['count']}")
    logger.info("=" * 80)
    
    logger.info("Building usage report...")
    usage_report = build_usage_report(run_path)
    
    # Add usage coefficients to usage report
    usage_report["alg_profiling"] = results["usage_coefficients"]
    
    save_usage_report(run_path, usage_report)
    
    final_report = {
        "run_id": run_id,
        "config": {
            "dataset_id": config["dataset_id"],
            "num_examples": config["num_examples"],
            "split": config["split"],
            "seed": config.get("seed", 42),
            "judge_id": judge_id,
            "model_name": config["models"]["model"]["name"]
        },
        "prompt_metadata": results["prompt_metadata"],
        "evaluation": results["evaluation"]["summary"],
        "usage": usage_report,
        "eval_usage": results["evaluation"].get("usage", {}),
        "usage_coefficients": results["usage_coefficients"]
    }
    save_final_report(run_path, final_report)
    
    logger.info("Meta-evaluation pipeline completed successfully!")
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Total tokens: {usage_report['total']['total_tokens']}")
    logger.info(f"Total calls: {usage_report['total']['num_calls']}")
    logger.info(f"Alg usage coefficients: usage={results['usage_coefficients']['usage_coefficient']:.6f}, "
                f"input={results['usage_coefficients']['input_coefficient']:.6f}, "
                f"output={results['usage_coefficients']['output_coefficient']:.6f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Meta-Evaluation Pipeline: metaeval(Alg, D, E, M)")
    
    # Основные параметры
    parser.add_argument("--eval-api-url", default=DEFAULT_EVAL_API_URL, help="Base URL of the evaluation API")
    parser.add_argument("--eval-api-key", default=DEFAULT_EVAL_API_KEY, help="Eval API key (for Authorization: Bearer ...)")
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID, help="Dataset ID from datasets_registry.json")
    parser.add_argument("--num-examples", type=int, default=10, help="Number of examples to process")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--judge-id", default=DEFAULT_JUDGE_ID, help="Judge prompt ID from judges.json")
    parser.add_argument("--agent-prompt", default=DEFAULT_AGENT_PROMPT, help="Initial agent prompt text")
    
    # Модель (для генерации ответов)
    model_group = parser.add_argument_group("Model (for answer generation)")
    model_group.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Model name")
    model_group.add_argument("--model-temperature", type=float, default=0.7, help="Model temperature")
    model_group.add_argument("--model-max-tokens", type=int, default=1000, help="Model max tokens")
    model_group.add_argument("--model-base-url", default=DEFAULT_LLM_API_URL, help="Model base URL (for OpenAI-compatible API)")
    model_group.add_argument("--model-api-key", default=DEFAULT_LLM_API_KEY, help="Model API key")
    
    # Judge (для оценки)
    judge_group = parser.add_argument_group("Judge (for evaluation)")
    judge_group.add_argument("--judge-name", default=DEFAULT_MODEL_NAME, help="Judge model name")
    judge_group.add_argument("--judge-temperature", type=float, default=0.0, help="Judge temperature")
    judge_group.add_argument("--judge-max-tokens", type=int, default=1500, help="Judge max tokens")
    judge_group.add_argument("--judge-base-url", default=DEFAULT_LLM_API_URL, help="Judge base URL (for OpenAI-compatible API)")
    judge_group.add_argument("--judge-api-key", default=DEFAULT_LLM_API_KEY, help="Judge API key")
    
    args = parser.parse_args()
    
    config = {
        "eval_api_url": args.eval_api_url,
        "eval_api_key": args.eval_api_key,
        "dataset_id": args.dataset_id,
        "num_examples": args.num_examples,
        "split": args.split,
        "seed": args.seed,
        "judge_id": args.judge_id,
        "agent_prompt": args.agent_prompt,
        "models": {
            "model": {
                "name": args.model_name,
                "temperature": args.model_temperature,
                "max_tokens": args.model_max_tokens,
            },
            "judge": {
                "name": args.judge_name,
                "temperature": args.judge_temperature,
                "max_tokens": args.judge_max_tokens,
            }
        }
    }
    
    if args.model_base_url:
        config["models"]["model"]["base_url"] = args.model_base_url
    if args.model_api_key:
        config["models"]["model"]["api_key"] = args.model_api_key
    if args.judge_base_url:
        config["models"]["judge"]["base_url"] = args.judge_base_url
    if args.judge_api_key:
        config["models"]["judge"]["api_key"] = args.judge_api_key
    
    asyncio.run(main(config))
