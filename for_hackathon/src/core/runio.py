import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


def generate_run_id() -> str:
    now = datetime.now()
    return f"run_{now.strftime('%Y%m%d_%H%M%S')}"


def create_run_path(run_id: str, base_dir: str = "runs") -> Path:
    run_path = Path(base_dir) / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    
    (run_path / "prompts").mkdir(exist_ok=True)
    (run_path / "regen").mkdir(exist_ok=True)
    (run_path / "judge").mkdir(exist_ok=True)
    (run_path / "usage").mkdir(exist_ok=True)
    
    return run_path


def save_config(run_path: Path, config: dict) -> None:
    with open(run_path / "config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def save_selected_ids(
    run_path: Path,
    train_ids: List[str],
    val_ids: List[str],
    seed: int,
    k_train: int
) -> None:
    data = {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "seed": seed,
        "k_train": k_train,
        "k_val": len(val_ids)
    }
    with open(run_path / "selected_ids.json", 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_prompt(run_path: Path, prompt_name: str, content: str) -> None:
    prompt_path = run_path / "prompts" / prompt_name
    prompt_path.write_text(content, encoding='utf-8')


def load_prompt(run_path: Path, prompt_name: str) -> str:
    prompt_path = run_path / "prompts" / prompt_name
    return prompt_path.read_text(encoding='utf-8')


def save_regen(
    run_path: Path,
    iter_idx: int,
    split: str,
    regen_data: dict
) -> None:
    filename = f"iter{iter_idx}_{split}_regen.json"
    filepath = run_path / "regen" / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(regen_data, f, ensure_ascii=False, indent=2)


def load_regen(run_path: Path, iter_idx: int, split: str) -> dict:
    filename = f"iter{iter_idx}_{split}_regen.json"
    filepath = run_path / "regen" / filename
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_judge(
    run_path: Path,
    iter_idx: int,
    split: str,
    judge_name: str,
    judge_data: dict
) -> None:
    filename = f"iter{iter_idx}_{split}__{judge_name}.json"
    filepath = run_path / "judge" / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(judge_data, f, ensure_ascii=False, indent=2)


def load_judge(
    run_path: Path,
    iter_idx: int,
    split: str,
    judge_name: str
) -> dict:
    filename = f"iter{iter_idx}_{split}__{judge_name}.json"
    filepath = run_path / "judge" / filename
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_summary(run_path: Path, iter_idx: int, summary: dict) -> None:
    filename = f"summary_iter{iter_idx}.json"
    filepath = run_path / "judge" / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def append_history(run_path: Path, record: dict) -> None:
    history_path = run_path / "history.jsonl"
    with open(history_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def save_usage_report(run_path: Path, report: dict) -> None:
    filepath = run_path / "usage" / "usage_report.json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def save_final_report(run_path: Path, report: dict) -> None:
    filepath = run_path / "final_report.json"
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
