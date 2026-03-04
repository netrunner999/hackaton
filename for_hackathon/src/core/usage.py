import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import tiktoken

logger = logging.getLogger(__name__)


def log_usage(run_path: Path, record: dict) -> None:
    log_path = run_path / "usage" / "usage_log.jsonl"
    
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    except Exception as e:
        logger.error(f"Failed to write usage log: {e}")


def load_usage_log(run_path: Path) -> List[dict]:
    log_path = run_path / "usage" / "usage_log.jsonl"
    
    if not log_path.exists():
        return []
    
    records = []
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    except Exception as e:
        logger.error(f"Failed to load usage log: {e}")
    
    return records


def build_usage_report_from_records(records: List[dict]) -> dict:
    """Build usage report from records list (in-memory)."""
    if not records:
        return {
            "total": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cached_tokens": 0,
                "latency_ms": 0,
                "num_calls": 0
            },
            "by_module": {},
            "by_model": {},
            "by_iter": {},
            "by_split": {}
        }
    
    total = defaultdict(int)
    by_module = defaultdict(lambda: defaultdict(int))
    by_model = defaultdict(lambda: defaultdict(int))
    by_iter = defaultdict(lambda: defaultdict(int))
    by_split = defaultdict(lambda: defaultdict(int))
    
    for record in records:
        total["prompt_tokens"] += record.get("prompt_tokens", 0)
        total["completion_tokens"] += record.get("completion_tokens", 0)
        total["total_tokens"] += record.get("total_tokens", 0)
        total["cached_tokens"] += record.get("cached_tokens", 0)
        total["latency_ms"] += record.get("latency_ms", 0)
        total["num_calls"] += 1
        
        module = record.get("module", "unknown")
        by_module[module]["prompt_tokens"] += record.get("prompt_tokens", 0)
        by_module[module]["completion_tokens"] += record.get("completion_tokens", 0)
        by_module[module]["total_tokens"] += record.get("total_tokens", 0)
        by_module[module]["cached_tokens"] += record.get("cached_tokens", 0)
        by_module[module]["latency_ms"] += record.get("latency_ms", 0)
        by_module[module]["num_calls"] += 1
        
        model = record.get("model", "unknown")
        by_model[model]["prompt_tokens"] += record.get("prompt_tokens", 0)
        by_model[model]["completion_tokens"] += record.get("completion_tokens", 0)
        by_model[model]["total_tokens"] += record.get("total_tokens", 0)
        by_model[model]["cached_tokens"] += record.get("cached_tokens", 0)
        by_model[model]["latency_ms"] += record.get("latency_ms", 0)
        by_model[model]["num_calls"] += 1
        
        iter_idx = record.get("iter_idx")
        if iter_idx is not None:
            by_iter[iter_idx]["prompt_tokens"] += record.get("prompt_tokens", 0)
            by_iter[iter_idx]["completion_tokens"] += record.get("completion_tokens", 0)
            by_iter[iter_idx]["total_tokens"] += record.get("total_tokens", 0)
            by_iter[iter_idx]["cached_tokens"] += record.get("cached_tokens", 0)
            by_iter[iter_idx]["latency_ms"] += record.get("latency_ms", 0)
            by_iter[iter_idx]["num_calls"] += 1
        
        split = record.get("split")
        if split:
            by_split[split]["prompt_tokens"] += record.get("prompt_tokens", 0)
            by_split[split]["completion_tokens"] += record.get("completion_tokens", 0)
            by_split[split]["total_tokens"] += record.get("total_tokens", 0)
            by_split[split]["cached_tokens"] += record.get("cached_tokens", 0)
            by_split[split]["latency_ms"] += record.get("latency_ms", 0)
            by_split[split]["num_calls"] += 1
    
    def to_dict(d):
        if isinstance(d, defaultdict):
            return {k: to_dict(v) for k, v in d.items()}
        return d
    
    return {
        "total": dict(total),
        "by_module": to_dict(by_module),
        "by_model": to_dict(by_model),
        "by_iter": to_dict(by_iter),
        "by_split": to_dict(by_split)
    }


def build_usage_report(run_path: Path) -> dict:
    """Build usage report from run_path (loads from file)."""
    records = load_usage_log(run_path)
    return build_usage_report_from_records(records)


def calculate_dataset_tokens(dialogues: List[dict]) -> int:
    """
    Calculate total tokens in dataset dialogues using tiktoken.

    In restricted environments, loading tokenizer files may require network.
    If tokenizer loading fails, uses a deterministic char-based approximation
    (1 token ~= 4 chars) to keep pipeline runnable.
    """
    encoding = None
    for encoding_name in ("cl100k_base", "gpt2"):
        try:
            encoding = tiktoken.get_encoding(encoding_name)
            break
        except Exception as e:
            logger.warning(f"Failed to load tiktoken encoding '{encoding_name}': {e}")

    total_tokens = 0

    for dialogue in dialogues:
        golden_history = dialogue.get("golden_history", [])
        for turn in golden_history:
            content = turn.get("content", "")
            if not content:
                continue
            if encoding is not None:
                total_tokens += len(encoding.encode(content))
            else:
                total_tokens += max(1, len(content) // 4)

        golden_answer = dialogue.get("golden_answer", "")
        if golden_answer:
            if encoding is not None:
                total_tokens += len(encoding.encode(golden_answer))
            else:
                total_tokens += max(1, len(golden_answer) // 4)

    return total_tokens


class TokenProfiler:
    """Profiler for tracking token usage during algorithm execution."""
    
    def __init__(self):
        self.is_profiling = False
        self.records: List[dict] = []
    
    def start(self):
        """Start profiling token usage."""
        self.is_profiling = True
        self.records = []
    
    def stop(self):
        """Stop profiling token usage."""
        self.is_profiling = False
    
    def log(self, record: dict):
        """Log a usage record if profiling is active."""
        if self.is_profiling:
            self.records.append(record)
    
    def get_stats(self) -> dict:
        """
        Get aggregated statistics from profiled records.
        
        Returns:
            dict with total_prompt_tokens, total_completion_tokens, total_cached_tokens,
            total_tokens (sum of all token types)
        """
        total_prompt = 0
        total_completion = 0
        total_cached = 0
        
        for record in self.records:
            total_prompt += record.get("prompt_tokens", 0)
            total_completion += record.get("completion_tokens", 0)
            total_cached += record.get("cached_tokens", 0)
        
        # Total tokens includes all types: prompt + completion + cached
        total_tokens = total_prompt + total_completion + total_cached
        
        return {
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_cached_tokens": total_cached,
            "total_tokens": total_tokens,
            "num_calls": len(self.records)
        }
